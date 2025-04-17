'''
Healthcare Worker (HCW) Processing Module (V4)
|_ @author Sidharth Raghavan
|_ Processes all HCW data from 2016-2023
'''

# import packages
import pandas as pd
import os
import networkx as nx
from datetime import datetime, timedelta
import numpy as np
import csv
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import logging
import traceback
import psutil

def setup_logging(year):
    """
    logging to find errors
    """
    log_dir = f'hcw_logs_{year}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = f"{log_dir}/process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def get_mdro_status(mrn, start_date, end_date, mdro_data):
    """
    processes mdro data for a given MRN and date range
    """
    # MDRO categories
    mdro_categories = [
        'VRE',
        'MDR-Pseudomonas',
        'CRE',
        'MDR-Acinetobacter',
        'ESBL',
        'Non-MDRO',
        'MRSA'
    ]
    
    # initialize MDRO vector with zeros
    mdro_vector = {f"mdro_{cat}": 0 for cat in mdro_categories}
    mdro_vector["specimen_time"] = ""
    
    # check if patient has MDRO data
    if mrn not in mdro_data:
        return mdro_vector
    
    # access MDRO records
    patient_mdro = mdro_data[mrn]
    
    # filter records before end_date
    valid_records = [record for record in patient_mdro 
                    if record['specimen_taken_time'].date() <= end_date]
    
    if not valid_records:
        return mdro_vector
    
    # find the latest specimen time before end_date
    latest_record = max(valid_records, key=lambda x: x['specimen_taken_time'])
    specimen_time = latest_record['specimen_taken_time']
    mdro_vector["specimen_time"] = str(specimen_time)
    
    # case 1: pt with specimen_taken_time before the window start_time
    if specimen_time.date() < start_date:
        # find positive MDROs
        for record in valid_records:
            if record['MDRO_Category']:
                mdro_key = f"mdro_{record['MDRO_Category']}"
                if mdro_key in mdro_vector:
                    mdro_vector[mdro_key] = 1
    
    return mdro_vector

def get_shift(timestamp):
    """determine shift (morning: 7am-7pm, night: 7pm-7am)"""
    hour = timestamp.hour
    if 7 <= hour < 19:
        return 'morning'
    else:
        return 'night' 

def stage1_preprocess_lab_data(labs_file, output_dir, year):
    """
    stage 1: preprocess lab data and save to CSV
    
    @param labs_file : Path to the labs CSV file
    @param output_dir : Directory to save output files
    @param year : Year to process
    """
    print(f"Stage 1: Preprocessing lab data for year {year}...")
    start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = f"{output_dir}/processed_lab_data_{year}.csv"
    
    # Check if already processed
    if os.path.exists(output_file):
        print(f"Found existing processed lab data at {output_file}")
        return output_file
    
    masked_mrn_data = {}
    
    with open(labs_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            masked_mrn = row.get('MaskedMRN')
            specimen_time = row.get('specimen_taken_time')
            comment = row.get('OrganismFinal')
            mdro_category = row.get('MDRO_Category', '')

            # parsing specimen taken time
            if masked_mrn and specimen_time:
                try:
                    # milliseconds
                    specimen_time_dt = datetime.strptime(specimen_time, '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    try:
                        specimen_time_dt = datetime.strptime(specimen_time, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        continue
                
                # only process data for the specified year
                if specimen_time_dt.year != year:
                    continue
                
                if masked_mrn not in masked_mrn_data or specimen_time_dt < masked_mrn_data[masked_mrn]['specimen_taken_time']:
                    masked_mrn_data[masked_mrn] = {
                        'specimen_taken_time': specimen_time_dt,
                        'OrganismFinal': comment,
                        'MDRO_Category': mdro_category
                    }
    
    masked_mrn_df = pd.DataFrame([
        {'MaskedMRN': mrn, 
         'specimen_taken_time': data['specimen_taken_time'], 
         'OrganismFinal': data['OrganismFinal'],
         'MDRO_Category': data['MDRO_Category']}
        for mrn, data in masked_mrn_data.items()
    ])
    
    # Save to CSV
    masked_mrn_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    print(f"Stage 1 complete. Found {len(masked_mrn_df)} unique patients. Processing time: {end_time - start_time:.2f} seconds")
    print(f"Saved to {output_file}")
    
    return output_file

def process_chunk(chunk_data):
    """
    Process a chunk of data for Stage 2
    
    Parameters:
    -----------
    chunk_data : tuple
        (chunk, chunk_type, chunk_idx, masked_mrn_df)
        
    Returns:
    --------
    DataFrame
        Processed contacts
    """
    chunk, chunk_type, chunk_idx, masked_mrn_df = chunk_data
    
    try:
        # Log memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logging.info(f"Processing {chunk_type} chunk {chunk_idx}, Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        
        if chunk_type == 'flowsheet':
            # process flowsheet chunk
            chunk['ContactTime'] = pd.to_datetime(chunk['RecordedTime'])
            chunk['prov_id'] = chunk['prov_id']
            chunk = chunk.dropna(subset=['prov_id'])
            chunk = chunk.drop_duplicates(subset=['ContactTime', 'prov_id'])
            
        elif chunk_type == 'medication':
            # process medication chunk
            chunk['ContactTime'] = pd.to_datetime(chunk['AdministrationTime'])
            chunk['prov_id'] = chunk['admin_prov_id']
            chunk = chunk.dropna(subset=['prov_id'])
            chunk = chunk.drop_duplicates(subset=['ContactTime', 'prov_id'])
        
        # filter only to patients in our lab data
        chunk = chunk.merge(masked_mrn_df[['MaskedMRN']], on='MaskedMRN', how='inner')
        
        if len(chunk) == 0:
            return pd.DataFrame()
        
        contacts = chunk[['MaskedMRN', 'ContactTime', 'prov_id']].copy()
        
        contacts['shift'] = contacts['ContactTime'].apply(get_shift)
        contacts['date'] = contacts['ContactTime'].dt.date
        contacts['date_str'] = contacts['date'].astype(str)
        
        logging.info(f"Successfully processed {chunk_type} chunk {chunk_idx}, found {len(contacts)} contacts")
        return contacts
        
    except Exception as e:
        logging.error(f"Error processing {chunk_type} chunk {chunk_idx}: {str(e)}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()  # Return empty dataframe on error
    
def stage2_process_contacts(year, flowsheet_file, medication_file, labs_csv, output_dir, num_processes=None):
    """
    stage 2: process contacts from flowsheet and medication data
    
    @param year : Year to process
    @param flowsheet_file : Path to flowsheet file
    @param medication_file : Path to medication file
    @param labs_csv : Path to processed lab data CSV
    @param output_dir :  Directory to save output files
    @param num_processes : Number of processes to use
    """
    logger = logging.getLogger()
    logger.info(f"Stage 2: Processing contacts for year {year}...")
    start_time = time.time()
    
    # Create checkpoint directory
    checkpoint_dir = f"{output_dir}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = f"{output_dir}/processed_contacts_{year}.csv"
    checkpoint_file = f"{checkpoint_dir}/last_processed_chunk.txt"
    
    # Check if there's a checkpoint to resume from
    last_processed_flowsheet = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = f.read().strip().split(',')
            if len(checkpoint_data) >= 2 and checkpoint_data[0] == 'flowsheet':
                last_processed_flowsheet = int(checkpoint_data[1])
                logger.info(f"Resuming from flowsheet chunk {last_processed_flowsheet}")
    
    if os.path.exists(output_file):
        logger.info(f"Found existing processed contacts at {output_file}")
        return output_file
    
    # lab data
    logger.info("Loading lab data...")
    masked_mrn_df = pd.read_csv(labs_csv)
    masked_mrn_df['specimen_taken_time'] = pd.to_datetime(masked_mrn_df['specimen_taken_time'])
    
    # parallel processing (optional)
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
        
    logger.info(f"Using {num_processes} processes for contact processing")
    
    # Function to process and save intermediate chunks
    def process_and_save_intermediate_chunks(chunks, all_contacts, num_processes, checkpoint_dir):
        """Process a batch of chunks and save intermediate results."""
        logger.info(f"Processing batch of {len(chunks)} chunks...")
        with mp.Pool(processes=num_processes) as pool:
            try:
                results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))
                
                valid_results = [df for df in results if not df.empty]
                if valid_results:
                    # Save intermediate results
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    intermediate_file = f"{checkpoint_dir}/intermediate_{timestamp}.csv"
                    pd.concat(valid_results, ignore_index=True).to_csv(intermediate_file, index=False)
                    
                    all_contacts.extend(valid_results)
                    logger.info(f"Processed {len(valid_results)} valid chunks, saved to {intermediate_file}")
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")
                logger.error(traceback.format_exc())
    
    all_contacts = []
    
    # Process flowsheet data
    logger.info("Processing flowsheet data...")
    flowsheet_chunks = []
    chunk_size = 250000  # Reduced chunk size to avoid memory issues
    
    try:
        # prepare chunks for parallel processing
        for chunk_idx, chunk in enumerate(pd.read_csv(flowsheet_file, encoding='latin-1', chunksize=chunk_size)):
            # Save checkpoint after each chunk is prepared
            with open(checkpoint_file, 'w') as f:
                f.write(f"flowsheet,{chunk_idx}")
            
            if chunk_idx < last_processed_flowsheet:
                logger.info(f"  Skipping already processed flowsheet chunk {chunk_idx}...")
                continue
                
            if chunk_idx % 5 == 0:
                logger.info(f"  Preparing flowsheet chunk {chunk_idx}...")
                # Log memory usage
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"  Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
            
            try:
                # filter by year
                chunk['RecordedTime'] = pd.to_datetime(chunk['RecordedTime'], errors='coerce')
                year_mask = chunk['RecordedTime'].dt.year == year
                
                if year_mask.any():
                    flowsheet_chunks.append((chunk[year_mask].copy(), 'flowsheet', chunk_idx, masked_mrn_df))
                
                # Save intermediate results periodically
                if len(flowsheet_chunks) >= 20:  # Process in smaller batches
                    logger.info(f"Processing batch of {len(flowsheet_chunks)} flowsheet chunks...")
                    process_and_save_intermediate_chunks(flowsheet_chunks, all_contacts, num_processes, checkpoint_dir)
                    flowsheet_chunks = []  # Clear processed chunks
                    
            except Exception as e:
                logger.error(f"Error preparing flowsheet chunk {chunk_idx}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Process any remaining flowsheet chunks
        if flowsheet_chunks:
            logger.info(f"Processing remaining {len(flowsheet_chunks)} flowsheet chunks...")
            process_and_save_intermediate_chunks(flowsheet_chunks, all_contacts, num_processes, checkpoint_dir)
            flowsheet_chunks = []
    
    except Exception as e:
        logger.error(f"Fatal error in flowsheet processing: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Update checkpoint for medication processing
    with open(checkpoint_file, 'w') as f:
        f.write("medication,0")
    
    # Process medication data with similar pattern
    logger.info("Processing medication data...")
    medication_chunks = []
    last_processed_medication = 0
    
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(medication_file, encoding='latin-1', chunksize=chunk_size)):
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                f.write(f"medication,{chunk_idx}")
            
            if chunk_idx % 5 == 0:
                logger.info(f"  Preparing medication chunk {chunk_idx}...")
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"  Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
            
            try:
                # filter by year
                chunk['AdministrationTime'] = pd.to_datetime(chunk['AdministrationTime'], errors='coerce')
                year_mask = chunk['AdministrationTime'].dt.year == year
                
                if year_mask.any():
                    medication_chunks.append((chunk[year_mask].copy(), 'medication', chunk_idx, masked_mrn_df))
                
                # Process in batches
                if len(medication_chunks) >= 20:
                    logger.info(f"Processing batch of {len(medication_chunks)} medication chunks...")
                    process_and_save_intermediate_chunks(medication_chunks, all_contacts, num_processes, checkpoint_dir)
                    medication_chunks = []
                    
            except Exception as e:
                logger.error(f"Error preparing medication chunk {chunk_idx}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Process remaining medication chunks
        if medication_chunks:
            logger.info(f"Processing remaining {len(medication_chunks)} medication chunks...")
            process_and_save_intermediate_chunks(medication_chunks, all_contacts, num_processes, checkpoint_dir)
    
    except Exception as e:
        logger.error(f"Fatal error in medication processing: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Combine all contacts
    if not all_contacts:
        logger.error(f"No valid contacts found for year {year}")
        return None
    
    # Check if we have intermediate files to load
    logger.info("Checking for intermediate files...")
    intermediate_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('intermediate_') and f.endswith('.csv')]
    
    if intermediate_files and not all_contacts:
        logger.info(f"Loading {len(intermediate_files)} intermediate files...")
        for file in intermediate_files:
            try:
                file_path = os.path.join(checkpoint_dir, file)
                df = pd.read_csv(file_path)
                if not df.empty:
                    df['ContactTime'] = pd.to_datetime(df['ContactTime'])
                    all_contacts.append(df)
                    logger.info(f"Loaded {len(df)} contacts from {file}")
            except Exception as e:
                logger.error(f"Error loading intermediate file {file}: {str(e)}")
    
    logger.info(f"Combining {len(all_contacts)} contact dataframes...")
    
    try:
        combined_contacts = pd.concat(all_contacts, ignore_index=True)
        logger.info(f"Combined contacts: {len(combined_contacts)}")
        
        # sort by contact time
        logger.info("Sorting contacts...")
        combined_contacts.sort_values(['MaskedMRN', 'prov_id', 'ContactTime'], inplace=True)
        
        # deduplicate contacts within 15 minute windows
        logger.info("Deduplicating contacts within 15 minute windows...")
        
        # create time difference column (shifted by one row within each group)
        combined_contacts['prev_contact'] = combined_contacts.groupby(['MaskedMRN', 'prov_id'])['ContactTime'].shift(1)
        combined_contacts['time_diff'] = (combined_contacts['ContactTime'] - combined_contacts['prev_contact']).dt.total_seconds() / 60
        
        # keep only rows with time_diff > 15 or NaN (first contact)
        combined_contacts = combined_contacts[(combined_contacts['time_diff'].isna()) | (combined_contacts['time_diff'] > 15)]
        combined_contacts = combined_contacts.drop(['prev_contact', 'time_diff'], axis=1)
        
        logger.info(f"After deduplication: {len(combined_contacts)} contacts remaining")
        
        # Save final result
        logger.info(f"Saving processed contacts to {output_file}...")
        combined_contacts.to_csv(output_file, index=False)
        
    except Exception as e:
        logger.error(f"Error in final processing: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
    end_time = time.time()
    logger.info(f"Stage 2 complete. Processing time: {(end_time - start_time) / 60:.2f} minutes")
    logger.info(f"Saved to {output_file}")
    
    return output_file


# def stage2_process_contacts(year, flowsheet_file, medication_file, labs_csv, output_dir, num_processes=None):
#     """
#     stage 2: process contacts from flowsheet and medication data
    
#     @param year : Year to process
#     @param flowsheet_file : Path to flowsheet file
#     @param medication_file : Path to medication file
#     @param labs_csv : Path to processed lab data CSV
#     @param output_dir :  Directory to save output files
#     @param num_processes : Number of processes to use
#     """
#     print(f"Stage 2: Processing contacts for year {year}...")
#     start_time = time.time()
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     output_file = f"{output_dir}/processed_contacts_{year}.csv"
    
#     if os.path.exists(output_file):
#         print(f"Found existing processed contacts at {output_file}")
#         return output_file
    
#     # lab data
#     masked_mrn_df = pd.read_csv(labs_csv)
#     masked_mrn_df['specimen_taken_time'] = pd.to_datetime(masked_mrn_df['specimen_taken_time'])
    
#     # parallel processing (optional)
#     if num_processes is None:
#         num_processes = max(1, mp.cpu_count() - 1)
        
#     print(f"Using {num_processes} processes for contact processing")
    
#     all_contacts = []
    
#     # Process flowsheet data
#     print("Processing flowsheet data...")
#     flowsheet_chunks = []
#     chunk_size = 500000
    
#     # prepare chunks for parallel processing
#     for chunk_idx, chunk in enumerate(pd.read_csv(flowsheet_file, encoding='latin-1', chunksize=chunk_size)):
#         if chunk_idx % 5 == 0:
#             print(f"  Preparing flowsheet chunk {chunk_idx}...")
        
#         # filter by year
#         chunk['RecordedTime'] = pd.to_datetime(chunk['RecordedTime'], errors='coerce')
#         year_mask = chunk['RecordedTime'].dt.year == year
        
#         if year_mask.any():
#             flowsheet_chunks.append((chunk[year_mask].copy(), 'flowsheet', masked_mrn_df))
    
#     print(f"Processing {len(flowsheet_chunks)} flowsheet chunks in parallel...")
#     with mp.Pool(processes=num_processes) as pool:
#         results = list(tqdm(pool.imap(process_chunk, flowsheet_chunks), total=len(flowsheet_chunks)))
    
#     valid_results = [df for df in results if not df.empty]
#     if valid_results:
#         all_contacts.extend(valid_results)
#         print(f"Processed {len(valid_results)} valid flowsheet chunks")
    
#     # process medication data
#     print("Processing medication data...")
#     medication_chunks = []
    
#     for chunk_idx, chunk in enumerate(pd.read_csv(medication_file, encoding='latin-1', chunksize=chunk_size)):
#         if chunk_idx % 5 == 0:
#             print(f"  Preparing medication chunk {chunk_idx}...")
        
#         # filter by year
#         chunk['AdministrationTime'] = pd.to_datetime(chunk['AdministrationTime'], errors='coerce')
#         year_mask = chunk['AdministrationTime'].dt.year == year
        
#         if year_mask.any():
#             medication_chunks.append((chunk[year_mask].copy(), 'medication', masked_mrn_df))
    
#     print(f"Processing {len(medication_chunks)} medication chunks in parallel...")
#     with mp.Pool(processes=num_processes) as pool:
#         results = list(tqdm(pool.imap(process_chunk, medication_chunks), total=len(medication_chunks)))
    
#     valid_results = [df for df in results if not df.empty]
#     if valid_results:
#         all_contacts.extend(valid_results)
#         print(f"Processed {len(valid_results)} valid medication chunks")
    
#     # combine all contacts
#     if not all_contacts:
#         print(f"No valid contacts found for year {year}")
#         return None
    
#     combined_contacts = pd.concat(all_contacts, ignore_index=True)
#     print(f"Combined contacts: {len(combined_contacts)}")
    
#     # sort by contact time
#     combined_contacts.sort_values(['MaskedMRN', 'prov_id', 'ContactTime'], inplace=True)
    
#     # deduplicate contacts within 15 minute windows
#     print("Deduplicating contacts within 15 minute windows...")
    
#     # create time difference column (shifted by one row within each group)
#     combined_contacts['prev_contact'] = combined_contacts.groupby(['MaskedMRN', 'prov_id'])['ContactTime'].shift(1)
#     combined_contacts['time_diff'] = (combined_contacts['ContactTime'] - combined_contacts['prev_contact']).dt.total_seconds() / 60
    
#     # keep only rows with time_diff > 15 or NaN (first contact)
#     combined_contacts = combined_contacts[(combined_contacts['time_diff'].isna()) | (combined_contacts['time_diff'] > 15)]
#     combined_contacts = combined_contacts.drop(['prev_contact', 'time_diff'], axis=1)
    
#     print(f"After deduplication: {len(combined_contacts)} contacts remaining")
    
#     combined_contacts.to_csv(output_file, index=False)
    
#     end_time = time.time()
#     print(f"Stage 2 complete. Processing time: {(end_time - start_time) / 60:.2f} minutes")
#     print(f"Saved to {output_file}")
    
#     return output_file

def stage3_create_graphs(year, contacts_csv, labs_csv, output_dir, batch_size=10):
    """
    stage 3: create contact network graphs from processed data
    
    @param batch_size : Number of periods to process in a batch
    """
    print(f"Stage 3: Creating contact network graphs for year {year}...")
    start_time = time.time()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # load processed data
    contacts_df = pd.read_csv(contacts_csv)
    contacts_df['ContactTime'] = pd.to_datetime(contacts_df['ContactTime'])
    contacts_df['date'] = contacts_df['ContactTime'].dt.date
    
    # load lab data with MDRO information
    labs_df = pd.read_csv(labs_csv)
    labs_df['specimen_taken_time'] = pd.to_datetime(labs_df['specimen_taken_time'])
    
    # lookup dictionary for MDRO data
    mdro_data = {}
    for _, row in labs_df.iterrows():
        mrn = row['MaskedMRN']
        if mrn not in mdro_data:
            mdro_data[mrn] = []
        
        mdro_data[mrn].append({
            'specimen_taken_time': row['specimen_taken_time'],
            'MDRO_Category': row['MDRO_Category'] if not pd.isna(row['MDRO_Category']) else '',
            'OrganismFinal': row['OrganismFinal'] if not pd.isna(row['OrganismFinal']) else ''
        })
    
    # lookup dictionary for specimen times
    specimen_times = dict(zip(labs_df['MaskedMRN'], labs_df['specimen_taken_time']))
    
    # generate date pairs for two-day windows
    start_date = pd.Timestamp(f'{year}-01-01').date()
    end_date = pd.Timestamp(f'{year}-12-31').date()
    all_dates = pd.date_range(start=start_date, end=end_date)
    
    # group dates into pairs for two-day graphs
    date_pairs = []
    for i in range(0, len(all_dates), 2):
        if i + 1 < len(all_dates):
            date_pairs.append((all_dates[i].date(), all_dates[i+1].date()))
        else:
            # Handle the case where there's an odd number of days
            date_pairs.append((all_dates[i].date(), all_dates[i].date()))
    
    total_periods = len(date_pairs)
    print(f"Processing {total_periods} two-day periods")
    
    # process date pairs in batches to manage memory
    graphs_created = 0
    
    for batch_start in range(0, total_periods, batch_size):
        batch_end = min(batch_start + batch_size, total_periods)
        current_batch = date_pairs[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_periods + batch_size - 1)//batch_size}: "
              f"periods {batch_start+1} to {batch_end}")
        
        for period_idx, (start_date, end_date) in enumerate(current_batch):
            graph_file = f"{output_dir}/network_{start_date}_to_{end_date}.graphml"
            
            # case where graph already exists
            if os.path.exists(graph_file):
                print(f"  Graph for {start_date} to {end_date} already exists, skipping")
                graphs_created += 1
                continue
                
            print(f"  Creating graph for period {batch_start + period_idx + 1}/{total_periods}: {start_date} to {end_date}")
            
            # filter contacts
            period_mask = (contacts_df['date'] >= start_date) & (contacts_df['date'] <= end_date)
            period_contacts = contacts_df[period_mask].copy()
            
            if len(period_contacts) == 0:
                print(f"    No contacts for this period, skipping")
                continue
            
            # create graph
            G = nx.MultiGraph()
            
            # process each shift within window
            shifts = []
            for date in pd.date_range(start=start_date, end=end_date):
                shifts.append((date.date(), 'morning'))
                shifts.append((date.date(), 'night'))
            
            # track all unique patients for this period
            all_patients_in_period = set()
            
            edges_added = 0
            
            for date, shift_type in shifts:
                # filter contacts for this shift
                date_str = str(date)
                shift_mask = (period_contacts['date_str'] == date_str) & (period_contacts['shift'] == shift_type)
                shift_contacts = period_contacts[shift_mask]
                
                # skip if no contacts in this shift
                if len(shift_contacts) == 0:
                    continue
                
                # update the set of all unique patients
                all_patients_in_period.update(shift_contacts['MaskedMRN'].unique())
                
                # create a provider-based lookup
                provider_patients = {}
                for _, row in shift_contacts.iterrows():
                    prov_id = row['prov_id']
                    if prov_id not in provider_patients:
                        provider_patients[prov_id] = {}
                    
                    mrn = row['MaskedMRN']
                    if mrn not in provider_patients[prov_id]:
                        provider_patients[prov_id][mrn] = []
                    
                    provider_patients[prov_id][mrn].append(row['ContactTime'])
                
                # process each provider
                for prov_id, patients in provider_patients.items():
                    # skip if provider saw only one patient
                    if len(patients) < 2:
                        continue
                    
                    # get list of patients seen by this provider
                    patient_list = list(patients.keys())
                    
                    # create edges between all pairs of patients seen by this provider
                    for i in range(len(patient_list)):
                        patient_i = patient_list[i]
                        # get specimen time for patient i
                        if patient_i not in specimen_times:
                            continue
                        patient_i_specimen_time = specimen_times[patient_i]
                        
                        for j in range(i+1, len(patient_list)):
                            patient_j = patient_list[j]
                            # get specimen time for patient j
                            if patient_j not in specimen_times:
                                continue
                            patient_j_specimen_time = specimen_times[patient_j]
                            
                            # get contact times for this provider with these patients
                            patient_i_contacts = patients[patient_i]
                            patient_j_contacts = patients[patient_j]
                            
                            # check for valid connection paths
                            valid_connection = False
                            edge_attributes = {}
                            
                            # case 1: Provider saw patient i, then patient j before patient j's specimen collection
                            for time_i in patient_i_contacts:
                                for time_j in patient_j_contacts:
                                    if time_i < time_j < patient_j_specimen_time:
                                        valid_connection = True
                                        edge_attributes = {
                                            'contact_time_i': str(time_i),
                                            'contact_time_j': str(time_j),
                                            'prov_id': str(prov_id),
                                            'direction': f"{patient_i}->{patient_j}"
                                        }
                                        break
                                if valid_connection:
                                    break
                            
                            # case 2: Provider saw patient j, then patient i before patient i's specimen collection
                            if not valid_connection:
                                for time_j in patient_j_contacts:
                                    for time_i in patient_i_contacts:
                                        if time_j < time_i < patient_i_specimen_time:
                                            valid_connection = True
                                            edge_attributes = {
                                                'contact_time_j': str(time_j),
                                                'contact_time_i': str(time_i),
                                                'prov_id': str(prov_id),
                                                'direction': f"{patient_j}->{patient_i}"
                                            }
                                            break
                                    if valid_connection:
                                        break
                            
                            # if there's a valid connection, add the edge
                            if valid_connection:
                                G.add_edge(
                                    str(patient_i),
                                    str(patient_j),
                                    **edge_attributes
                                )
                                edges_added += 1
            
            # add all patients as nodes with MDRO status attributes
            for patient in all_patients_in_period:
                # get MDRO status for this patient
                mdro_vector = get_mdro_status(patient, start_date, end_date, mdro_data)
                
                # add the patient as a node with MDRO attributes
                G.add_node(
                    str(patient),
                    **mdro_vector  # add all MDRO attributes
                )
            
            # save the graph if it has any edges
            if G.number_of_edges() > 0:
                try:
                    # try to use the simpler GraphML writer if lxml is not available
                    try:
                        nx.write_graphml(G, graph_file, prettyprint=False)
                    except ImportError:
                        # If lxml is not available, use the basic GraphML writer
                        print("    Warning: lxml not available, using basic GraphML writer")
                        nx.write_graphml(G, graph_file, prettyprint=False)
                        
                    print(f"    Saved graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
                    graphs_created += 1
                        
                except Exception as e:
                    print(f"    Error saving graph: {e}")
            else:
                print(f"    No edges in graph for period {start_date} to {end_date}. Skipping save.")
    
    return graphs_created

if __name__ == "__main__":
    
    year = 2018 # edit year param
    microbio_file = 'categorized_microbio_250326.csv'
    flowsheet_file = 'Flowsheet_20241210.csv'
    medication_file = 'MedicationAdmin_20241210.csv'
    base_output_dir = f'hcw_output_{year}'
    
    # base output directory
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # processing directories
    stage_dir = f"{base_output_dir}/staged_processing_{year}"
    graph_dir = f"{base_output_dir}/contact_networks_{year}"
    
    if not os.path.exists(stage_dir):
        os.makedirs(stage_dir)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    # find number of cpu cores
    num_processes = max(1, mp.cpu_count() - 1)  # leave one cpu for OS
    print(f"System has {mp.cpu_count()} CPU cores, will use {num_processes} for processing")
    
    start_time = time.time()
    
    # stage 1: preprocess lab data
    labs_csv = stage1_preprocess_lab_data(microbio_file, stage_dir, year)
    
    # stage 2: process contacts
    contacts_csv = stage2_process_contacts(
        year, 
        flowsheet_file, 
        medication_file, 
        labs_csv, 
        stage_dir, 
        num_processes=num_processes
    )
    
    if contacts_csv is None:
        print(f"No contacts found for year {year}")
        exit(1)
    
    # stage 3: create graphs
    graphs_created = stage3_create_graphs(year, contacts_csv, labs_csv, graph_dir)
    
    # calculate total processing time
    total_time = (time.time() - start_time) / 60
    
    print("\nProcessing complete!")
    print(f"Created {graphs_created} graphs for year {year}")
    print(f"Total processing time: {total_time:.2f} minutes")
    print(f"Output directory: {base_output_dir}")