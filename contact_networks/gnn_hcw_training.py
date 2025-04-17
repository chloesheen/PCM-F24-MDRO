import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
MODEL_TYPE = 'GCN'  # Options: 'GCN', 'GraphSAGE', 'GAT'
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 100
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPHS_DIR = 'training_room_graphs_20250514'
MDRO_DATA_PATH = 'categorized_microbio_250326.csv'  # Path to your MDRO results file
OUTPUT_DIR = 'model_outputs'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Model type: {MODEL_TYPE}")

# Load MDRO data
def load_mdro_data(file_path):
    """Load MDRO results from CSV file"""
    # Try different encodings until one works
    encodings = ['latin-1', 'utf-8', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"Trying to read file with {encoding} encoding...")
            mdro_df = pd.read_csv(file_path, encoding=encoding, low_memory=False, on_bad_lines='skip')
            print(f"Successfully read file with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding")
            if encoding == encodings[-1]:
                print("All encodings failed. Trying engine='python'...")
                try:
                    mdro_df = pd.read_csv(file_path, encoding='latin-1', engine='python', on_bad_lines='skip')
                    print("Successfully read file with python engine")
                except Exception as e:
                    raise Exception(f"Could not read file: {str(e)}")
    
    # Convert MDRO_Category to binary MDRO status
    # If 'MDRO_Category' is 'Non-MDRO', MDRO is negative (0)
    # Otherwise, MDRO is positive (1)
    mdro_df['mdro_positive'] = mdro_df['MDRO_Category'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip().lower() == 'non-mdro' else 1
    )
    
    # Create a dictionary for quick lookup: {patient_id: mdro_status}
    mdro_dict = dict(zip(mdro_df['MaskedMRN'].astype(str), mdro_df['mdro_positive']))
    
    print(f"Loaded MDRO data: {len(mdro_df)} patients")
    print(f"Positive MDRO cases: {mdro_df['mdro_positive'].sum()}")
    
    return mdro_dict

# Load all graphs and prepare PyG dataset
def process_graphs(graphs_dir, mdro_dict):
    """Load all GraphML files and convert to PyG Data objects"""
    graph_paths = glob.glob(os.path.join(graphs_dir, '*.graphml'))
    print(f"Found {len(graph_paths)} graph files")
    
    dataset = []
    patient_ids_with_mdro_data = set()
    patient_ids_total = set()
    skipped_graphs = 0
    
    for graph_path in tqdm(graph_paths, desc="Processing graphs"):
        try:
            # Load the NetworkX graph
            G = nx.read_graphml(graph_path)
            
            # Skip graphs without edges
            if G.number_of_edges() == 0:
                skipped_graphs += 1
                continue
            
            # First, identify which nodes have MDRO data
            nodes_with_mdro_data = {}
            for node in G.nodes():
                patient_ids_total.add(node)
                if node in mdro_dict:
                    nodes_with_mdro_data[node] = mdro_dict[node]
                    patient_ids_with_mdro_data.add(node)
            
            # Skip graphs with no nodes that have MDRO data
            if not nodes_with_mdro_data:
                print(f"Skipping graph {os.path.basename(graph_path)}: No nodes with MDRO data")
                skipped_graphs += 1
                continue
            
            # Create node features and labels
            node_features = []
            node_labels = []
            
            # Mapping of node IDs to consecutive indices for PyG
            node_mapping = {}
            filtered_nodes = []
            
            # Only process nodes that have MDRO data
            for i, node in enumerate([n for n in G.nodes() if n in nodes_with_mdro_data]):
                node_mapping[node] = i
                filtered_nodes.append(node)
                
                # Node degree as a simple feature (using the full graph for degree calculation)
                degree = G.degree(node)
                
                # More features can be added here:
                # - Clustering coefficient
                # - Page rank
                # - Provider diversity (number of unique providers)
                
                # Create a simple feature vector for each node
                features = [degree]
                
                # Get MDRO status (guaranteed to exist)
                mdro_status = nodes_with_mdro_data[node]
                
                node_features.append(features)
                node_labels.append(mdro_status)
            
            # Create a subgraph of only the nodes with MDRO data
            sub_G = G.subgraph(filtered_nodes)
            
            # Create edge index
            edge_index = []
            edge_attr = []
            
            for u, v, data in sub_G.edges(data=True):
                # Convert node IDs to indices
                src_idx = node_mapping[u]
                dst_idx = node_mapping[v]
                
                # Add edges in both directions (undirected graph)
                edge_index.append([src_idx, dst_idx])
                edge_index.append([dst_idx, src_idx])
                
                # Edge features (optional)
                if 'prov_id' in data:
                    # Simple hash of provider ID as a numeric feature
                    prov_feature = hash(data['prov_id']) % 1000 / 1000
                    edge_attr.append([prov_feature])
                    edge_attr.append([prov_feature])  # Same feature for reverse edge
                else:
                    edge_attr.append([0.0])
                    edge_attr.append([0.0])
            
            if not edge_index:  # Skip if no edges after processing
                print(f"Skipping graph {os.path.basename(graph_path)}: No edges between nodes with MDRO data")
                skipped_graphs += 1
                continue
                
            # Convert to PyG Data object
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_labels, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            dataset.append(data)
            
        except Exception as e:
            print(f"Error processing {graph_path}: {e}")
            skipped_graphs += 1
    
    print(f"Successfully processed {len(dataset)} graphs")
    print(f"Skipped {skipped_graphs} graphs due to errors or no edges")
    print(f"Total unique patients: {len(patient_ids_total)}")
    print(f"Patients with MDRO data: {len(patient_ids_with_mdro_data)}")
    print(f"Patients without MDRO data (excluded): {len(patient_ids_total) - len(patient_ids_with_mdro_data)}")
    
    return dataset

# Define GNN model
class GNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, model_type='GCN'):
        super(GNN, self).__init__()
        
        self.model_type = model_type
        
        # Choose the appropriate layer type
        if model_type == 'GCN':
            self.conv1 = GCNConv(input_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'GraphSAGE':
            self.conv1 = SAGEConv(input_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(input_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Output layer
        self.lin = nn.Linear(hidden_channels, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Final prediction layer
        x = self.lin(x)
        
        return x.squeeze(1)

# Training function
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Handle empty input
        if data.x.shape[0] == 0 or data.edge_index.shape[1] == 0:
            continue
            
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Calculate loss (with class weighting for imbalanced data)
        pos_weight = torch.tensor([5.0]).to(device)  # Assuming positive class is less frequent
        loss = F.binary_cross_entropy_with_logits(
            out, data.y, 
            pos_weight=pos_weight if pos_weight is not None else None
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Handle empty input
            if data.x.shape[0] == 0 or data.edge_index.shape[1] == 0:
                continue
                
            # Forward pass
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            # Store true and predicted values
            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    
    if not y_true or not y_pred:
        return {
            'loss': float('inf'),
            'auc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': None
        }
    
    # Concatenate results
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.0
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Calculate loss
    loss = F.binary_cross_entropy(
        torch.tensor(y_pred, dtype=torch.float), 
        torch.tensor(y_true, dtype=torch.float)
    ).item()
    
    return {
        'loss': loss,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Main execution
def main():
    # Load MDRO data
    mdro_dict = load_mdro_data(MDRO_DATA_PATH)
    
    # Process graphs and create dataset
    dataset = process_graphs(GRAPHS_DIR, mdro_dict)
    
    if not dataset:
        print("No valid graphs to process. Exiting.")
        return
    
    # Split dataset into train, validation, and test sets
    train_val_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)
    
    print(f"Train set: {len(train_data)} graphs")
    print(f"Validation set: {len(val_data)} graphs")
    print(f"Test set: {len(test_data)} graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    
    # Determine input channel size from data
    input_channels = dataset[0].x.shape[1]
    
    # Initialize model
    model = GNN(input_channels, HIDDEN_CHANNELS, MODEL_TYPE).to(DEVICE)
    print(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    val_metrics = []
    
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train(model, train_loader, optimizer, DEVICE)
        train_losses.append(train_loss)
        
        # Validate
        val_results = evaluate(model, val_loader, DEVICE)
        val_loss = val_results['loss']
        val_losses.append(val_loss)
        val_metrics.append(val_results)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
            # Save best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_results,
            }, os.path.join(OUTPUT_DIR, f'{MODEL_TYPE}_best_model.pt'))
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val AUC: {val_results['auc']:.4f}, Val F1: {val_results['f1']:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_results = evaluate(model, test_loader, DEVICE)
    
    print("\nTest Results:")
    print(f"  Loss: {test_results['loss']:.4f}")
    print(f"  AUC: {test_results['auc']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    print(f"  F1: {test_results['f1']:.4f}")
    print(f"  Confusion Matrix:\n{test_results['confusion_matrix']}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([m['auc'] for m in val_metrics], label='AUC')
    plt.plot([m['f1'] for m in val_metrics], label='F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_TYPE}_training_curves.png'))
    plt.close()
    
    # Plot confusion matrix
    cm = test_results['confusion_matrix']
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_TYPE}_confusion_matrix.png'))
    plt.close()

    # Plot ROC curve
    y_true_all = []
    y_pred_all = []
    
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            if data.x.shape[0] == 0 or data.edge_index.shape[1] == 0:
                continue
            
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            y_true_all.append(data.y.cpu().numpy())
            y_pred_all.append(out.cpu().numpy())
    
    if y_true_all and y_pred_all:
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        
        fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{MODEL_TYPE}_roc_curve.png'))
        plt.close()

    # Save test results
    with open(os.path.join(OUTPUT_DIR, f'{MODEL_TYPE}_test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_results['loss']:.4f}\n")
        f.write(f"Test AUC: {test_results['auc']:.4f}\n")
        f.write(f"Test Precision: {test_results['precision']:.4f}\n")
        f.write(f"Test Recall: {test_results['recall']:.4f}\n")
        f.write(f"Test F1: {test_results['f1']:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_results['confusion_matrix']}\n")
    
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()