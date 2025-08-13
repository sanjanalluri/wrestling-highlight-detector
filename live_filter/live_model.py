#!/usr/bin/env python3
"""
Live vs Non-Live Detection Model
Custom 3D CNN with audio-visual fusion for wrestling match analysis
Uses k-fold cross-validation for robust evaluation with limited data
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database_manager import DatabaseManager

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50
K_FOLDS = 5
RANDOM_SEED = 42
FEATURES_DIR = "live_filter/features"
MODEL_SAVE_DIR = "live_filter/models"

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class LiveDataset(Dataset):
    """Dataset for loading chunk features"""
    
    def __init__(self, chunk_ids, labels):
        self.chunk_ids = chunk_ids
        self.labels = labels
        self.db_manager = DatabaseManager()
        
    def __len__(self):
        return len(self.chunk_ids)
    
    def __getitem__(self, idx):
        chunk_id = self.chunk_ids[idx]
        label = self.labels[idx]
        
        # Load features from .npz file
        features_path = os.path.join(FEATURES_DIR, f"chunk_{chunk_id}.npz")
        
        try:
            data = np.load(features_path)
            visual_features = data['visual_features'].astype(np.float32)
            audio_features = data['audio_features'].astype(np.float32)
            
            # Validate feature dimensions
            if visual_features.shape != (8, 54):
                print(f"Warning: Visual features for chunk {chunk_id} have unexpected shape: {visual_features.shape}")
                visual_features = np.zeros((8, 54), dtype=np.float32)
            
            if audio_features.shape != (29,):
                print(f"Warning: Audio features for chunk {chunk_id} have unexpected shape: {audio_features.shape}")
                audio_features = np.zeros(29, dtype=np.float32)
            
            # Convert label to tensor
            label_tensor = torch.tensor(1 if label == 'LIVE' else 0, dtype=torch.long)
            
            return {
                'visual': torch.from_numpy(visual_features),
                'audio': torch.from_numpy(audio_features),
                'label': label_tensor,
                'chunk_id': chunk_id
            }
            
        except Exception as e:
            print(f"Error loading chunk {chunk_id}: {e}")
            # Return dummy data if loading fails
            return {
                'visual': torch.zeros(8, 54),  # 8 frames, 54 features per frame
                'audio': torch.zeros(29),      # 29 audio features
                'label': torch.tensor(0, dtype=torch.long),
                'chunk_id': chunk_id
            }

class Live3DCNN(nn.Module):
    """Custom 3D CNN for live vs non-live detection"""
    
    def __init__(self, num_classes=2):
        super(Live3DCNN, self).__init__()
        
        # Visual feature dimensions: 8 frames, 54 features per frame
        self.visual_input_size = 54
        self.num_frames = 8
        
        # Audio feature dimensions: 29 features
        self.audio_input_size = 29
        
        # Visual processing branch (simplified to reduce overfitting)
        self.visual_branch = nn.Sequential(
            # Reshape: (batch, 8, 54) -> (batch, 1, 8, 54) for 2D conv
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),  # Reduced from 32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),  # Added dropout
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),  # Reduced from 64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),  # Added dropout
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),  # Reduced from 128
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Dropout(0.5)
        )
        
        # Audio processing branch
        self.audio_branch = nn.Sequential(
            nn.Linear(self.audio_input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Fusion and classification (simplified)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128, 128),  # 64 from visual + 128 from audio (reduced)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),  # Reduced
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, visual, audio):
        # Process visual features
        # Reshape: (batch, 8, 54) -> (batch, 1, 8, 54)
        batch_size = visual.size(0)
        visual = visual.view(batch_size, 1, self.num_frames, self.visual_input_size)
        visual_features = self.visual_branch(visual)
        
        # Process audio features
        audio_features = self.audio_branch(audio)
        
        # Fusion
        combined = torch.cat([visual_features, audio_features], dim=1)
        output = self.fusion(combined)
        
        return output

class LiveModelTrainer:
    """Handles training and evaluation with k-fold cross-validation"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
    def load_data(self):
        """Load all chunk data from database"""
        print("📊 Loading chunk data from database...")
        
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, label, metadata_path
            FROM training_chunks
            ORDER BY chunk_id
        ''')
        
        chunks = []
        for row in cursor.fetchall():
            chunk_id, label, metadata_path = row
            chunks.append({
                'chunk_id': chunk_id,
                'label': label,
                'metadata_path': metadata_path
            })
        
        conn.close()
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Count by label
        live_count = sum(1 for c in chunks if c['label'] == 'LIVE')
        not_live_count = sum(1 for c in chunks if c['label'] == 'NOT LIVE')
        print(f"  LIVE: {live_count}")
        print(f"  NOT LIVE: {not_live_count}")
        
        return chunks
    
    def prepare_k_fold_data(self, chunks, k=K_FOLDS):
        """Prepare k-fold cross-validation splits"""
        print(f"🔄 Preparing {k}-fold cross-validation...")
        
        # Extract features for stratification
        chunk_ids = [c['chunk_id'] for c in chunks]
        labels = [c['label'] for c in chunks]
        
        # Convert labels to numeric for stratification
        label_map = {'NOT LIVE': 0, 'LIVE': 1}
        numeric_labels = [label_map[label] for label in labels]
        
        # Create stratified k-fold splits
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        
        fold_data = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(chunk_ids, numeric_labels)):
            train_chunks = [chunks[i] for i in train_idx]
            val_chunks = [chunks[i] for i in val_idx]
            
            fold_data.append({
                'fold': fold_idx + 1,
                'train': train_chunks,
                'val': val_chunks
            })
            
            # Print fold statistics
            train_live = sum(1 for c in train_chunks if c['label'] == 'LIVE')
            train_not_live = sum(1 for c in train_chunks if c['label'] == 'NOT LIVE')
            val_live = sum(1 for c in val_chunks if c['label'] == 'LIVE')
            val_not_live = sum(1 for c in val_chunks if c['label'] == 'NOT LIVE')
            
            print(f"  Fold {fold_idx + 1}: Train ({train_live} LIVE, {train_not_live} NOT LIVE), "
                  f"Val ({val_live} LIVE, {val_not_live} NOT LIVE)")
        
        return fold_data
    
    def split_train_test(self, chunks, test_size=0.2):
        """Split data into training and test sets with stratification"""
        
        # Extract features for stratification
        chunk_ids = [c['chunk_id'] for c in chunks]
        labels = [c['label'] for c in chunks]
        
        # Convert labels to numeric for stratification
        label_map = {'NOT LIVE': 0, 'LIVE': 1}
        numeric_labels = [label_map[label] for label in labels]
        
        # Split with stratification to maintain class balance
        train_ids, test_ids, train_labels, test_labels = train_test_split(
            chunk_ids, numeric_labels, 
            test_size=test_size, 
            random_state=RANDOM_SEED,
            stratify=numeric_labels
        )
        
        # Convert back to chunk dictionaries
        chunks_dict = {c['chunk_id']: c for c in chunks}
        
        train_chunks = [chunks_dict[chunk_id] for chunk_id in train_ids]
        test_chunks = [chunks_dict[chunk_id] for chunk_id in test_ids]
        
        # Print split statistics
        train_live = sum(1 for c in train_chunks if c['label'] == 'LIVE')
        train_not_live = sum(1 for c in train_chunks if c['label'] == 'NOT LIVE')
        test_live = sum(1 for c in test_chunks if c['label'] == 'LIVE')
        test_not_live = sum(1 for c in test_chunks if c['label'] == 'NOT LIVE')
        
        print(f"  Training set: {train_live} LIVE, {train_not_live} NOT LIVE")
        print(f"  Test set: {test_live} LIVE, {test_not_live} NOT LIVE")
        
        return train_chunks, test_chunks
    
    def train_fold(self, fold_data, fold_idx):
        """Train model for one fold"""
        print(f"\n🎯 Training Fold {fold_idx + 1}/{len(fold_data)}")
        print("=" * 50)
        
        # Prepare data
        train_chunks = fold_data[fold_idx]['train']
        val_chunks = fold_data[fold_idx]['val']
        
        train_ids = [c['chunk_id'] for c in train_chunks]
        train_labels = [c['label'] for c in train_chunks]
        val_ids = [c['chunk_id'] for c in val_chunks]
        val_labels = [c['label'] for c in val_chunks]
        
        # Create datasets
        train_dataset = LiveDataset(train_ids, train_labels)
        val_dataset = LiveDataset(val_ids, val_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Validate data loaders
        if len(train_loader) == 0:
            raise ValueError(f"Training dataset is empty for fold {fold_idx + 1}")
        if len(val_loader) == 0:
            raise ValueError(f"Validation dataset is empty for fold {fold_idx + 1}")
        
        # Initialize model
        model = Live3DCNN().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)  # Increased regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10  # Stop if no improvement for 10 epochs
        
        print(f"Training for {EPOCHS} epochs...")
        
        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                visual = batch['visual'].to(self.device)
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(visual, audio)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    visual = batch['visual'].to(self.device)
                    audio = batch['audio'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(visual, audio)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            
            train_losses.append(train_loss / len(train_loader) if len(train_loader) > 0 else 0)
            val_losses.append(val_loss / len(val_loader) if len(val_loader) > 0 else 0)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}: "
                      f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%, "
                      f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model and check for early stopping
            current_val_loss = val_losses[-1]
            
            # Save if validation accuracy improves OR validation loss decreases
            if val_acc > best_val_acc or current_val_loss < best_val_loss:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                
                patience_counter = 0
                best_model_path = os.path.join(MODEL_SAVE_DIR, f"fold_{fold_idx + 1}_best.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': current_val_loss,
                    'fold': fold_idx + 1
                }, best_model_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break
        
        # Calculate final metrics
        if len(all_labels) > 0 and len(set(all_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        fold_results = {
            'fold': fold_idx + 1,
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        print(f"Fold {fold_idx + 1} Results:")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return fold_results
    
    def evaluate_on_test_set(self, test_chunks, all_results):
        """Evaluate the best model from each fold on the holdout test set"""
        print("Evaluating all fold models on holdout test set...")
        
        # Prepare test data
        test_ids = [c['chunk_id'] for c in test_chunks]
        test_labels = [c['label'] for c in test_chunks]
        test_dataset = LiveDataset(test_ids, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        if len(test_loader) == 0:
            print("Warning: Test dataset is empty!")
            return None
        
        # Convert labels to numeric
        label_map = {'NOT LIVE': 0, 'LIVE': 1}
        numeric_test_labels = [label_map[label] for label in test_labels]
        
        # Evaluate each fold's best model
        fold_results = []
        for fold_idx, fold_result in enumerate(all_results):
            print(f"  Testing Fold {fold_idx + 1} model...")
            
            # Load the best model from this fold
            model_path = os.path.join(MODEL_SAVE_DIR, f"fold_{fold_idx + 1}_best.pth")
            if not os.path.exists(model_path):
                print(f"    Warning: Model file not found: {model_path}")
                continue
            
            # Load model
            try:
                model = Live3DCNN().to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                print(f"    Error loading model from {model_path}: {e}")
                continue
            model.eval()
            
            # Evaluate
            test_correct = 0
            test_total = 0
            all_predictions = []
            
            with torch.no_grad():
                for batch in test_loader:
                    visual = batch['visual'].to(self.device)
                    audio = batch['audio'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(visual, audio)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    all_predictions.extend(predicted.cpu().numpy())
            
            # Calculate metrics
            test_acc = 100 * test_correct / test_total
            if len(numeric_test_labels) > 0 and len(set(numeric_test_labels)) > 1:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    numeric_test_labels, all_predictions, average='weighted'
                )
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
            conf_matrix = confusion_matrix(numeric_test_labels, all_predictions)
            
            fold_test_result = {
                'fold': fold_idx + 1,
                'test_accuracy': test_acc,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_confusion_matrix': conf_matrix
            }
            
            fold_results.append(fold_test_result)
            print(f"    Test Accuracy: {test_acc:.2f}%, F1: {f1:.4f}")
        
        # Find best performing fold on test set
        best_fold = max(fold_results, key=lambda x: x['test_accuracy'])
        
        # Calculate ensemble prediction (majority vote)
        print("  Computing ensemble predictions...")
        ensemble_predictions = self.compute_ensemble_predictions(test_loader, all_results)
        
        # Calculate ensemble metrics
        if len(ensemble_predictions) > 0:
            ensemble_acc = 100 * accuracy_score(numeric_test_labels, ensemble_predictions)
            if len(numeric_test_labels) > 0 and len(set(numeric_test_labels)) > 1:
                ensemble_precision, ensemble_recall, ensemble_f1, _ = precision_recall_fscore_support(
                    numeric_test_labels, ensemble_predictions, average='weighted'
                )
            else:
                ensemble_precision, ensemble_recall, ensemble_f1 = 0.0, 0.0, 0.0
            ensemble_conf_matrix = confusion_matrix(numeric_test_labels, ensemble_predictions)
        else:
            print("    Warning: No ensemble predictions generated")
            ensemble_acc = 0.0
            ensemble_precision = 0.0
            ensemble_recall = 0.0
            ensemble_f1 = 0.0
            ensemble_conf_matrix = np.zeros((2, 2))
        
        final_results = {
            'individual_folds': fold_results,
            'best_fold': best_fold,
            'ensemble': {
                'test_accuracy': ensemble_acc,
                'test_precision': ensemble_precision,
                'test_recall': ensemble_recall,
                'test_f1': ensemble_f1,
                'test_confusion_matrix': ensemble_conf_matrix
            }
        }
        
        # Print final results
        print(f"\n🏆 FINAL TEST RESULTS:")
        print(f"  Best Individual Fold: {best_fold['fold']} (Acc: {best_fold['test_accuracy']:.2f}%)")
        print(f"  Ensemble Model: Acc: {ensemble_acc:.2f}%, F1: {ensemble_f1:.4f}")
        print(f"  Test Set Size: {len(test_chunks)} chunks")
        
        return final_results
    
    def compute_ensemble_predictions(self, test_loader, all_results):
        """Compute ensemble predictions using all fold models"""
        all_predictions = []
        
        # Collect predictions from all models
        for fold_idx in range(len(all_results)):
            model_path = os.path.join(MODEL_SAVE_DIR, f"fold_{fold_idx + 1}_best.pth")
            if not os.path.exists(model_path):
                continue
            
            try:
                model = Live3DCNN().to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            except Exception as e:
                print(f"    Error loading model from {model_path}: {e}")
                continue
            
            fold_predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    visual = batch['visual'].to(self.device)
                    audio = batch['audio'].to(self.device)
                    
                    outputs = model(visual, audio)
                    _, predicted = torch.max(outputs.data, 1)
                    fold_predictions.extend(predicted.cpu().numpy())
            
            all_predictions.append(fold_predictions)
        
        # Compute majority vote
        if all_predictions:
            all_predictions = np.array(all_predictions)
            # For binary classification, use mode (majority vote)
            from scipy.stats import mode
            ensemble_predictions, _ = mode(all_predictions, axis=0)
            return ensemble_predictions.flatten()
        else:
            return []
    

    
    def run_k_fold_training(self):
        """Run complete k-fold cross-validation training with holdout test set"""
        print("🚀 Starting K-Fold Cross-Validation Training with Holdout Test Set")
        print("=" * 70)
        
        # Load data
        chunks = self.load_data()
        
        if not chunks:
            print("❌ No chunk data found!")
            return None, None
        
        # Split data: 80% for k-fold training, 20% for final testing
        print("📊 Splitting data: 80% training/validation, 20% final testing")
        train_chunks, test_chunks = self.split_train_test(chunks)
        
        print(f"  Training/Validation set: {len(train_chunks)} chunks")
        print(f"  Final Test set: {len(test_chunks)} chunks")
        
        # Prepare k-fold splits (only on training data)
        fold_data = self.prepare_k_fold_data(train_chunks, K_FOLDS)
        
        # Train each fold
        all_results = []
        for fold_idx in range(len(fold_data)):
            try:
                fold_results = self.train_fold(fold_data, fold_idx)
                all_results.append(fold_results)
            except Exception as e:
                print(f"❌ Error training fold {fold_idx + 1}: {str(e)}")
                continue
        
        if not all_results:
            print("❌ No folds completed successfully!")
            return None, None
        
        # Aggregate k-fold results
        self.aggregate_results(all_results)
        
        # Final evaluation on holdout test set
        print("\n🧪 FINAL EVALUATION ON HOLDOUT TEST SET")
        print("=" * 50)
        try:
            final_results = self.evaluate_on_test_set(test_chunks, all_results)
        except Exception as e:
            print(f"❌ Error during final evaluation: {str(e)}")
            final_results = None
        
        # Save final ensemble model
        self.save_ensemble_model(all_results, final_results)
        
        return all_results, final_results
    
    def aggregate_results(self, all_results):
        """Aggregate and display k-fold results"""
        print("\n📊 K-FOLD CROSS-VALIDATION RESULTS")
        print("=" * 60)
        
        # Calculate averages
        avg_val_acc = np.mean([r['best_val_acc'] for r in all_results])
        avg_precision = np.mean([r['precision'] for r in all_results])
        avg_recall = np.mean([r['recall'] for r in all_results])
        avg_f1 = np.mean([r['f1'] for r in all_results])
        
        std_val_acc = np.std([r['best_val_acc'] for r in all_results])
        std_precision = np.std([r['precision'] for r in all_results])
        std_recall = np.std([r['recall'] for r in all_results])
        std_f1 = np.std([r['f1'] for r in all_results])
        
        print(f"Average Validation Accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
        print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f} ± {std_f1:.4f}")
        
        print(f"\nIndividual Fold Results:")
        for result in all_results:
            print(f"  Fold {result['fold']}: "
                  f"Val Acc: {result['best_val_acc']:.2f}%, "
                  f"F1: {result['f1']:.4f}")
        
        # Plot training curves
        self.plot_training_curves(all_results)
        
        # Plot confusion matrices
        self.plot_confusion_matrices(all_results)
    
    def plot_training_curves(self, all_results):
        """Plot training and validation curves for all folds"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves Across All Folds', fontsize=16)
        
        # Training loss
        ax1 = axes[0, 0]
        for result in all_results:
            ax1.plot(result['train_losses'], label=f"Fold {result['fold']}")
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Validation loss
        ax2 = axes[0, 1]
        for result in all_results:
            ax2.plot(result['val_losses'], label=f"Fold {result['fold']}")
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Validation accuracy
        ax3 = axes[1, 0]
        for result in all_results:
            ax3.plot(result['val_accuracies'], label=f"Fold {result['fold']}")
        ax3.set_title('Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True)
        
        # Learning rate (if available)
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, 'Model Performance Summary\n\n' + 
                f'Best Avg Val Acc: {np.mean([r["best_val_acc"] for r in all_results]):.2f}%\n' +
                f'Best Avg F1: {np.mean([r["f1"] for r in all_results]):.4f}',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, all_results):
        """Plot confusion matrices for all folds"""
        n_folds = len(all_results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Confusion Matrices Across All Folds', fontsize=16)
        
        for i, result in enumerate(all_results):
            row = i // 3
            col = i % 3
            
            if row < 2:  # Only plot first 6 folds
                cm = result['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['NOT LIVE', 'LIVE'],
                           yticklabels=['NOT LIVE', 'LIVE'],
                           ax=axes[row, col])
                axes[row, col].set_title(f'Fold {result["fold"]} (Acc: {result["best_val_acc"]:.1f}%)')
                axes[row, col].set_xlabel('Predicted')
                axes[row, col].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_folds, 6):
            row = i // 3
            col = i % 3
            if row < 2:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_ensemble_model(self, all_results, final_results=None):
        """Save ensemble model combining all fold models"""
        print("\n💾 Saving ensemble model...")
        
        # Find best performing fold (from validation)
        best_fold = max(all_results, key=lambda x: x['best_val_acc'])
        best_fold_num = best_fold['fold']
        
        # Copy best model as ensemble model
        best_model_path = os.path.join(MODEL_SAVE_DIR, f"fold_{best_fold_num}_best.pth")
        ensemble_path = os.path.join(MODEL_SAVE_DIR, "ensemble_model.pth")
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, ensemble_path)
            print(f"✅ Saved ensemble model from Fold {best_fold_num} (Best Val Acc: {best_fold['best_val_acc']:.2f}%)")
        
        # Save training summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'k_folds': K_FOLDS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'best_fold': best_fold_num,
            'best_val_acc': best_fold['best_val_acc'],
            'all_results': all_results
        }
        
        # Add final test results if available
        if final_results:
            summary['final_test_results'] = final_results
            print(f"✅ Final Test Accuracy: {final_results['ensemble']['test_accuracy']:.2f}%")
        
        summary_path = os.path.join(MODEL_SAVE_DIR, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Saved training summary to {summary_path}")

def main():
    """Main training function"""
    print("🏆 Live vs Non-Live Detection Model Training")
    print("=" * 60)
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        # Initialize trainer
        trainer = LiveModelTrainer()
        
        # Run k-fold training with holdout test set
        kfold_results, final_results = trainer.run_k_fold_training()
        
        if kfold_results is None:
            print("❌ Training failed!")
            return
        
        print("\n🎉 Training Complete!")
        print("=" * 60)
        print("📊 SUMMARY:")
        print(f"  K-Fold Validation Accuracy: {np.mean([r['best_val_acc'] for r in kfold_results]):.2f}% ± {np.std([r['best_val_acc'] for r in kfold_results]):.2f}%")
        if final_results:
            print(f"  Final Test Accuracy: {final_results['ensemble']['test_accuracy']:.2f}%")
            print(f"  Final Test F1-Score: {final_results['ensemble']['test_f1']:.4f}")
        
        print("\nCheck the 'live_filter/models' directory for saved models and plots.")
        
    except Exception as e:
        print(f"❌ Unexpected error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 