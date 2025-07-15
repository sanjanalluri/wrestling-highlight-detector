"""
Wrestling Highlight Detection with 3D CNN
Handles data preparation, model training, and inference for highlight detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from database.database_manager import DatabaseManager
import pandas as pd

# Configuration
NUM_FRAMES = 16  # Number of frames to sample from each clip
FRAME_SIZE = (224, 224)  # Resize frames to this size
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WrestlingDataset(Dataset):
    """Dataset for wrestling highlight clips"""
    
    def __init__(self, clip_paths, labels, transform=None):
        self.clip_paths = clip_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.clip_paths)
    
    def __getitem__(self, idx):
        clip_path = self.clip_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess frames
        frames = self.load_frames(clip_path)
        
        if self.transform:
            frames = self.transform(frames)
        
        return torch.FloatTensor(frames), torch.LongTensor([label])
    
    def load_frames(self, clip_path):
        """Load and sample frames from video clip"""
        cap = cv2.VideoCapture(clip_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"Warning: No frames found in {clip_path}")
            cap.release()
            return np.zeros((NUM_FRAMES, 3, *FRAME_SIZE))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames-1, NUM_FRAMES, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize and normalize
                frame = cv2.resize(frame, FRAME_SIZE)
                frame = frame / 255.0  # Normalize to [0, 1]
                frames.append(frame)
            else:
                # If frame read fails, use zero frame
                frames.append(np.zeros((*FRAME_SIZE, 3)))
        
        cap.release()
        
        # Convert to (C, T, H, W) format for 3D CNN
        frames = np.array(frames)  # (T, H, W, C)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        
        return frames

class Wrestling3DCNN(nn.Module):
    """3D CNN for wrestling highlight detection"""
    
    def __init__(self, num_classes=2):
        super(Wrestling3DCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv3d_1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3d_4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Calculate the size after convolutions
        # Starting with (3, 16, 224, 224)
        # After pool1: (32, 16, 112, 112)
        # After pool2: (64, 8, 56, 56)
        # After pool3: (128, 4, 28, 28)
        # After pool4: (256, 2, 14, 14)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 3D Convolutional layers
        x = F.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv3d_4(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class WrestlingModelTrainer:
    """Handles training and evaluation of the wrestling highlight model"""
    
    def __init__(self):
        self.model = Wrestling3DCNN().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.db_manager = DatabaseManager()
    
    def prepare_data(self):
        """Prepare training data from database"""
        print("📊 Preparing training data from database...")
        
        # Get all labeled clips with gender information for analysis
        conn = self.db_manager.get_connection()
        query = """
        SELECT c.filepath, l.label, v.gender
        FROM clips c 
        JOIN labels l ON c.id = l.clip_id 
        JOIN videos v ON c.video_id = v.id
        WHERE l.label IN ('Highlight', 'Not Highlight')
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("❌ No labeled clips found in database!")
            return None, None, None, None
        
        # Show gender distribution for transparency
        gender_counts = df['gender'].value_counts()
        print(f"📊 Gender distribution in training data:")
        for gender, count in gender_counts.items():
            print(f"   {gender}: {count} clips")
        
        # Convert labels to numeric
        label_map = {'Not Highlight': 0, 'Highlight': 1}
        df['label_numeric'] = df['label'].map(label_map)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['filepath'].values, 
            df['label_numeric'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['label_numeric'].values
        )
        
        # Create datasets
        train_dataset = WrestlingDataset(X_train, y_train)
        test_dataset = WrestlingDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"✅ Prepared {len(X_train)} training clips and {len(X_test)} test clips")
        print(f"   Training: {sum(y_train)} highlights, {len(y_train) - sum(y_train)} not highlights")
        print(f"   Test: {sum(y_test)} highlights, {len(y_test) - sum(y_test)} not highlights")
        
        return train_loader, test_loader, X_train, X_test
    
    def train(self, train_loader, test_loader):
        """Train the model"""
        print(f"🚀 Starting training on {DEVICE}...")
        
        train_losses = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(EPOCHS):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.squeeze().to(DEVICE)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Evaluation phase
            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(DEVICE), target.squeeze().to(DEVICE)
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            accuracy = 100. * correct / total
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Accuracy: {accuracy:.2f}%")
            print("-" * 50)
        
        # Save the trained model
        torch.save(self.model.state_dict(), 'wrestling_highlight_model.pth')
        print("✅ Model saved as 'wrestling_highlight_model.pth'")
        
        return train_losses, test_losses, test_accuracies
    
    def evaluate(self, test_loader):
        """Evaluate the trained model"""
        print("🔍 Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.squeeze().to(DEVICE)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Print classification report
        print("\n📊 Classification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=['Not Highlight', 'Highlight']))
        
        # Print confusion matrix
        print("\n📈 Confusion Matrix:")
        cm = confusion_matrix(all_targets, all_predictions)
        print(cm)
        
        return all_predictions, all_targets
    
    def predict_clip(self, clip_path):
        """Predict highlight probability for a single clip"""
        self.model.eval()
        
        # Load and preprocess clip
        dataset = WrestlingDataset([clip_path], [0])  # Dummy label
        frames, _ = dataset[0]
        frames = frames.unsqueeze(0).to(DEVICE)  # Add batch dimension
        
        with torch.no_grad():
            output = self.model(frames)
            probabilities = F.softmax(output, dim=1)
            highlight_prob = probabilities[0][1].item()
        
        return highlight_prob

def main():
    """Main training pipeline"""
    print("🏆 Wrestling Highlight Detection Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = WrestlingModelTrainer()
    
    # Prepare data
    train_loader, test_loader, X_train, X_test = trainer.prepare_data()
    
    if train_loader is None:
        print("❌ No data available for training. Please label some clips first!")
        return
    
    # Train model
    train_losses, test_losses, test_accuracies = trainer.train(train_loader, test_loader)
    
    # Evaluate model
    predictions, targets = trainer.evaluate(test_loader)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("✅ Training complete! Model ready for inference.")

if __name__ == "__main__":
    main() 