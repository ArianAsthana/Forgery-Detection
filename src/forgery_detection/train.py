import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from .model import ImprovedConvNextModel
from ..utils.data import EnhancedDocumentDataset
from ..utils.visualization import plot_training_curves, plot_confusion_matrix

def train_forgery_model(train_data, val_data, config):
    """
    Train the forgery detection model
    
    Args:
        train_data: List of (path, label) tuples for training
        val_data: List of (path, label) tuples for validation
        config: Dictionary containing training configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data augmentation and preprocessing
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = EnhancedDocumentDataset(train_data, transform=train_transform, use_ocr=config['use_ocr'])
    val_dataset = EnhancedDocumentDataset(val_data, transform=val_transform, use_ocr=config['use_ocr'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Initialize model
    model = ImprovedConvNextModel(num_classes=2, use_text_features=config['use_ocr'])
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            if config['use_ocr']:
                images, text_features, labels = [x.to(device) for x in batch]
                outputs = model(images, text_features)
            else:
                images, labels = [x.to(device) for x in batch]
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if config['use_ocr']:
                    images, text_features, labels = [x.to(device) for x in batch]
                    outputs = model(images, text_features)
                else:
                    images, labels = [x.to(device) for x in batch]
                    outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print metrics
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print('\nValidation Classification Report:')
        print(classification_report(val_labels, val_preds, target_names=['Real', 'Forged']))
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config['save_dir'], 'best_forgery_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    plot_confusion_matrix(val_labels, val_preds, ['Real', 'Forged'])
    
    return model 