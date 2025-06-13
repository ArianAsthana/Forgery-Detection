import os
import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from .model import DocumentClassifier
from ..utils.data import DocumentDataset
from ..utils.visualization import plot_training_curves, plot_confusion_matrix

def train_classifier(train_data, val_data, config):
    """
    Train the document classifier model
    
    Args:
        train_data: List of (path, label) tuples for training
        val_data: List of (path, label) tuples for validation
        config: Dictionary containing training configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize processor and model
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    model = DocumentClassifier(num_labels=config['num_labels'])
    model = model.to(device)
    
    # Create datasets
    train_dataset = DocumentDataset(train_data, processor, max_length=config['max_length'])
    val_dataset = DocumentDataset(val_data, processor, max_length=config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            train_preds.extend(predictions.cpu().numpy())
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
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bbox = batch['bbox'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print metrics
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print('\nValidation Classification Report:')
        print(classification_report(val_labels, val_preds))
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config['save_dir'], 'best_classifier_model')
            model.save_pretrained(model_path)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    plot_confusion_matrix(val_labels, val_preds)
    
    return model 