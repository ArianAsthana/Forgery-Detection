import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import easyocr

class EnhancedDocumentDataset(Dataset):
    def __init__(self, samples, transform=None, use_ocr=False):
        self.samples = samples
        self.transform = transform
        self.use_ocr = use_ocr
        if use_ocr:
            self.ocr_reader = easyocr.Reader(['en'])
    
    def __len__(self):
        return len(self.samples)
    
    def extract_text_features(self, img):
        """Extract text-based features that might indicate forgery"""
        try:
            results = self.ocr_reader.readtext(img)
            text_features = []
            
            for (bbox, text, confidence) in results:
                # Extract bounding box coordinates
                x1, y1 = int(min([point[0] for point in bbox])), int(min([point[1] for point in bbox]))
                x2, y2 = int(max([point[0] for point in bbox])), int(max([point[1] for point in bbox]))
                
                # Extract text region
                text_region = img[y1:y2, x1:x2]
                if text_region.size > 0:
                    # Analyze text region for potential forgery indicators
                    gray_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate sharpness (Laplacian variance)
                    sharpness = cv2.Laplacian(gray_region, cv2.CV_64F).var()
                    
                    # Calculate edge density
                    edges = cv2.Canny(gray_region, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    
                    text_features.extend([confidence, sharpness, edge_density])
            
            # Pad or truncate to fixed size
            target_size = 30  # 10 text regions * 3 features each
            if len(text_features) < target_size:
                text_features.extend([0.0] * (target_size - len(text_features)))
            else:
                text_features = text_features[:target_size]
                
            return np.array(text_features, dtype=np.float32)
        except:
            return np.zeros(30, dtype=np.float32)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        
        if img is None:
            raise ValueError(f"Failed to load image at {path}")
        
        # Extract text features if OCR is enabled
        text_features = None
        if self.use_ocr:
            text_features = self.extract_text_features(img)
        
        if self.transform:
            img = self.transform(img)
        
        if text_features is not None:
            return img, torch.tensor(text_features), label
        else:
            return img, label

class DocumentDataset(Dataset):
    def __init__(self, samples, processor, max_length=512):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.ocr_reader = easyocr.Reader(['en'])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Read image
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform OCR
        results = self.ocr_reader.readtext(image)
        
        # Extract words and boxes
        words = []
        boxes = []
        
        for bbox, text, conf in results:
            if conf > 0.5:  # Filter low confidence predictions
                words.append(text)
                # Convert to [x1, y1, x2, y2] format and normalize
                x1, y1 = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
                x2, y2 = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
                boxes.append([x1, y1, x2, y2])
        
        # Handle empty OCR results
        if not words:
            words = [""]
            boxes = [[0, 0, 0, 0]]
        
        # Prepare inputs for LayoutLM
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)
        
        # Add label
        encoding['labels'] = torch.tensor(label)
        
        return encoding 