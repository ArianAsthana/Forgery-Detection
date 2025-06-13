import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import logging

logger = logging.getLogger(__name__)

class ImprovedConvNextModel(nn.Module):
    def __init__(self, num_classes=2, use_text_features=False, dropout_rate=0.3):
        super(ImprovedConvNextModel, self).__init__()
        
        # Load pretrained ConvNeXt
        try:
            logger.info("Loading ConvNeXt model...")
            self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            logger.info("ConvNeXt model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ConvNeXt model: {str(e)}")
            raise
        
        # Remove the original classifier
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Enhanced classifier with additional layers
        feature_dim = 768
        self.use_text_features = use_text_features
        
        if use_text_features:
            # Text feature processor
            self.text_processor = nn.Sequential(
                nn.Linear(30, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            feature_dim += 32
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, text_features=None):
        # Extract visual features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Combine with text features if available
        if self.use_text_features and text_features is not None:
            text_features = self.text_processor(text_features)
            x = torch.cat([x, text_features], dim=1)
        
        # Apply classifier
        x = self.classifier(x)
        return x 