import torch
from torch import nn
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Config

class DocumentClassifier(nn.Module):
    def __init__(self, num_labels, model_name="microsoft/layoutlmv3-base"):
        super(DocumentClassifier, self).__init__()
        
        # Load LayoutLM configuration and model
        self.config = LayoutLMv3Config.from_pretrained(model_name, num_labels=num_labels)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )
        
        # Freeze some layers for fine-tuning
        modules_to_freeze = [
            self.model.layoutlmv3.embeddings,
            *self.model.layoutlmv3.encoder.layer[:8]
        ]
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )
        
        return outputs
    
    def save_pretrained(self, path):
        """Save the model to the specified path"""
        self.model.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path, num_labels):
        """Load a pretrained model from the specified path"""
        instance = cls(num_labels=num_labels)
        instance.model = LayoutLMv3ForSequenceClassification.from_pretrained(path)
        return instance 