import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import torch
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

class ConvNextWrapper(torch.nn.Module):
    """Wrapper class for ConvNeXt model to make it compatible with CAM"""
    def __init__(self, model):
        super(ConvNextWrapper, self).__init__()
        self.model = model
        self.target_layers = [model.features[-1]]  # Last layer of features

    def forward(self, x):
        # Forward pass through features
        x = self.model.features(x)
        # Forward pass through avgpool
        x = self.model.avgpool(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Forward pass through classifier
        x = self.model.classifier(x)
        return x

def generate_visualization(model, image):
    """Generate ScoreCAM visualization for model decision"""
    # Create model wrapper
    model_wrapper = ConvNextWrapper(model)
    
    # Ensure model is in eval mode
    model_wrapper.eval()

    # Initialize ScoreCAM with target_layers from wrapper
    cam = ScoreCAM(
        model=model_wrapper,
        target_layers=model_wrapper.target_layers,
        use_cuda=torch.cuda.is_available()
    )

    # Handle input image format
    if isinstance(image, torch.Tensor):
        # If it's a tensor, ensure it's on CPU and convert to numpy
        input_tensor = image.detach().cpu()
        # If it's a single channel or grayscale image, repeat to make it 3 channels
        if input_tensor.shape[0] == 1:
            input_tensor = input_tensor.repeat(3, 1, 1)
        # Ensure we have the right shape (C, H, W)
        if input_tensor.ndim == 2:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 3 and input_tensor.shape[0] != 3:
            input_tensor = input_tensor.permute(2, 0, 1)
    else:
        # If it's a numpy array, convert to tensor
        if image.ndim == 2:
            image = np.stack([image] * 3)
        elif image.ndim == 3 and image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        input_tensor = torch.from_numpy(image).float()

    # Add batch dimension if needed
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Normalize if not already in [0, 1]
    if input_tensor.max() > 1.0:
        input_tensor = input_tensor / 255.0

    # Create a normalized copy for visualization
    input_for_visualization = input_tensor[0].numpy()
    if input_for_visualization.shape[0] == 3:
        input_for_visualization = np.transpose(input_for_visualization, (1, 2, 0))

    # Move tensor to appropriate device
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model_wrapper = model_wrapper.cuda()

    # Get model prediction
    with torch.no_grad():
        outputs = model_wrapper(input_tensor)
        prediction = outputs.argmax(dim=1)
        
    # Create target for visualization
    targets = [ClassifierOutputTarget(prediction.item())]

    # Generate activation map
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # Create visualization
    visualization = show_cam_on_image(
        input_for_visualization,
        grayscale_cam[0],
        use_rgb=True
    )

    return visualization