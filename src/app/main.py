import os
import io
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import easyocr
from PIL import Image
from transformers import LayoutLMv3Processor
import logging
import traceback
import uuid

from src.classification.model import DocumentClassifier
from src.forgery_detection.model import ImprovedConvNextModel
from src.preprocessing import apply_clahe, apply_hsv_equalization, apply_lthe
from src.utils.visualization import generate_visualization

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Forgery Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mount static files and templates
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "templates"))

# Create necessary directories
os.makedirs(os.path.join(ROOT_DIR, "static/uploads"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "static/results"), exist_ok=True)

try:
    # Initialize models and processors
    logger.info("Initializing models and processors...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load forgery detection model
    logger.info("Loading forgery detection model...")
    forgery_model = ImprovedConvNextModel(num_classes=2, use_text_features=True)
    model_path = "models/enhanced_forgery_model_best.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info("Found model file, loading state dict...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        logger.info(f"Loaded state dict type: {type(state_dict)}")
        
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                logger.info("Loading from model_state_dict key")
                forgery_model.load_state_dict(state_dict["model_state_dict"])
            elif "state_dict" in state_dict:
                logger.info("Loading from state_dict key")
                forgery_model.load_state_dict(state_dict["state_dict"])
            elif all(k.startswith(('backbone.', 'features.', 'classifier.', 'text_processor.')) for k in state_dict.keys()):
                logger.info("Loading direct state dict")
                forgery_model.load_state_dict(state_dict)
            else:
                logger.error(f"Unexpected state dict keys: {list(state_dict.keys())}")
                raise ValueError("Invalid state dict format")
        else:
            logger.error(f"Expected dict but got {type(state_dict)}")
            raise ValueError("Invalid state dict format")
        
        logger.info("Model state dict loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model state dict: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    forgery_model = forgery_model.to(device)
    forgery_model.eval()
    logger.info("Forgery detection model loaded successfully")

    # Load document classifier
    logger.info("Loading document classifier...")
    classifier = DocumentClassifier(num_labels=18)
    if os.path.exists("models/best_classifier_model"):
        classifier = DocumentClassifier.from_pretrained("models/best_classifier_model", num_labels=18)
    classifier = classifier.to(device)
    classifier.eval()
    logger.info("Document classifier loaded successfully")

    # Initialize processors
    logger.info("Initializing processors...")
    layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
    ocr_reader = easyocr.Reader(['en'])
    logger.info("Processors initialized successfully")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Label mappings
doc_type_labels = {
    0: "ID", 1: "advertisement", 2: "budget", 3: "email", 
    4: "file_folder", 5: "form", 6: "handwritten", 7: "invoice",
    8: "letter", 9: "memo", 10: "news_article", 11: "presentation",
    12: "questionnaire", 13: "receipt", 14: "resume", 
    15: "scientific_publication", 16: "scientific_report", 17: "specification"
}

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Apply preprocessing
        img = apply_clahe(img)
        img = apply_hsv_equalization(img)
        
        return img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_text_features(img):
    """Extract text features for forgery detection"""
    try:
        results = ocr_reader.readtext(img)
        text_features = []
        
        for (bbox, text, confidence) in results:
            x1, y1 = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
            x2, y2 = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
            
            text_region = img[y1:y2, x1:x2]
            if text_region.size > 0:
                gray_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray_region, cv2.CV_64F).var()
                edges = cv2.Canny(gray_region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                text_features.extend([confidence, sharpness, edge_density])
        
        # Pad or truncate features
        target_size = 30
        if len(text_features) < target_size:
            text_features.extend([0.0] * (target_size - len(text_features)))
        else:
            text_features = text_features[:target_size]
        
        return np.array(text_features, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error in extract_text_features: {str(e)}")
        logger.error(traceback.format_exc())
        return np.zeros(30, dtype=np.float32)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """Serve the about page"""
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/detect", response_class=HTMLResponse)
async def detect_page(request: Request):
    """Serve the detection page"""
    return templates.TemplateResponse("detect.html", {
        "request": request,
        "result": None,
        "highlighted_img": None,
        "original_img": None
    })

@app.post("/detect")
async def detect_forgery(request: Request, document: UploadFile = File(...)):
    """Handle document upload and forgery detection"""
    try:
        logger.info("Starting document analysis...")
        
        # Read and save the uploaded file
        contents = await document.read()
        file_extension = os.path.splitext(document.filename)[1]
        original_filename = f"{uuid.uuid4()}{file_extension}"
        original_path = os.path.join(ROOT_DIR, "static", "uploads", original_filename)
        
        with open(original_path, "wb") as f:
            f.write(contents)
        
        logger.info("Processing uploaded image...")
        # Process the image
        img = preprocess_image(contents)
        
        # Extract text features for forgery detection
        logger.info("Extracting text features...")
        text_features = extract_text_features(img)
        text_features = torch.tensor(text_features).unsqueeze(0).to(device)
        
        # Prepare image for forgery detection
        logger.info("Preparing image for forgery detection...")
        forgery_input = cv2.resize(img, (224, 224))
        forgery_input = torch.tensor(forgery_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
        forgery_input = forgery_input / 255.0
        
        # Get forgery prediction
        logger.info("Running forgery detection...")
        with torch.no_grad():
            forgery_output = forgery_model(forgery_input, text_features)
            forgery_probs = torch.softmax(forgery_output, dim=1)
            forgery_pred = torch.argmax(forgery_probs, dim=1).item()
            forgery_confidence = forgery_probs[0][forgery_pred].item()
        
        # Generate visualization
        logger.info("Generating visualization...")
        try:
            # Get the preprocessed image for visualization
            vis_input = forgery_input[0]  # Get the first image
            visualization = generate_visualization(
                model=forgery_model,
                image=vis_input
            )
            
            # Save visualization
            result_filename = f"result_{uuid.uuid4()}.png"
            result_path = os.path.join(ROOT_DIR, "static", "results", result_filename)
            
            # Ensure visualization is in the correct format
            if isinstance(visualization, torch.Tensor):
                visualization = visualization.cpu().numpy()
            
            # Save the visualization
            cv2.imwrite(result_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            logger.info("Visualization generated successfully")
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            result_filename = None
        
        # Get document type prediction
        logger.info("Running document classification...")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = ocr_reader.readtext(rgb_img)
        
        words = []
        boxes = []
        for bbox, text, conf in results:
            if conf > 0.5:
                words.append(text)
                x1, y1 = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
                x2, y2 = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
                boxes.append([x1, y1, x2, y2])
        
        if not words:
            words = [""]
            boxes = [[0, 0, 0, 0]]
        
        encoding = layoutlm_processor(
            rgb_img,
            words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            doc_outputs = classifier(**encoding)
            doc_probs = torch.softmax(doc_outputs.logits, dim=1)
            doc_pred = torch.argmax(doc_probs, dim=1).item()
            doc_confidence = doc_probs[0][doc_pred].item()
        
        # Prepare result message with HTML formatting
        result_message = f"""
        <div class='result-details'>
            <div class='result-item {"forgery-detected" if forgery_pred else "no-forgery"}'>
                <h3>Forgery Detection</h3>
                <p class='result-status'>Status: <strong>{"FORGED" if forgery_pred else "AUTHENTIC"}</strong></p>
                <p class='confidence'>Confidence: {forgery_confidence:.2%}</p>
            </div>
            <div class='result-item'>
                <h3>Document Classification</h3>
                <p class='doc-type'>Type: <strong>{doc_type_labels[doc_pred]}</strong></p>
                <p class='confidence'>Confidence: {doc_confidence:.2%}</p>
            </div>
            <div class='result-item'>
                <h3>Visualization Explanation</h3>
                <p>The highlighted regions in the image show areas that influenced the forgery detection decision. 
                Brighter colors indicate stronger influence on the model's decision.</p>
            </div>
        </div>
        """
        
        logger.info("Analysis completed successfully")
        return templates.TemplateResponse("detect.html", {
            "request": request,
            "result": result_message,
            "highlighted_img": result_filename,
            "original_img": original_filename
        })
        
    except Exception as e:
        logger.error(f"Error in detect_forgery: {str(e)}")
        logger.error(traceback.format_exc())
        return templates.TemplateResponse("detect.html", {
            "request": request,
            "result": f"<div class='error-message'>Error processing document: {str(e)}</div>",
            "highlighted_img": None,
            "original_img": None
        })

@app.get("/test")
async def test_endpoint():
    """Test endpoint to check if server is running"""
    return {"status": "Server is running", "device": str(device)}

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return {"message": "Document Forgery Detection API is running. Use /analyze endpoint to analyze documents."} 