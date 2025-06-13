import cv2
import numpy as np

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel = clahe.apply(l_channel)
        lab[:,:,0] = l_channel
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

def apply_hsv_equalization(img):
    """Apply histogram equalization in HSV color space"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_lthe(img):
    """Apply Local Threshold Histogram Equalization"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter for noise reduction
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply adaptive thresholding
    block_size = 11
    C = 2
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)
    
    return binary 