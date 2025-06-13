def check_dependencies():
    dependencies = [
        'torch',
        'torchvision',
        'numpy',
        'cv2',
        'PIL',
        'easyocr',
        'transformers',
        'sklearn',
        'matplotlib',
        'seaborn',
        'fastapi',
        'uvicorn',
        'tqdm',
        'pytorch_grad_cam'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'PIL':
                from PIL import Image
            else:
                __import__(dep)
            print(f"✅ {dep} is installed")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep} is missing")
    
    if missing:
        print("\n❌ Some dependencies are missing. Please install them using:")
        if 'cv2' in missing:
            missing[missing.index('cv2')] = 'opencv-python'
        if 'PIL' in missing:
            missing[missing.index('PIL')] = 'Pillow'
        print(f"pip install {' '.join(missing)}")
    else:
        print("\n✅ All dependencies are installed!")

if __name__ == "__main__":
    check_dependencies() 