from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from PIL import Image
import os

# Create Flask app (frontend)
flask_app = Flask(__name__)

# Create FastAPI app (main app)
fastapi_app = FastAPI()

# Paths for uploads and results
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
# -------- Flask routes --------

@flask_app.route("/")

def home():
    return render_template('home.html')


@flask_app.route("/about")
def about():
    return render_template("about.html")

@flask_app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        file = request.files.get("document")  # ✅ matches input name in HTML
        if file:
            filename = secure_filename(file.filename)  # Always sanitize filenames
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Call detection backend
            response = fastapi_app_detector(file_path)
            result_path = response.get("highlighted_image")
            result_label = response.get("label")

            return render_template(
                "detect.html",
                result=result_label,
                highlighted_img=result_path,
                original_img=filename  # ✅ Send original file name to show input image
            )

    # GET method: initial load
    return render_template("detect.html", result=None, highlighted_img=None, original_img=None)


# Dummy backend detector logic
def fastapi_app_detector(image_path):
    # For demo: just copy input image to result folder and mark as forged
    img = Image.open(image_path)
    result_filename = "result_" + os.path.basename(image_path)
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    img.save(result_path)

    # Return relative path for HTML to load
    return {"highlighted_image": "/" + result_path.replace("\\", "/"), "label": "Forgery Detected (Demo)"}

# -------- Mount Flask inside FastAPI --------
fastapi_app.mount("/", WSGIMiddleware(flask_app))

# -------- Run with uvicorn --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, reload=True)
