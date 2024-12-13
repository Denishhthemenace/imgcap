from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import logging
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility to setup BLIP model
def setup_blip_model():
    try:
        logging.info("Initializing BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        logging.info("BLIP model initialized successfully.")
        return processor, model
    except Exception as e:
        logging.error(f"Failed to initialize BLIP model: {str(e)}")
        raise RuntimeError("Model setup failed")

processor, model = setup_blip_model()

# Route to display the homepage
@app.route('/')
def index():
    logging.info("Rendering index page.")
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering index page: {str(e)}")
        return "Error loading page.", 500

# Route to generate caption for an uploaded image
@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        logging.warning("Request missing image file.")
        return jsonify({"error": "No file provided"}), 400

    image_file = request.files['image']
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + image_file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image_file.save(filepath)
        logging.info(f"Image successfully saved to {filepath}")

        raw_image = Image.open(filepath).convert('RGB')
        inputs = processor(raw_image, "a photography of", return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        logging.info(f"Generated caption: {caption}")

        return jsonify({"caption": caption})
    except Exception as e:
        logging.error(f"Error generating caption: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                logging.info(f"Temporary file {filename} deleted.")
            except Exception as e:
                logging.warning(f"Failed to delete temporary file {filename}: {str(e)}")

# Route to display recent captions
def recent_captions():
    logging.info("Fetching recent captions.")
    captions = []  # Placeholder for future storage integration
    return jsonify({"captions": captions})

# Route to get app health status
@app.route('/health', methods=['GET'])
def health_check():
    logging.info("Health check endpoint called.")
    return jsonify({"status": "OK"})

# Route to view logs
@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        with open('app.log', 'r') as log_file:
            logs = log_file.read()
        logging.info("Logs successfully retrieved.")
        return logs, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        logging.error(f"Error retrieving logs: {str(e)}")
        return "Unable to retrieve logs.", 500

# Utility for periodic file cleanup
def cleanup_old_files():
    logging.info("Cleaning up old files.")
    try:
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_age.total_seconds() > 3600:
                    os.remove(file_path)
                    logging.info(f"Deleted old file: {file}")
    except Exception as e:
        logging.warning(f"Error during cleanup: {str(e)}")

# Route to delete log files
@app.route('/delete_logs', methods=['POST'])
def delete_logs():
    logging.info("Request to delete logs received.")
    try:
        open('app.log', 'w').close()
        logging.info("Logs successfully deleted.")
        return jsonify({"status": "Logs deleted"}), 200
    except Exception as e:
        logging.error(f"Error deleting logs: {str(e)}")
        return jsonify({"error": "Unable to delete logs"}), 500

# Route to list uploaded files
@app.route('/file_list', methods=['GET'])
def list_uploaded_files():
    logging.info("Fetching list of uploaded files.")
    try:
        files = os.listdir(UPLOAD_FOLDER)
        file_list = [f for f in files if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        return jsonify({"files": file_list}), 200
    except Exception as e:
        logging.error(f"Error listing files: {str(e)}")
        return jsonify({"error": "Unable to list files"}), 500

# Route to download a specific file
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    logging.info(f"Download request received for file: {filename}")
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            logging.warning(f"File not found: {filename}")
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logging.error(f"Error downloading file {filename}: {str(e)}")
        return jsonify({"error": "Unable to download file"}), 500

# Route to clear the upload folder
@app.route('/clear_folder', methods=['POST'])
def clear_upload_folder():
    logging.info("Request to clear upload folder received.")
    try:
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logging.info(f"Deleted file: {file}")
        return jsonify({"status": "Upload folder cleared"}), 200
    except Exception as e:
        logging.error(f"Error clearing upload folder: {str(e)}")
        return jsonify({"error": "Unable to clear folder"}), 500

# Improved UI rendering
@app.route('/ui', methods=['GET'])
def ui():
    logging.info("Serving enhanced UI page.")
    try:
        return render_template('ui.html')  # Assume enhanced UI HTML file
    except Exception as e:
        logging.error(f"Error loading UI page: {str(e)}")
        return "UI Page Not Available.", 500

if __name__ == '__main__':
    logging.info("Starting Flask application.")
    cleanup_old_files()
    app.run(debug=True)
