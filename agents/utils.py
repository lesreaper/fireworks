import base64
import cv2
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def preprocess_and_encode_image(image_path: str) -> str:
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at {image_path} could not be read.")

        # Directly encode the original image to JPEG and then to Base64
        success, buffer = cv2.imencode(".jpg", img)
        if not success:
            raise ValueError("Could not encode image to JPEG format.")

        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        logger.error(f"Error in encoding: {str(e)}")
        raise


class FileSaver:
    @staticmethod
    def save_output(result: dict, default_filename: str = "output.json") -> str:
        output_dir = "base_src/output"
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, default_filename)

        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

        return filename
