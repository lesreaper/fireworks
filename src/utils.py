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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        noise_removed = cv2.medianBlur(gray, 3)

        _, thresh = cv2.threshold(noise_removed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        coords = cv2.findNonZero(thresh)
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle

        # Rotate image
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        success, buffer = cv2.imencode(".jpg", rotated)
        if not success:
            raise ValueError("Could not encode image to JPEG format.")

        image_base64 = base64.b64encode(buffer).decode("utf-8")
        return image_base64
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
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
