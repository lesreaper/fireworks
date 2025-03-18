import logging
from typing import Dict
import requests
import base64
from .BaseAgent import BaseAgent
from .types import State

logger = logging.getLogger(__name__)


class VisionOCRAgent(BaseAgent):
    """Agent for performing OCR using Google Cloud Vision API"""

    def __init__(self, api_client, api_key: str = None):
        """
        Initialize the Vision OCR Agent

        Args:
            api_client: The Fireworks API client (unused but required for BaseAgent)
            api_key: Google Cloud API key. If not provided, will try to use service account credentials
        """
        super().__init__(api_client)
        self.api_key = api_key
        self.base_url = "https://vision.googleapis.com/v1/images:annotate"
        logger.info(f"Initialized VisionOCRAgent with API key: {'present' if api_key else 'not present'}")

    def _extract_text_from_response(self, response: Dict) -> tuple[str, float]:
        """Extract clean text and confidence from API response"""
        if self.api_key:
            if "responses" not in response or not response["responses"]:
                return "", 0.0

            result = response["responses"][0]
            if "fullTextAnnotation" in result:
                text = result["fullTextAnnotation"]["text"]
                confidence = result.get("confidence", 0.0)
            elif "textAnnotations" in result and result["textAnnotations"]:
                text = result["textAnnotations"][0]["description"]
                confidence = result.get("confidence", 0.95)
            else:
                return "", 0.0
        else:
            if hasattr(response, 'full_text_annotation') and response.full_text_annotation:
                text = response.full_text_annotation.text
                confidence = response.full_text_annotation.pages[0].confidence if response.full_text_annotation.pages else 0.0
            elif hasattr(response, 'text_annotations') and response.text_annotations:
                text = response.text_annotations[0].description
                confidence = 0.95
            else:
                return "", 0.0

        return text, confidence

    def _make_request(self, image_path: str, feature_type: str) -> Dict:
        """Make a request to the Vision API"""
        try:
            logger.info(f"Making Vision API request for {feature_type} on {image_path}")

            with open(image_path, "rb") as image_file:
                content = base64.b64encode(image_file.read()).decode("utf-8")
            logger.debug(f"Successfully read and encoded image from {image_path}")

            request_body = {
                "requests": [{
                    "image": {
                        "content": content
                    },
                    "features": [{
                        "type": feature_type,
                        "maxResults": 50
                    }]
                }]
            }

            if self.api_key:
                logger.info("Using API key authentication")
                response = requests.post(
                    f"{self.base_url}?key={self.api_key}",
                    json=request_body
                )

                if not response.ok:
                    error_msg = f"Vision API request failed: {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                return response.json()
            else:
                logger.info("Using service account authentication")
                from google.cloud import vision
                client = vision.ImageAnnotatorClient()
                image = vision.Image(content=base64.b64decode(content))

                if feature_type == "DOCUMENT_TEXT_DETECTION":
                    return client.document_text_detection(image=image)
                else:
                    return client.text_detection(image=image)

        except Exception as e:
            logger.error(f"Error making Vision API request: {str(e)}")
            raise

    def process(self, state: State) -> State:
        """Process the document image and extract text"""
        try:
            logger.info(f"Starting OCR processing for image: {state.get('image_path', 'No image path found')}")

            text_result = self._make_request(state["image_path"], "TEXT_DETECTION")
            doc_result = self._make_request(state["image_path"], "DOCUMENT_TEXT_DETECTION")

            text, text_confidence = self._extract_text_from_response(text_result)
            doc_text, doc_confidence = self._extract_text_from_response(doc_result)

            final_text = doc_text if doc_text else text
            final_confidence = doc_confidence if doc_text else text_confidence

            logger.info(f"Extracted text length: {len(final_text)} characters")
            logger.info(f"Text sample: {final_text[:1000]}...")

            state.update({
                "ocr_text": final_text,
                "ocr_confidence": final_confidence,
                "image_data": final_text
            })

            logger.info(f"Successfully processed document with confidence: {final_confidence:.2%}")
            return state

        except Exception as e:
            logger.error(f"Error in VisionOCRAgent: {str(e)}")
            state["error_message"] = f"OCR error: {str(e)}"
            return state
