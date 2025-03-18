from google.cloud import vision
import logging
from typing import Dict, Any
import io

logger = logging.getLogger(__name__)


class VisionAPI:
    def __init__(self):
        """Initialize the Google Cloud Vision client"""
        try:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Successfully initialized Google Cloud Vision client")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision client: {str(e)}")
            raise

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from an image using Google Cloud Vision API

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing:
                - full_text: Complete extracted text
                - blocks: List of text blocks with positions
                - confidence: Overall confidence score
        """
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            if not texts:
                logger.warning("No text detected in the image")
                return {
                    "full_text": "",
                    "blocks": [],
                    "confidence": 0.0
                }

            full_text = texts[0].description

            blocks = []
            for text in texts[1:]:
                vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                blocks.append({
                    "text": text.description,
                    "confidence": text.confidence if hasattr(text, 'confidence') else None,
                    "bounds": vertices
                })

            confidences = [block["confidence"] for block in blocks if block["confidence"] is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            result = {
                "full_text": full_text,
                "blocks": blocks,
                "confidence": avg_confidence
            }

            logger.info(f"Successfully extracted text with confidence: {avg_confidence:.2%}")
            return result

        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            raise

    def detect_document(self, image_path: str) -> Dict[str, Any]:
        """
        Perform document text detection using Google Cloud Vision API

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing structured document information
        """
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Perform document text detection
            response = self.client.document_text_detection(image=image)
            document = response.full_text_annotation

            # Extract pages, blocks, paragraphs, words
            result = {
                "full_text": document.text,
                "pages": [],
                "confidence": document.pages[0].confidence if document.pages else 0.0
            }

            for page in document.pages:
                page_data = {
                    "blocks": [],
                    "confidence": page.confidence
                }

                for block in page.blocks:
                    block_data = {
                        "text": "",
                        "paragraphs": [],
                        "confidence": block.confidence
                    }

                    for paragraph in block.paragraphs:
                        para_text = ""
                        para_data = {
                            "words": [],
                            "confidence": paragraph.confidence
                        }

                        for word in paragraph.words:
                            word_text = "".join([
                                symbol.text for symbol in word.symbols
                            ])
                            para_text += word_text + " "
                            para_data["words"].append({
                                "text": word_text,
                                "confidence": word.confidence,
                                "bounds": [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                            })

                        block_data["text"] += para_text
                        block_data["paragraphs"].append(para_data)

                    page_data["blocks"].append(block_data)

                result["pages"].append(page_data)

            logger.info(f"Successfully performed document text detection with confidence: {result['confidence']:.2%}")
            return result

        except Exception as e:
            logger.error(f"Error performing document text detection: {str(e)}")
            raise

    def analyze_document(self, image_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive document analysis including text detection,
        layout analysis, and entity extraction

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing comprehensive document analysis
        """
        try:
            text_result = self.extract_text(image_path)
            doc_result = self.detect_document(image_path)
            result = {
                "text_extraction": text_result,
                "document_analysis": doc_result,
                "confidence": {
                    "text_confidence": text_result["confidence"],
                    "document_confidence": doc_result["confidence"],
                    "overall_confidence": (text_result["confidence"] + doc_result["confidence"]) / 2
                }
            }

            logger.info("Successfully performed comprehensive document analysis")
            return result

        except Exception as e:
            logger.error(f"Error performing document analysis: {str(e)}")
            raise
