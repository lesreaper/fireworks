import logging
from .types import State, DocTypeResponse
from .BaseAgent import BaseAgent
import json
from pydantic import ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoctypeAgent(BaseAgent):
    def process(self, state: State) -> State:
        """Identify the type of document using LLM"""
        try:
            prompt = """You are a document analysis expert. Please analyze this image and identify the type of identification document.

            Return ONLY a JSON object with the following fields:
            - document_type: either "Passport" or "Driver's License"
            - confidence: a number between 0.0 and 1.0 indicating your confidence
            - detected_state: two-letter state code if it's a Driver's License (e.g., "CA", "NY")
            - additional_info: any other relevant details you notice

            Focus on key visual indicators like:
            - Document layout and format
            - Official seals or emblems
            - Header text or document title
            - State-specific features for Driver's Licenses
            """

            response = self.call_extraction_api(
                prompt=prompt,
                image_base64=state["image_data"],
                response_schema=DocTypeResponse.model_json_schema()
            )

            raw_output = response.choices[0].message.content
            try:
                parsed_output = json.loads(raw_output)
                logger.info(f"Received response: {parsed_output}")
                doc_type_response = DocTypeResponse(**parsed_output)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Error parsing document type response: {str(e)}")
                logger.error(f"Raw output was: {raw_output}")
                raise

            state["doc_type"] = doc_type_response.document_type
            if doc_type_response.detected_state:
                state["detected_state"] = doc_type_response.detected_state
            state["doc_type_confidence"] = doc_type_response.confidence
            state["total_tokens"] += response.usage.total_tokens

            if doc_type_response.confidence < 0.7:
                logger.warning(f"Low confidence in document type detection: {doc_type_response.confidence}")

            return state
        except Exception as e:
            logger.error(f"Document type detection failed: {str(e)}")
            state["error_message"] = f"Document type detection error: {str(e)}"
            return state
