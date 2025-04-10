import logging
from .BaseAgent import BaseAgent
from .types import State, IdentificationResult
import re

logger = logging.getLogger(__name__)


class StateIdAgent(BaseAgent):
    """Agent for extracting information from state IDs"""

    def process(self, state: State) -> State:
        """Process the state ID and extract information"""
        try:
            logger.info("Starting state ID extraction with state:")
            logger.info(f"Current extraction attempts: {state['extraction_attempts']}")
            logger.info(f"Detected state: {state['detected_state']}")

            text_to_process = state.get("ocr_text", "")
            if not text_to_process:
                raise ValueError("No OCR text available for processing")

            logger.info(f"Processing text of length: {len(text_to_process)}")
            logger.info(f"Text sample: {text_to_process[:200]}...")

            prompt = f"""
            Please extract the following information from this driver's license text.
            The text was obtained via OCR from a driver's license image:

            {text_to_process}

            Please extract and return the information in the following JSON format:
            {{
                "document_type": "Driver's License",
                "document_number": "The license number",
                "name": {{
                    "first_name": "First name",
                    "middle_name": "Middle name if present",
                    "last_name": "Last name"
                }},
                "date_of_birth": "MM/DD/YYYY format",
                "issue_date": "MM/DD/YYYY format",
                "expiry_date": "MM/DD/YYYY format",
                "address": {{
                    "street": "Street address",
                    "city": "City",
                    "state": "Two-letter state code",
                    "zip_code": "ZIP code"
                }},
                "state": "Two-letter state code of the issuing state"
            }}

            Please ensure all dates are in MM/DD/YYYY format and the state codes are two letters.
            """

            response_data, tokens = self.call_extraction_api(
                prompt=prompt,
                image_base64=None,
                response_format={"type": "json_object", "schema": IdentificationResult.model_json_schema()}
            )

            extracted_data = IdentificationResult(**response_data).model_dump()

            # Rules check
            extracted_data["document_number"] = re.sub(r"^(DL|DLN|DL NO|DL#|DLN NO|D\.L\.|DL:)\s*", "", extracted_data["document_number"], flags=re.I)

            state["extracted_data"] = extracted_data
            state["extraction_attempts"] += 1
            state["extraction_tokens"] = tokens
            state["total_tokens"] += tokens

            logger.info("Successfully extracted data from state ID")
            logger.info(f"Extracted data: {state['extracted_data']}")

            return state

        except Exception as e:
            logger.error(f"State ID extraction failed: {str(e)}")
            state["error_message"] = f"State ID extraction error: {str(e)}"
            return state
