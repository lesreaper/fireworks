import logging
from .BaseAgent import BaseAgent
from .types import State, IdentificationResult

logger = logging.getLogger(__name__)


class PassportAgent(BaseAgent):
    """Agent for extracting information from passports"""

    def process(self, state: State) -> State:
        """Process the passport and extract information"""
        try:
            logger.info("Starting passport extraction:")
            logger.info(f"Current extraction attempts: {state['extraction_attempts']}")
            
            # Get the OCR text
            text_to_process = state.get("ocr_text", "")
            if not text_to_process:
                raise ValueError("No OCR text available for processing")
                
            logger.info(f"Processing text of length: {len(text_to_process)}")
            logger.info(f"Text sample: {text_to_process[:200]}...")

            # Construct prompt for information extraction
            prompt = f"""
            Please extract the following information from this passport text.
            The text was obtained via OCR from a passport image:

            {text_to_process}

            Please extract and return the information in the following JSON format:
            {{
                "document_type": "Passport",
                "document_number": "The passport number",
                "name": {{
                    "first_name": "First name",
                    "middle_name": "Middle name if present",
                    "last_name": "Last name"
                }},
                "date_of_birth": "MM/DD/YYYY format",
                "issue_date": "MM/DD/YYYY format",
                "expiry_date": "MM/DD/YYYY format",
                "nationality": "USA",
                "state": "Two-letter state code of birth place if available"
            }}

            Please ensure:
            1. All dates are in MM/DD/YYYY format
            2. Names are in all caps
            3. Nationality is "USA" for US passports
            4. State code is two letters if available
            """

            # Call LLM for information extraction
            response_data, tokens = self.call_extraction_api(
                prompt=prompt,
                image_base64=None,  # No need to send image again
                response_format={"type": "json_object", "schema": IdentificationResult.model_json_schema()}
            )

            # Parse response and update state
            extracted_data = IdentificationResult(**response_data)
            
            # Update state
            state["extracted_data"] = extracted_data.model_dump()
            state["extraction_attempts"] += 1
            state["extraction_tokens"] = tokens
            state["total_tokens"] += tokens

            logger.info(f"Successfully extracted data from passport")
            logger.info(f"Extracted data: {state['extracted_data']}")
            
            return state

        except Exception as e:
            logger.error(f"Passport extraction failed: {str(e)}")
            state["error_message"] = f"Passport extraction error: {str(e)}"
            return state
