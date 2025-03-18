import logging
from .BaseAgent import BaseAgent
from .types import State

logger = logging.getLogger(__name__)


class PassportAgent(BaseAgent):
    """Agent for extracting information from passports"""

    def process(self, state: State) -> State:
        """Process the passport and extract information"""
        try:
            logger.info("Starting passport extraction:")
            logger.info(f"Current extraction attempts: {state['extraction_attempts']}")

            text_to_process = state.get("ocr_text", "")
            if not text_to_process:
                raise ValueError("No OCR text available for processing")

            logger.info(f"Processing text of length: {len(text_to_process)}")
            logger.info(f"Text sample: {text_to_process[:200]}...")

            # Construct extraction prompt
            prompt = f"""
            You are an expert in extracting information from United States passports.
            Please analyze this text carefully and extract the required information.

            Text from passport:
            {text_to_process}

            Extract and return ONLY these fields in this exact format:
            {{
                "document_type": "Passport",
                "document_number": "The passport number (9-10 digits)",
                "name": {{
                    "first_name": "First name in CAPS",
                    "middle_name": "Middle name in CAPS if present",
                    "last_name": "Last name in CAPS"
                }},
                "date_of_birth": "MM/DD/YYYY format",
                "issue_date": "MM/DD/YYYY format",
                "expiry_date": "MM/DD/YYYY format",
                "nationality": "US",
                "place_of_birth": "City, Country format",
                "sex": "M or F"
            }}

            Important rules:
            1. ALL dates must be in MM/DD/YYYY format
            2. ALL names must be in CAPS
            3. Use "US" for nationality (not "USA")
            4. Place of birth MUST be included and in "City, Country" format
            5. DO NOT include an address field - passports don't have addresses
            6. DO NOT include a state field - passports don't use this
            7. Sex/Gender must be "M" or "F"
            8. Document number should be 9-10 digits
            """

            response_data, tokens = self.call_extraction_api(
                prompt=prompt,
                image_base64=None,
                response_format={"type": "json_object"}
            )

            extracted_dict = response_data

            if not extracted_dict.get("place_of_birth"):
                logger.warning("Place of birth not found in extraction")
                # Try to construct it from any city/state information
                place_parts = []
                if extracted_dict.get("city"):
                    place_parts.append(extracted_dict["city"])
                if extracted_dict.get("state"):
                    place_parts.append(extracted_dict["state"])
                if place_parts:
                    extracted_dict["place_of_birth"] = ", ".join(place_parts) + ", U.S.A"
                    logger.info(f"Created place_of_birth: {extracted_dict['place_of_birth']}")

            if "address" in extracted_dict:
                del extracted_dict["address"]
                logger.info("Removed address field from passport data")

            if "state" in extracted_dict:
                del extracted_dict["state"]
                logger.info("Removed state field from passport data")

            state["extracted_data"] = extracted_dict
            state["extraction_attempts"] += 1
            state["extraction_tokens"] = tokens
            state["total_tokens"] = state.get("total_tokens", 0) + tokens

            logger.info("Successfully extracted passport data")
            logger.info(f"Extracted data: {state['extracted_data']}")

            return state

        except Exception as e:
            logger.error(f"Passport extraction failed: {str(e)}")
            state["error_message"] = f"Passport extraction error: {str(e)}"
            return state
