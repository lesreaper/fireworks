import logging
from .types import State, IdentificationResult, PersonName, Address
from .BaseAgent import BaseAgent
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StateIdAgent(BaseAgent):
    def process(self, state: State) -> State:
        """Extract information from state ID with enhanced logging"""
        try:
            # Log initial state
            logger.info("Starting state ID extraction with state:")
            logger.info(f"Current extraction attempts: {state.get('extraction_attempts', 0)}")
            logger.info(f"Detected state: {state.get('detected_state', 'Not detected')}")

            # Include detected state in prompt if available
            state_specific = f"This is a {state.get('detected_state', '')} Driver's License/State ID." if state.get('detected_state') else ""

            prompt = f"""
            {state_specific}
            Please analyze this Driver's License/State ID and extract the following information in JSON format:
            {{
                "document_type": "Driver's License",
                "document_number": "string",
                "name": {{
                    "first_name": "string",
                    "middle_name": "string (optional)",
                    "last_name": "string"
                }},
                "date_of_birth": "YYYY-MM-DD",
                "issue_date": "YYYY-MM-DD",
                "expiry_date": "YYYY-MM-DD",
                "address": {{
                    "street": "string",
                    "city": "string",
                    "state": "two-letter state code",
                    "zip_code": "5-digit string"
                }},
                "nationality": "USA",
                "state": "two-letter state code"
            }}

            Please ensure:
            - All dates are in YYYY-MM-DD format
            - The state code is a two-letter abbreviation
            - The zip code is a 5-digit string
            - Include "USA" as nationality for all US licenses
            - The state field should be the state of issuance
            """

            # Make API call
            response = self.call_extraction_api(
                prompt=prompt,
                image_base64=state["image_data"],
                response_schema=IdentificationResult.model_json_schema()
            )

            # Parse response
            extracted_data = json.loads(response.choices[0].message.content)
            tokens = response.usage.total_tokens

            # Log extracted data
            logger.info("Extraction completed. Results:")
            logger.info(f"Extracted data: {extracted_data}")
            logger.info(f"Tokens used: {tokens}")

            # Update state
            state["extracted_data"] = extracted_data
            state["extraction_attempts"] += 1
            state["total_tokens"] += tokens
            state["extraction_tokens"] = tokens

            # Log final state
            logger.info("Updated state after extraction:")
            logger.info(f"Total tokens: {state['total_tokens']}")
            logger.info(f"Extraction attempts: {state['extraction_attempts']}")

            return state
        except Exception as e:
            logger.error(f"State ID extraction failed: {str(e)}")
            state["error_message"] = f"State ID extraction error: {str(e)}"
            return state
