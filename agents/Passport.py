import logging
from .types import State, IdentificationResult
from .BaseAgent import BaseAgent
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PassportAgent(BaseAgent):
    def process(self, state: State) -> State:
        """Extract information from passport"""
        try:
            prompt = """
            Please analyze this US Passport image and extract the following information in JSON format:
            {
                "document_type": "Passport",
                "document_number": "string",
                "name": {
                    "first_name": "string",
                    "middle_name": "string (optional)",
                    "last_name": "string"
                },
                "date_of_birth": "YYYY-MM-DD",
                "issue_date": "YYYY-MM-DD",
                "expiry_date": "YYYY-MM-DD",
                "nationality": "USA",
                "state": "USA"
            }

            Please ensure:
            - All dates are in YYYY-MM-DD format
            - The nationality should be "USA" for US passports
            - The state field should be "USA" for passports
            - Include all visible fields, marking any unclear or missing fields as empty strings
            """

            response = self.call_extraction_api(
                prompt=prompt,
                image_base64=state["image_data"],
                response_schema=IdentificationResult.model_json_schema()
            )

            extracted_data = json.loads(response.choices[0].message.content)

            # Add empty address for schema compatibility
            extracted_data["address"] = {
                "street": "",
                "city": "",
                "state": "USA",
                "zip_code": ""
            }

            state["extracted_data"] = extracted_data
            state["extraction_attempts"] += 1
            state["total_tokens"] += response.usage.total_tokens
            return state
        except Exception as e:
            logger.error(f"Passport extraction failed: {str(e)}")
            state["error_message"] = f"Passport extraction error: {str(e)}"
            return state
