
from fireworks.client import Fireworks
from typing import Tuple
import json
from pydantic import ValidationError

from .models_agents import IdentificationResult, DocTypeResponse, ValidationResponse

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIAgentsClient:
    def __init__(self, client: Fireworks):
        self.client = client

    def call_extraction_api(self, prompt: str, image_base64: str) -> Tuple[dict, int]:
        """API call for document field extraction"""
        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}#transform=inline"
                }
            },
        ]
        messages = [{"role": "user", "content": message_content}]

        response = self.client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": IdentificationResult.model_json_schema()
            }
        )

        raw_output = response.choices[0].message.content

        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed_output = {"raw_output": raw_output}
        return parsed_output, response.usage.total_tokens

    def call_doctype_api(self, image_base64: str) -> Tuple[DocTypeResponse, int]:
        """API call for document type identification"""
        prompt = """
        Please analyze this image and identify the type of identification document.

        Determine:
        1. If this is a US Passport or Driver's License/State ID
        2. If it's a Driver's License/State ID, identify which state it's from
        3. Your confidence level in this identification (0.0 to 1.0)
        4. Any additional relevant details about the document type

        Return your analysis in a structured JSON format with the following fields:
        - document_type: either "Passport" or "Driver's License"
        - confidence: a float between 0.0 and 1.0
        - detected_state: two-letter state code if applicable
        - additional_info: any other relevant details
        """

        message_content = [
            {"type": "text", "text": prompt},
            {"top_p": 1},
            {"top_k": 100},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}#transform=inline"
                }
            },
        ]
        messages = [{"role": "user", "content": message_content}]

        response = self.client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": DocTypeResponse.model_json_schema()
            }
        )

        raw_output = response.choices[0].message.content
        try:
            parsed_output = json.loads(raw_output)
            return DocTypeResponse(**parsed_output), response.usage.total_tokens
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Error parsing document type response: {str(e)}")
            raise

    def call_validation_api(self, extracted_data: dict) -> Tuple[ValidationResponse, int]:
        """API call for complex validation cases"""
        prompt = f"""
        Please analyze this extracted identification document data for validity and consistency.

        Document Data:
        {json.dumps(extracted_data, indent=2)}

        Please verify:
        1. All dates are valid and logically consistent (issue date before expiry, etc.)
        2. Name formatting is consistent and reasonable
        3. Address format is valid (if present)
        4. Document numbers follow expected patterns
        5. All required fields are present and properly formatted

        Return your analysis in a structured JSON format with:
        - is_valid: boolean indicating if the data appears valid
        - confidence: your confidence in this assessment (0.0 to 1.0)
        - error_details: map of any found errors
        - suggested_corrections: map of suggested fixes for any issues
        """

        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": ValidationResponse.model_json_schema()
            }
        )

        raw_output = response.choices[0].message.content
        try:
            parsed_output = json.loads(raw_output)
            return ValidationResponse(**parsed_output), response.usage.total_tokens
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Error parsing validation response: {str(e)}")
            raise
