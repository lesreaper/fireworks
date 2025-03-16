import logging
from .types import State, IdentificationResult, ValidationResponse
from .BaseAgent import BaseAgent
from pydantic import ValidationError
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    def _transform_data(self, data: dict) -> dict:
        """Transform extracted data to match IdentificationResult schema"""
        try:
            # Handle different name formats
            if "full_name" in data:
                name_data = data["full_name"]
            elif "name" in data:
                name = data["name"]
                if isinstance(name, dict):
                    if "first_name" in name:
                        name_data = name
                    else:
                        # Convert from first/middle/last to first_name/middle_name/last_name
                        name_data = {
                            "first_name": name.get("first", ""),
                            "middle_name": name.get("middle", ""),
                            "last_name": name.get("last", "")
                        }
            else:
                name_data = {
                    "first_name": data.get("first_name", ""),
                    "middle_name": data.get("middle_name", ""),
                    "last_name": data.get("last_name", "")
                }

            # Handle different address formats
            if "complete_address" in data:
                address_data = data["complete_address"]
            elif "address" in data:
                address_data = data["address"]
            else:
                address_data = {
                    "street": data.get("street", ""),
                    "city": data.get("city", ""),
                    "state": data.get("state", ""),
                    "zip_code": data.get("zip_code", "")
                }

            # Build transformed data
            transformed = {
                "document_type": data.get("document_type", ""),
                "document_number": data.get("document_number", ""),
                "name": name_data,
                "date_of_birth": data.get("date_of_birth", ""),
                "issue_date": data.get("issue_date", ""),
                "expiry_date": data.get("expiry_date", ""),
                "address": address_data,
                "nationality": data.get("nationality", "USA"),  # Default to USA for US documents
                "state": data.get("state_of_issuance") or data.get("state", "")
            }

            logger.info(f"Transformed data: {transformed}")
            return transformed
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise

    def process(self, state: State) -> State:
        """Validate extracted data with enhanced logging"""
        try:
            logger.info("Starting validation with state:")
            logger.info(f"Extracted data to validate: {state.get('extracted_data')}")

            # First, validate with Pydantic
            if state["extracted_data"]:
                try:
                    # Transform data before validation
                    transformed_data = self._transform_data(state["extracted_data"])
                    validation_result = IdentificationResult(**transformed_data)
                    state["extracted_data"] = validation_result.model_dump()  # Update with validated data
                    basic_validation_passed = True
                    logger.info("Pydantic validation passed")
                except ValidationError as e:
                    basic_validation_passed = False
                    logger.error(f"Pydantic validation failed: {str(e)}")
                    state["error_message"] = f"Basic validation error: {str(e)}"
                    return state

                # If basic validation passes, use LLM for complex validation
                if basic_validation_passed:
                    prompt = f"""
                    Please validate the following extracted identification document data for validity and consistency:

                    {json.dumps(state["extracted_data"], indent=2)}

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

                    response = self.call_validation_api(
                        prompt=prompt,
                        response_schema=ValidationResponse.model_json_schema()
                    )
                    validation_response = ValidationResponse(**json.loads(response.choices[0].message.content))
                    tokens = response.usage.total_tokens
                    state["total_tokens"] += tokens

                    logger.info("LLM Validation results:")
                    logger.info(f"Validation confidence: {validation_response.confidence}")
                    logger.info(f"Is valid: {validation_response.is_valid}")
                    if validation_response.error_details:
                        logger.info(f"Error details: {validation_response.error_details}")
                    if validation_response.suggested_corrections:
                        logger.info(f"Suggested corrections: {validation_response.suggested_corrections}")

                    # Set validation status based on 0.5 confidence threshold
                    if validation_response.is_valid and validation_response.confidence >= 0.5:
                        state["validation_status"] = True
                        state["validation_confidence"] = validation_response.confidence
                        logger.info(f"Validation passed with confidence: {validation_response.confidence}")
                    else:
                        state["validation_status"] = False
                        state["error_message"] = "LLM validation failed or confidence too low"
                        state["validation_errors"] = validation_response.error_details
                        state["suggested_corrections"] = validation_response.suggested_corrections
                        logger.info(f"Validation failed. Confidence: {validation_response.confidence}")

                    # Log very low confidence validations
                    if validation_response.confidence < 0.5:
                        logger.warning(f"Very low confidence in validation: {validation_response.confidence}")
            else:
                logger.error("No data to validate")
                state["validation_status"] = False
                state["error_message"] = "No data to validate"

            # Log final validation state
            logger.info("Final validation state:")
            logger.info(f"Validation status: {state['validation_status']}")
            logger.info(f"Validation confidence: {state.get('validation_confidence')}")
            logger.info(f"Validation errors: {state.get('validation_errors')}")
            logger.info(f"Error message: {state.get('error_message')}")

            return state
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            state["validation_status"] = False
            state["error_message"] = f"Validation error: {str(e)}"
            return state
