import logging
from .types import State, IdentificationResult, ValidationResponse
from .BaseAgent import BaseAgent
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    """Agent for validating extracted information"""

    def _transform_data(self, data: dict) -> dict:
        """Transform data to match expected schema"""
        if not data:
            return data

        transformed = data.copy()

        if transformed.get("document_type") == "Passport" and "nationality" not in transformed:
            transformed["nationality"] = "US"

        logger.info(f"Transformed data: {transformed}")
        return transformed

    def process(self, state: State) -> State:
        """Validate the extracted data"""
        try:
            logger.info("Starting validation with state:")
            logger.info(f"Extracted data to validate: {state.get('extracted_data')}")

            if not state.get("extracted_data"):
                logger.error("No data to validate")
                state["validation_status"] = False
                state["validation_confidence"] = 0.0
                return state

            transformed_data = self._transform_data(state["extracted_data"])
            state["extracted_data"] = transformed_data

            try:
                IdentificationResult(**transformed_data)
                logger.info("Pydantic validation passed")
            except Exception as e:
                logger.error(f"Pydantic validation failed: {str(e)}")
                state["validation_status"] = False
                state["validation_confidence"] = 0.0
                state["validation_errors"] = {"schema_error": str(e)}
                return state

            validation_rules = ""
            if transformed_data.get("document_type") == "Passport":
                validation_rules = """
                Validation Rules for Passport:
                1. All dates must be valid and in MM/DD/YYYY format
                2. Name must be in all caps
                3. Nationality must be "US" (not "USA")
                4. Document number must follow passport format (9-10 digits)
                5. Place of birth must be in "City, Country" format
                6. Gender/Sex must be "M" or "F"
                7. Issue date must be before expiry date
                8. Date of birth must be reasonable (not in future, not too old)
                """
            else:
                validation_rules = """
                Validation Rules for Driver's License/State ID:
                1. All dates must be valid and in MM/DD/YYYY format
                2. Name must be in all caps
                3. Address must have street number, street name, city, state code, and ZIP
                4. Document number must be present and follow state format
                5. State code must be valid US state
                6. Issue date must be before expiry date
                7. Date of birth must be reasonable (not in future, not too old)
                """

            prompt = f"""
            Please validate this extracted identification document data.
            This is a {transformed_data.get('document_type', 'unknown document type')}.

            Document Data:
            {json.dumps(transformed_data, indent=2)}

            {validation_rules}

            Return a JSON response with:
            1. is_valid: boolean indicating if all data is valid
            2. confidence: your confidence in the validation (0.0 to 1.0)
            3. warnings: list of non-critical issues found
            4. errors: list of critical validation failures
            5. suggested_corrections: map of field names to suggested fixes

            Note: If confidence is 0.9 or higher, set is_valid to true even if there are minor formatting issues.
            """

            validation_result, tokens = self.call_validation_api(
                prompt=prompt,
                response_schema=ValidationResponse.model_json_schema()
            )

            transformed_result = {
                "is_valid": validation_result.get("is_valid", False),
                "confidence": validation_result.get("confidence", 0.0),
                "error_details": {}
            }

            warnings = validation_result.get("warnings", [])
            if isinstance(warnings, list):
                transformed_result["error_details"]["warnings"] = {
                    str(i): str(warning) for i, warning in enumerate(warnings)
                }
            elif isinstance(warnings, dict):
                transformed_result["error_details"]["warnings"] = {
                    str(k): str(v) for k, v in warnings.items()
                }
            elif warnings:
                transformed_result["error_details"]["warnings"] = {"0": str(warnings)}

            errors = validation_result.get("errors", [])
            if isinstance(errors, list):
                transformed_result["error_details"]["errors"] = {
                    str(i): str(error) for i, error in enumerate(errors)
                }
            elif isinstance(errors, dict):
                transformed_result["error_details"]["errors"] = {
                    str(k): str(v) for k, v in errors.items()
                }
            elif errors:
                transformed_result["error_details"]["errors"] = {"0": str(errors)}

            corrections = validation_result.get("suggested_corrections", {})
            if isinstance(corrections, dict):
                transformed_result["error_details"]["suggested_corrections"] = {}
                for field, correction in corrections.items():
                    if isinstance(correction, dict):
                        transformed_result["error_details"]["suggested_corrections"][field] = json.dumps(correction)
                    else:
                        transformed_result["error_details"]["suggested_corrections"][field] = str(correction)

            validation = ValidationResponse(**transformed_result)

            if "validation_warnings" not in state:
                state["validation_warnings"] = []

            # Handle high confidence auto-pass
            if validation.confidence >= 0.9:
                logger.info(f"Auto-passing validation due to high confidence: {validation.confidence}")
                validation.is_valid = True

                if validation.error_details:
                    warnings_dict = validation.error_details.get("warnings", {})
                    for warning in warnings_dict.values():
                        state["validation_warnings"].append(str(warning))

                    validation.error_details = None
                    logger.info(f"Converted errors to warnings: {state['validation_warnings']}")
            state["validation_status"] = validation.is_valid
            state["validation_confidence"] = validation.confidence
            state["validation_errors"] = validation.error_details if validation.error_details else {}
            state["suggested_corrections"] = validation.suggested_corrections if validation.suggested_corrections else {}
            state["validation_tokens"] = tokens
            state["total_tokens"] = state.get("total_tokens", 0) + tokens

            logger.info(f"Validation complete - Status: {state['validation_status']}, Confidence: {state['validation_confidence']:.2%}")
            if state.get("validation_warnings"):
                logger.info(f"Warnings: {state['validation_warnings']}")
            if state.get("validation_errors"):
                logger.info(f"Errors: {state['validation_errors']}")

            return state

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            state["validation_status"] = False
            state["validation_confidence"] = 0.0
            state["validation_errors"] = {"validation_error": str(e)}
            return state
