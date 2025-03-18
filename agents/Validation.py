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

        # Create a copy to avoid modifying original
        transformed = data.copy()

        # Only add nationality for passports
        if transformed.get("document_type") == "Passport" and "nationality" not in transformed:
            transformed["nationality"] = "US"

        # Log the transformation
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

            # Transform data if needed
            transformed_data = self._transform_data(state["extracted_data"])

            # First, validate with Pydantic
            try:
                IdentificationResult(**transformed_data)
                logger.info("Pydantic validation passed")
            except Exception as e:
                logger.error(f"Pydantic validation failed: {str(e)}")
                state["validation_status"] = False
                state["validation_confidence"] = 0.0
                state["validation_errors"] = {"schema_error": str(e)}
                return state

            # Construct prompt for LLM validation
            prompt = f"""
            Please validate this extracted identification document data for accuracy and consistency.
            This is a {transformed_data.get('document_type', 'unknown document type')}.

            Document Data:
            {json.dumps(transformed_data, indent=2)}

            Validation Rules:
            1. For Driver's License:
               - All dates should be valid and in MM/DD/YYYY format
               - Name should be in all caps
               - Address should have street number, street name, city, state code, and ZIP
               - Document number should be present
               - State code should be valid US state

            2. For Passport:
               - All dates should be valid and in MM/DD/YYYY format
               - Name should be in all caps
               - Nationality should be present (defaults to "US")
               - Document number should follow passport format
               - No address is required

            Return a JSON response with:
            - is_valid: boolean indicating if all data is valid
            - confidence: your confidence in the validation (0.0 to 1.0)
            - error_details: map of field names to error messages (if any)
            - suggested_corrections: map of field names to suggested fixes

            Note: If your confidence is 0.9 or higher, you should set is_valid to true
            even if there are minor formatting issues.
            """

            # Call LLM for validation
            validation_result, tokens = self.call_validation_api(
                prompt=prompt,
                response_schema=ValidationResponse.model_json_schema()
            )

            # Parse validation response
            validation = ValidationResponse(**validation_result)

            # Auto-pass if confidence is very high
            if validation.confidence >= 0.9:
                logger.info(f"Auto-passing validation due to high confidence: {validation.confidence}")
                validation.is_valid = True
                # Convert any errors to warnings if we're auto-passing
                if validation.error_details:
                    state["validation_warnings"] = validation.error_details
                    validation.error_details = None  # Clear errors since we're auto-passing
                    logger.info(f"Converted errors to warnings: {state['validation_warnings']}")

            # Update state with validation results
            state["validation_status"] = validation.is_valid
            state["validation_confidence"] = validation.confidence
            
            # Make sure warnings are preserved in the state
            if validation.error_details:
                if validation.confidence >= 0.9:
                    state["validation_warnings"] = validation.error_details
                else:
                    state["validation_errors"] = validation.error_details
            
            if validation.suggested_corrections:
                state["suggested_corrections"] = validation.suggested_corrections
                
            state["validation_tokens"] = tokens
            state["total_tokens"] += tokens

            # Log final state for debugging
            logger.info("Final validation state:")
            logger.info(f"Status: {state['validation_status']}")
            logger.info(f"Confidence: {state['validation_confidence']}")
            if state.get("validation_warnings"):
                logger.info(f"Warnings: {state['validation_warnings']}")
            elif state.get("validation_errors"):
                logger.info(f"Errors: {state['validation_errors']}")

            # Make sure warnings are preserved in the final state
            if "validation_warnings" not in state:
                state["validation_warnings"] = {}

            return state

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            state["validation_status"] = False
            state["validation_confidence"] = 0.0
            state["validation_errors"] = {"validation_error": str(e)}
            return state
