import logging
from .BaseAgent import BaseAgent
from .types import State, DocTypeResponse

logger = logging.getLogger(__name__)


class DoctypeAgent(BaseAgent):
    """Agent for detecting document type using OCR output"""

    def process(self, state: State) -> State:
        """
        Process the OCR output to determine document type

        Args:
            state: Current state containing OCR results

        Returns:
            Updated state with document type information
        """
        try:
            logger.info("Starting document type detection")
            logger.info("Available state keys: " + ", ".join(state.keys()))

            if not state.get("ocr_text"):
                logger.error("No OCR text found in state")
                logger.error(f"State contents: {state}")
                raise ValueError("No OCR text available in state")

            logger.info(f"OCR text length: {len(state['ocr_text'])}")
            logger.info(f"OCR doc text length: {len(state['ocr_doc_text'])}")
            logger.info(f"OCR confidence: {state['ocr_confidence']:.2%}")

            prompt = f"""
            Please analyze the following text extracted from an identification document and determine its type.
            The document is either a US Passport or a Driver's License/State ID.

            Raw OCR Text:
            {state["ocr_text"]}

            Document-Optimized OCR Text:
            {state["ocr_doc_text"]}

            OCR Confidence: {state["ocr_confidence"]:.2%}

            Please determine:
            1. If this is a US Passport or Driver's License/State ID
            2. If it's a Driver's License/State ID, identify which state it's from
            3. Your confidence level in this identification (0.0 to 1.0)

            Return your analysis in JSON format with:
            - document_type: either "Passport" or "Driver's License"
            - confidence: a float between 0.0 and 1.0
            - detected_state: two-letter state code if applicable
            - additional_info: any other relevant details
            """

            logger.info("Calling LLM for document type detection")
            response_data, tokens = self.call_extraction_api(
                prompt=prompt,
                image_base64=None,
                response_format={"type": "json_object", "schema": DocTypeResponse.model_json_schema()}
            )

            logger.info(f"Received response from LLM: {response_data}")

            # Parse response and update state
            doc_type_response = DocTypeResponse(**response_data)

            state["doc_type"] = doc_type_response.document_type
            state["detected_state"] = doc_type_response.detected_state
            state["doc_type_confidence"] = doc_type_response.confidence
            state["doc_type_tokens"] = tokens
            state["total_tokens"] += tokens

            logger.info(
                f"Detected document type: {state['doc_type']} "
                f"(State: {state['detected_state']}) "
                f"with confidence: {state['doc_type_confidence']:.2%}"
            )

            logger.info("Final state keys: " + ", ".join(state.keys()))
            return state

        except Exception as e:
            logger.error(f"Error in document type detection: {str(e)}")
            logger.error(f"State at error: {state}")
            state["error_message"] = f"Document type detection error: {str(e)}"
            return state
