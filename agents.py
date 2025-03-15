from typing import Optional, TypedDict, Dict, Any
from pydantic import ValidationError
from langgraph.graph import StateGraph, Graph
import logging
from dotenv import load_dotenv
import json
import os
import openai
# import graphviz
from src.api_agents import APIAgentsClient
from src.models_agents import IdentificationResult, EvaluationMetrics
from src.utils import preprocess_and_encode_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class State(TypedDict):
    image_data: str
    image_path: str
    doc_type: Optional[str]
    detected_state: Optional[str]
    doc_type_confidence: Optional[float]
    extraction_attempts: int
    extracted_data: Optional[Dict]
    validation_status: bool
    validation_confidence: Optional[float]
    validation_errors: Optional[Dict[str, str]]
    suggested_corrections: Optional[Dict[str, Any]]
    error_message: Optional[str]
    total_tokens: int
    api_client: APIAgentsClient


def preprocess(state: State) -> State:
    """Preprocess the document image"""
    try:
        processed_image = preprocess_and_encode_image(state["image_data"])
        state["image_data"] = processed_image
        return state
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        state["error_message"] = f"Preprocessing error: {str(e)}"
        return state


def identify_doc_type(state: State) -> State:
    """Identify the type of document using LLM"""
    try:
        doc_type_response, tokens = state["api_client"].call_doctype_api(
            image_base64=state["image_data"]
        )

        state["doc_type"] = doc_type_response.document_type
        if doc_type_response.detected_state:
            state["detected_state"] = doc_type_response.detected_state
        state["doc_type_confidence"] = doc_type_response.confidence
        state["total_tokens"] += tokens

        if doc_type_response.confidence < 0.7:
            logger.warning(f"Low confidence in document type detection: {doc_type_response.confidence}")

        return state
    except Exception as e:
        logger.error(f"Document type detection failed: {str(e)}")
        state["error_message"] = f"Document type detection error: {str(e)}"
        return state


def extract_passport(state: State) -> State:
    """Extract information from passport"""
    try:
        prompt = """
        Please analyze this US Passport image and extract the following information in JSON format:
        - Document number
        - Full name (first, middle if present, and last name)
        - Date of birth
        - Issue date
        - Expiry date
        - Nationality (should be "USA" for US passports)

        Please ensure all dates are in YYYY-MM-DD format and the document type is set to "Passport".
        """

        extracted_data, tokens = state["api_client"].call_api(
            prompt=prompt,
            image_base64=state["image_data"]
        )

        state["extracted_data"] = extracted_data
        state["extraction_attempts"] += 1
        state["total_tokens"] += tokens
        return state
    except Exception as e:
        logger.error(f"Passport extraction failed: {str(e)}")
        state["error_message"] = f"Passport extraction error: {str(e)}"
        return state


def extract_state_id(state: State) -> State:
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
        - Document type (should be "Driver's License")
        - Document number
        - Full name (first, middle if present, and last name)
        - Date of birth
        - Issue date
        - Expiry date
        - Complete address (street, city, state, zip code)
        - State of issuance

        Please ensure:
        - All dates are in YYYY-MM-DD format
        - The state code is a two-letter abbreviation
        - The zip code is a 5-digit string
        """

        # Make API call
        extracted_data, tokens = state["api_client"].call_extraction_api(
            prompt=prompt,
            image_base64=state["image_data"]
        )

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


def validate_data(state: State) -> State:
    """Validate extracted data with enhanced logging"""
    try:
        logger.info("Starting validation with state:")
        logger.info(f"Extracted data to validate: {state.get('extracted_data')}")

        # First, validate with Pydantic
        if state["extracted_data"]:
            try:
                IdentificationResult(**state["extracted_data"])
                basic_validation_passed = True
                logger.info("Pydantic validation passed")
            except ValidationError as e:
                basic_validation_passed = False
                logger.error(f"Pydantic validation failed: {str(e)}")
                state["error_message"] = f"Basic validation error: {str(e)}"
                return state

            # If basic validation passes, use LLM for complex validation
            if basic_validation_passed:
                validation_response, tokens = state["api_client"].call_validation_api(
                    state["extracted_data"]
                )
                state["total_tokens"] += tokens

                logger.info("LLM Validation results:")
                logger.info(f"Validation confidence: {validation_response.confidence}")
                logger.info(f"Is valid: {validation_response.is_valid}")
                if hasattr(validation_response, 'error_details'):
                    logger.info(f"Error details: {validation_response.error_details}")
                if hasattr(validation_response, 'suggested_corrections'):
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


def human_evaluation(state: State) -> State:
    """Handle human evaluation"""
    logger.info(f"Human evaluation required for document. Error: {state['error_message']}")
    # In practice, integrate with your human review system
    return state


def evaluate_results(state: Dict[str, Any]) -> EvaluationMetrics:
    """Calculate detailed evaluation metrics by comparing with ground truth"""
    try:
        # Load ground truth data from file
        try:
            with open('./eval/actual_results.json', 'r') as f:
                ground_truth_data = json.loads(f.read())
            logger.info(f"Successfully loaded ground truth data with {len(ground_truth_data)} entries")
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {str(e)}")
            raise

        # Get the current file being processed
        image_path = state.get("image_path", "")
        file_name = os.path.basename(image_path)
        logger.info(f"Evaluating file: {file_name}")

        # Find matching ground truth
        ground_truth = None
        for entry in ground_truth_data:
            if entry["file"].replace(" ", "-") == file_name:  # Handle space/dash differences
                ground_truth = entry["data"]
                break

        if not ground_truth:
            logger.error(f"No ground truth found for file: {file_name}")
            return EvaluationMetrics(
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                error_rate=1.0,
                per_field_stats={},
                processing_stats={},
                token_usage={},
                document_type="Unknown",
                validation_status=False,
                confidence_scores={},
                error_messages=["No ground truth data found"]
            )

        extracted_data = state.get("extracted_data", {}) or {}

        normalized_extracted = {
            "document_type": extracted_data.get("document_type"),
            "document_number": extracted_data.get("document_number"),
            "first_name": extracted_data.get("name", {}).get("first_name"),
            "last_name": extracted_data.get("name", {}).get("last_name"),
            "state": extracted_data.get("state"),
            "date_of_birth": extracted_data.get("date_of_birth"),
            "issue_date": extracted_data.get("issue_date"),
            "expiration_date": extracted_data.get("expiry_date"),  # Note the field name difference
            "address": extracted_data.get("address", {}).get("street", "") + ", " + extracted_data.get("address", {}).get("city", "") + ", " + extracted_data.get("address", {}).get("state", "") + " " + extracted_data.get("address", {}).get("zip_code", "")
        }

        # Calculate per-field metrics
        field_stats = {}
        total_correct = 0
        total_predicted = 0
        total_actual = 0

        for field in ground_truth.keys():
            gt_value = str(ground_truth.get(field, "")).lower().strip()
            extracted_value = str(normalized_extracted.get(field, "")).lower().strip()

            has_gt = bool(gt_value)
            has_prediction = bool(extracted_value)
            is_correct = gt_value == extracted_value

            if has_gt:
                total_actual += 1
            if has_prediction:
                total_predicted += 1
            if is_correct:
                total_correct += 1

            precision = 1.0 if (has_prediction and is_correct) else 0.0
            recall = 1.0 if (has_gt and is_correct) else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            field_stats[field] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            logger.info(f"Field {field}:")
            logger.info(f"  Ground truth: {gt_value}")
            logger.info(f"  Extracted: {extracted_value}")
            logger.info(f"  Metrics: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        recall = total_correct / total_actual if total_actual > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = total_correct / total_actual if total_actual > 0 else 0.0
        error_rate = 1.0 - accuracy

        # Get process statistics
        processing_stats = {
            "extraction_attempts": state.get("extraction_attempts", 0),
            "total_tokens": state.get("total_tokens", 0),
            "document_type": state.get("doc_type", "Unknown"),
            "validation_status": state.get("validation_status", False)
        }

        # Token usage
        token_usage = {
            "total": state.get("total_tokens", 0),
            "doc_type_detection": state.get("doc_type_tokens", 0),
            "extraction": state.get("extraction_tokens", 0),
            "validation": state.get("validation_tokens", 0)
        }

        # Confidence scores
        confidence_scores = {
            "document_type": state.get("doc_type_confidence", 0.0),
            "validation": state.get("validation_confidence", 0.0)
        }

        # Calculate overall image metrics
        total_fields_correct = sum(1 for field, stats in field_stats.items() if stats['precision'] == 1.0 and stats['recall'] == 1.0)
        total_fields_predicted = sum(1 for field, stats in field_stats.items() if stats['precision'] > 0)
        total_fields_actual = len(ground_truth.keys())

        # Calculate overall image P/R/F1
        image_precision = total_fields_correct / total_fields_predicted if total_fields_predicted > 0 else 0.0
        image_recall = total_fields_correct / total_fields_actual if total_fields_actual > 0 else 0.0
        image_f1 = (2 * image_precision * image_recall) / (image_precision + image_recall) if (image_precision + image_recall) > 0 else 0.0

        # Print summary
        print("\n=== OVERALL IMAGE METRICS ===")
        print(f"Image Precision: {image_precision:.3f}")
        print(f"Image Recall: {image_recall:.3f}")
        print(f"Image F1 Score: {image_f1:.3f}")
        print(f"Correct Fields: {total_fields_correct}")
        print(f"Total Predicted Fields: {total_fields_predicted}")
        print(f"Total Actual Fields: {total_fields_actual}")
        print("===========================\n")

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            error_rate=error_rate,
            per_field_stats=field_stats,
            processing_stats=processing_stats,
            token_usage=token_usage,
            document_type=state.get("doc_type", "Unknown"),
            validation_status=state.get("validation_status", False),
            confidence_scores=confidence_scores,
            error_messages=[state.get("error_message")] if state.get("error_message") else None
        )

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


def build_pipeline(save_visualization: bool = False, visualization_path: str = "workflow.png") -> Graph:
    """Build the document processing pipeline with optional visualization

    Args:
        save_visualization: Whether to save the workflow visualization
        visualization_path: Path where to save the visualization
    """
    workflow = StateGraph(State)

    workflow.add_node("preprocess", preprocess)
    workflow.add_node("identify_doc_type", identify_doc_type)
    workflow.add_node("extract_passport", extract_passport)
    workflow.add_node("extract_state_id", extract_state_id)
    workflow.add_node("validate", validate_data)
    workflow.add_node("human_evaluation", human_evaluation)
    workflow.add_node("evaluate", evaluate_results)

    def route_to_extractor(state: State) -> str:
        route = "extract_passport" if state["doc_type"] == "Passport" else "extract_state_id"
        logger.info(f"Routing to: {route}")
        return route

    def handle_validation(state: State) -> str:
        if state["validation_status"]:
            logger.info("Validation successful, routing to evaluation")
            return "evaluate"
        elif state["extraction_attempts"] < 3:
            logger.info(f"Validation failed, attempt {state['extraction_attempts']}, retrying extraction")
            if state.get("previous_attempt_score", 0) >= state.get("validation_confidence", 0):
                logger.warning("No improvement in validation score, moving to human evaluation")
                return "human_evaluation"
            state["previous_attempt_score"] = state.get("validation_confidence", 0)
            return route_to_extractor(state)
        else:
            logger.info("Max attempts reached, routing to human evaluation")
            return "human_evaluation"

    workflow.add_edge("preprocess", "identify_doc_type")
    workflow.add_conditional_edges("identify_doc_type", route_to_extractor)
    workflow.add_edge("extract_passport", "validate")
    workflow.add_edge("extract_state_id", "validate")
    workflow.add_conditional_edges("validate", handle_validation)
    workflow.add_edge("human_evaluation", "evaluate")

    workflow.set_entry_point("preprocess")

    if save_visualization:
        try:
            import graphviz
            dot = graphviz.Digraph(comment='Document Processing Pipeline')
            dot.attr(rankdir='LR')

            for node_name in workflow.nodes:
                dot.node(node_name, node_name)

            dot.edge("identify_doc_type", "extract_passport", "if Passport")
            dot.edge("identify_doc_type", "extract_state_id", "if Driver's License")
            dot.edge("extract_passport", "validate")
            dot.edge("extract_state_id", "validate")
            dot.edge("validate", "evaluate", "if valid")
            dot.edge("validate", "extract_passport", "if retry needed (Passport)")
            dot.edge("validate", "extract_state_id", "if retry needed (License)")
            dot.edge("validate", "human_evaluation", "if max retries")
            dot.edge("human_evaluation", "evaluate")

            dot.render(visualization_path, format='png', cleanup=True)
            logger.info(f"Workflow visualization saved to {visualization_path}")
        except Exception as e:
            logger.warning(f"Could not save workflow visualization: {str(e)}")

    app = workflow.compile()

    return app


def process_document(
    image_path: str,
    save_visualization: bool = False,
    visualization_path: str = "workflow.png"
) -> Dict[str, Any]:
    """
    Process a single document image through the pipeline

    Args:
        image_path: Path to the document image
        save_visualization: Whether to save the workflow visualization
        visualization_path: Path where to save the visualization
    """
    load_dotenv()
    api_key = os.getenv("FIREWORKS_API_KEY")

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key
    )
    api_client = APIAgentsClient(client)

    # Initialize state
    initial_state = {
        "image_data": image_path,
        "image_path": image_path,
        "doc_type": None,
        "detected_state": None,
        "doc_type_confidence": None,
        "extraction_attempts": 0,
        "extracted_data": None,
        "validation_status": False,
        "validation_confidence": 0.0,
        "previous_attempt_score": 0.0,
        "validation_errors": None,
        "suggested_corrections": None,
        "error_message": None,
        "total_tokens": 0,
        "doc_type_tokens": 0,
        "extraction_tokens": 0,
        "validation_tokens": 0,
        "api_client": api_client
    }

    try:
        pipeline = build_pipeline(save_visualization, visualization_path)
        result = pipeline.invoke(initial_state)

        metrics = result.get("evaluate")

        return {
            "status": "success",
            "metrics": metrics,
            "visualization_path": visualization_path if save_visualization else None,
            "final_state": result
        }

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "visualization_path": visualization_path if save_visualization else None
        }


if __name__ == "__main__":
    image_path = "./images/License-2.jpg"
    result = process_document(image_path)
