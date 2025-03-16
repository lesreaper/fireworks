from typing import Dict, Any
from langgraph.graph import StateGraph, Graph
import logging
from dotenv import load_dotenv
import os
import openai
# import graphviz
from agents.types import State
from agents.utils import preprocess_and_encode_image
from agents.Doctype import DoctypeAgent
from agents.HumanEval import HumanEvalAgent
from agents.Passport import PassportAgent
from agents.StateId import StateIdAgent
from agents.SysEval import SysEvalAgent
from agents.Validation import ValidationAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def build_pipeline(api_client, save_visualization: bool = False, visualization_path: str = "workflow.png") -> Graph:
    """Build the document processing pipeline with optional visualization

    Args:
        api_client: The Fireworks API client to use for LLM calls
        save_visualization: Whether to save the workflow visualization
        visualization_path: Path where to save the visualization
    """
    workflow = StateGraph(State)

    # Initialize agents with API client
    doctype_agent = DoctypeAgent(api_client)
    passport_agent = PassportAgent(api_client)
    state_id_agent = StateIdAgent(api_client)
    validation_agent = ValidationAgent(api_client)
    human_eval_agent = HumanEvalAgent(api_client)
    sys_eval_agent = SysEvalAgent(api_client)

    workflow.add_node("preprocess", preprocess)
    workflow.add_node("DoctypeAgent", doctype_agent.process)
    workflow.add_node("PassportAgent", passport_agent.process)
    workflow.add_node("StateIdAgent", state_id_agent.process)
    workflow.add_node("validate", validation_agent.process)
    workflow.add_node("HumanEvalAgent", human_eval_agent.process)
    workflow.add_node("evaluate", sys_eval_agent.process)

    def route_to_extractor(state: State) -> str:
        route = "PassportAgent" if state["doc_type"] == "Passport" else "StateIdAgent"
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
                return "HumanEvalAgent"
            state["previous_attempt_score"] = state.get("validation_confidence", 0)
            return route_to_extractor(state)
        else:
            logger.info("Max attempts reached, routing to human evaluation")
            return "HumanEvalAgent"

    workflow.add_edge("preprocess", "DoctypeAgent")
    workflow.add_conditional_edges("DoctypeAgent", route_to_extractor)
    workflow.add_edge("PassportAgent", "validate")
    workflow.add_edge("StateIdAgent", "validate")
    workflow.add_conditional_edges("validate", handle_validation)
    workflow.add_edge("HumanEvalAgent", "evaluate")

    workflow.set_entry_point("preprocess")

    if save_visualization:
        try:
            import graphviz
            dot = graphviz.Digraph(comment='Document Processing Pipeline')
            dot.attr(rankdir='LR')

            for node_name in workflow.nodes:
                dot.node(node_name, node_name)

            dot.edge("DoctypeAgent", "PassportAgent", "if Passport")
            dot.edge("DoctypeAgent", "StateIdAgent", "if Driver's License")
            dot.edge("PassportAgent", "validate")
            dot.edge("StateIdAgent", "validate")
            dot.edge("validate", "evaluate", "if valid")
            dot.edge("validate", "PassportAgent", "if retry needed (Passport)")
            dot.edge("validate", "StateIdAgent", "if retry needed (License)")
            dot.edge("validate", "HumanEvalAgent", "if max retries")
            dot.edge("HumanEvalAgent", "evaluate")

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
        "validation_tokens": 0
    }

    try:
        pipeline = build_pipeline(client, save_visualization, visualization_path)
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
    image_path = "./images/License-1.png"
    result = process_document(image_path)
