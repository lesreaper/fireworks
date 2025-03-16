import logging
from .types import State
from .BaseAgent import BaseAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HumanEvalAgent(BaseAgent):
    def process(self, state: State) -> State:
        """Handle human evaluation"""
        logger.info(f"Human evaluation required for document. Error: {state['error_message']}")
        # In practice, integrate with your human review system
        return state
