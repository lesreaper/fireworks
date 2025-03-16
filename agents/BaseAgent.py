from typing import Optional
import logging
from .types import State
from fireworks.client import Fireworks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, api_client: Optional[Fireworks] = None):
        self.api_client = api_client

    def call_extraction_api(self, prompt: str, image_base64: str, response_schema: dict = None):
        """Base method for extraction API calls"""
        if not self.api_client:
            raise ValueError("API client not initialized")

        completion_args = {
            "model": "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }],
            "temperature": 0.1,  # Lower temperature for more consistent outputs
            "response_format": {"type": "json_object"}
        }

        if response_schema:
            completion_args["response_format"]["schema"] = response_schema

        return self.api_client.chat.completions.create(**completion_args)

    def call_validation_api(self, prompt: str, response_schema: dict = None):
        """Base method for validation API calls"""
        if not self.api_client:
            raise ValueError("API client not initialized")

        completion_args = {
            "model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Lower temperature for more consistent outputs
            "response_format": {"type": "json_object"}
        }

        if response_schema:
            completion_args["response_format"]["schema"] = response_schema

        return self.api_client.chat.completions.create(**completion_args)

    def process(self, state: State) -> State:
        """Base process method to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement process()")
