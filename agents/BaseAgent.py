from typing import Optional, Dict, Any, Tuple
import logging
import json
from .types import State
from fireworks.client import Fireworks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, api_client: Optional[Fireworks] = None):
        self.api_client = api_client

    def call_extraction_api(self, prompt: str, image_base64: str = None, response_format: dict = None) -> Tuple[Dict[str, Any], int]:
        """Call the extraction API with optional image and response format"""
        if not self.api_client:
            raise ValueError("API client not initialized")

        # Construct message content based on whether we have an image
        message_content = [{"type": "text", "text": prompt}]
        if image_base64:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        completion_args = {
            "model": "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
            "messages": [{"role": "user", "content": message_content}],
            "temperature": 0.1  # Lower temperature for more consistent outputs
        }

        if response_format:
            completion_args["response_format"] = response_format

        try:
            response = self.api_client.chat.completions.create(**completion_args)
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                parsed_content = json.loads(content)
                logger.debug(f"Parsed API response: {parsed_content}")
                return parsed_content, response.usage.total_tokens
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response as JSON: {content}")
                raise ValueError(f"API returned invalid JSON: {str(e)}")
                
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def call_validation_api(self, prompt: str, response_schema: dict = None) -> Tuple[Dict[str, Any], int]:
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

        try:
            response = self.api_client.chat.completions.create(**completion_args)
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                parsed_content = json.loads(content)
                logger.debug(f"Parsed validation response: {parsed_content}")
                return parsed_content, response.usage.total_tokens
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse validation API response as JSON: {content}")
                raise ValueError(f"Validation API returned invalid JSON: {str(e)}")
                
        except Exception as e:
            logger.error(f"Validation API call failed: {str(e)}")
            raise

    def process(self, state: State) -> State:
        """Process the current state and return updated state"""
        raise NotImplementedError("Subclasses must implement process method")
