from fireworks.client import Fireworks
from .models import IdentificationResult
import json


class APIInlineClient:
    def __init__(self, client: Fireworks):
        self.client = client

    def call_api(self, prompt: str, image_base64: str) -> dict:
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
