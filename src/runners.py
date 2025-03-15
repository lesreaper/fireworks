import os
from dotenv import load_dotenv
import openai
from src.api import APIInlineClient
from src.prompts import PromptGenerator
from src.utils import encode_image, FileSaver, preprocess_and_encode_image
from eval.eval import evaluate_run

load_dotenv()

client = openai.OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY")
)

image_paths = [
    "./images/License 1.png",
    "./images/License-2.jpg",
    "./images/License-3.jpeg",
    "./images/passport-1.jpeg",
    "./images/passport-2.jpg",
]


def run_second_base():
    print("Started processing using Document Inlining and JSON mode \n")
    prompt = PromptGenerator.generate_prompt()
    api_client = APIInlineClient(client)

    run_tokens = 0

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        image_base64 = encode_image(image_path)
        result, total_tokens = api_client.call_api(prompt, image_base64)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = FileSaver.save_output(result, f"{base_filename}.json")

        run_tokens += total_tokens

        print(f"Results have been saved to {output_filename}")
        print(f"Total tokens used: {total_tokens}")
        print(f"Approximate cost: ${total_tokens * 0.0001} \n")
        print("---------------------------------------------------\n")

    evaluate_run()
    print("---------------------------------------------------\n")
    print("All done using Document Inlining and JSON mode \n")
    print("Total tokens used for all images:", run_tokens)
    print(f"Approximate cost: ${run_tokens * 0.0001} \n")
    print("---------------------------------------------------")


def run_first_base():
    print("Started processing using Document Inlining, JSON mode, and OpenCV pre-processing. \n")
    prompt = PromptGenerator.generate_prompt()
    api_client = APIInlineClient(client)

    run_tokens = 0

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        try:
            image_base64 = preprocess_and_encode_image(image_path)
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            continue

        # Call the API.
        result, total_tokens = api_client.call_api(prompt, image_base64)
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = FileSaver.save_output(result, f"{base_filename}.json")

        run_tokens += total_tokens

        print(f"Results have been saved to {output_filename}")
        print(f"Total tokens used: {total_tokens}")
        print(f"Approximate cost: ${total_tokens * 0.0001} \n")

    evaluate_run()
    print("---------------------------------------------------\n")
    print("All done using Document Inlining, JSON mode, and OpenCV pre-processing. \n")
    print("Total tokens used for all images:", run_tokens)
    print(f"Approximate cost: ${run_tokens * 0.0001} \n")
    print("---------------------------------------------------")
