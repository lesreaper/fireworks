# Identification Document Extraction using Fireworks AI

This project demonstrates a few end-to-end proof-of-concepts for extracting structured information from images of U.S. identification documents (drivers licenses, state IDs, and passports) using Fireworks AI.

## To Run Apps
```
streamlit run app.py
stremalit run test_passports.py
```

## What's Inside
This is an agentic workflow for OCR processing. It sends the image ot Google for OCR, and then back to the DocType Operator to start the pipeline. You can see the approximate path below.

![worflow](workflow.png)

I attempted to fine tune a VLM, starting with Qwen2.5 VL and Donut. Unfortunately, but ran out of time for experimentation. Optimally, we can create a small 7B or 32B custom agent per state and passport card and doc, which would drastically reduce costs and increase performance.

## Quick Notes
- The goal here was to be able to show a client some of the cool things unique to the Fireworks AI, including the document inlining and structured response integration. The goal was to show the unique adds here for structured data extraction.
- I didn't batch the images, but that could have easily been done with the Fireworks API. Normally, that would reduce costs.
- YES, I used AI in a bunch of the coding work, but I understand EVERYTHING that is going on here, it's just for speed
- I would rather use URLs than Base64 (Base64 adds 30% to filze size over the wire), but that's what made it work here
- Fine tuned smaller models would handle the image space structure better probably, reduce cost, and could be more effective

## Key Features in Document Inlining

- **Document Inlining:**  
  The solution leverages Fireworks AI’s [Document Inlining](https://docs.fireworks.ai/firesearch/inline-multimodal) feature to process mulitple images by embedding image data directly (via Base64 encoding) into the API request.

- **JSON Mode Structured Responses:**  
  We use Fireworks AI’s [JSON Mode](https://docs.fireworks.ai/structured-responses/structured-response-formatting) to instruct the model to return results in a well-structured JSON format. This allows easy validation and further processing of the extracted data.

## Setup and Installation (Docker removed for now)

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
    ```
2. **Create a .env file with your key:**

    ```bash
    touch .env
    ```
    Add these values to your `.env`:
    ```
    BASE_URL="https://api.fireworks.ai/inference/v1"
    FIREWORKS_API_KEY="xxxx"
    GOOGLE_API_KEY="xxxx"
    ```

3. **Run the apps:**

    ```
    pip install -r requirements.txt
    ```
    To run the single image test:
    ```
    streamlit run app.py
    ```
    To run the multimage testing, you'll need to have images installed in `synthetic_passports`, along with an appropriate `jsonl` file. You can use the `generate_passports.py` file under the `fine_tuning` folder to select a template (hard coded), and then it should populate. From there, you simply come back, and run:
    ```
    streamlit run test_passports.py
    ```
    


