import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import json
import os
import sys
import time
from PIL import Image, ImageTk
import cv2
import numpy as np

# For traditional OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract not available. Traditional OCR will be disabled.")

# For transformer models
try:
    import torch
    from transformers import AutoProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Deep learning OCR will be disabled.")

class PassportOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Passport OCR Demo")
        self.root.geometry("900x700")
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize variables
        self.image_path = None
        self.current_image = None
        self.models = {}
        
        # Add methods
        self.available_methods = []
        
        if TESSERACT_AVAILABLE:
            self.available_methods.append("Traditional OCR (Tesseract)")
        
        if TRANSFORMERS_AVAILABLE:
            self.available_methods.extend([
                "Donut (Fine-tuned)",
                "Donut (Base)",
                "Donut (Receipt Model)"
            ])
        
        if not self.available_methods:
            self.available_methods.append("No OCR methods available")
        
        self.method_combobox['values'] = self.available_methods
        if self.available_methods:
            self.method_combobox.current(0)

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # File selection
        ttk.Label(control_frame, text="Image:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.file_entry = ttk.Entry(control_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        browse_button = ttk.Button(control_frame, text="Browse...", command=self.browse_file)
        browse_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Method selection
        ttk.Label(control_frame, text="Method:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.method_combobox = ttk.Combobox(control_frame, width=38)
        self.method_combobox.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        run_button = ttk.Button(control_frame, text="Run OCR", command=self.run_ocr)
        run_button.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Middle frame for image and result
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Left side for image
        image_frame = ttk.LabelFrame(middle_frame, text="Passport Image")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side for results
        result_frame = ttk.LabelFrame(middle_frame, text="Extracted Information")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, width=40, height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom frame for log
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill=tk.X, expand=False, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=2)

    def log(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Passport Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            self.image_path = filename
            self.display_image(filename)

    def display_image(self, image_path):
        try:
            # Open and resize image to fit the display area
            img = Image.open(image_path)
            img.thumbnail((400, 400))
            
            # Convert to PhotoImage for display
            photo_img = ImageTk.PhotoImage(img)
            
            # Update image label
            self.image_label.configure(image=photo_img)
            self.image_label.image = photo_img  # Keep a reference
            
            self.current_image = img
            self.log(f"Loaded image: {image_path}")
        except Exception as e:
            self.log(f"Error loading image: {str(e)}")

    def run_ocr(self):
        if not self.image_path:
            self.log("Please select an image first.")
            return
        
        method = self.method_combobox.get()
        
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Run OCR in a separate thread to keep UI responsive
        threading.Thread(target=self.process_image, args=(method,), daemon=True).start()

    def process_image(self, method):
        self.update_status(f"Processing with {method}...")
        start_time = time.time()
        
        try:
            if method == "Traditional OCR (Tesseract)" and TESSERACT_AVAILABLE:
                result = self.traditional_ocr()
            elif method == "Donut (Fine-tuned)" and TRANSFORMERS_AVAILABLE:
                result = self.donut_ocr("fine-tuned")
            elif method == "Donut (Base)" and TRANSFORMERS_AVAILABLE:
                result = self.donut_ocr("base")
            elif method == "Donut (Receipt Model)" and TRANSFORMERS_AVAILABLE:
                result = self.donut_ocr("receipt")
            else:
                result = {"error": "Selected method is not available."}
            
            # Display result
            self.display_result(result)
            
        except Exception as e:
            self.log(f"Error processing image: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            self.result_text.insert(tk.END, f"Error: {str(e)}")
        
        elapsed_time = time.time() - start_time
        self.update_status(f"Completed in {elapsed_time:.2f} seconds.")

    def traditional_ocr(self):
        self.log("Starting traditional OCR with Tesseract...")
        
        # Preprocess the image
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Run OCR
        text = pytesseract.image_to_string(thresh)
        self.log(f"Raw OCR text: {text[:100]}...")
        
        # Extract information using regex patterns
        import re
        
        patterns = {
            'passport_type': r'Type\s*[:]?\s*([A-Z0-9]+)',
            'passport_no': r'Passport No[.:]?\s*([A-Z0-9]+)',
            'surname': r'Surname\s*[:]?\s*([A-Za-z\s\-\']+)',
            'given_names': r'Given Names\s*[:]?\s*([A-Za-z\s\-\']+)',
            'nationality': r'Nationality\s*[:]?\s*([A-Za-z\s]+)',
            'birth_date': r'Date of Birth\s*[:]?\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
            'gender': r'Sex\s*[:]?\s*([MF])',
            'issue_date': r'Date of Issue\s*[:]?\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
            'expiry_date': r'Date of Expiry\s*[:]?\s*(\d{1,2}\s*[A-Za-z]{3}\s*\d{4}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})',
        }
        
        # Extract information
        extracted_info = {"raw_text": text}
        for field, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                extracted_info[field] = match.group(1).strip()
        
        self.log(f"Extracted {len(extracted_info) - 1} fields from passport")
        return extracted_info

    def donut_ocr(self, model_type):
        if model_type == "fine-tuned":
            model_path = "./fine-tuned-donut-ocr"
            if not os.path.exists(model_path):
                self.log(f"Model path {model_path} not found, falling back to base model")
                model_type = "base"
        
        if model_type == "base":
            model_path = "naver-clova-ix/donut-base"
        elif model_type == "receipt":
            model_path = "naver-clova-ix/donut-base-finetuned-cord-v2"
        
        self.log(f"Loading Donut model: {model_path}")
        
        # Load model (reuse if already loaded)
        if model_path not in self.models:
            self.log("First use of this model, loading...")
            processor = AutoProcessor.from_pretrained(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to("cuda")
                self.log("Using GPU for inference")
            else:
                self.log("Using CPU for inference")
            
            self.models[model_path] = (processor, model)
        else:
            self.log("Reusing previously loaded model")
            processor, model = self.models[model_path]
        
        # Process image
        image = Image.open(self.image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt", legacy=False).pixel_values
        
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")
        
        # Generate
        prompt = "extract information:"
        self.log(f"Running inference with prompt: '{prompt}'")
        
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        if torch.cuda.is_available():
            decoder_input_ids = decoder_input_ids.input_ids.to("cuda")
        else:
            decoder_input_ids = decoder_input_ids.input_ids
        
        # Different parameters based on model type
        if model_type == "receipt":
            # Receipt model works better with beam search
            outputs = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                num_beams=4,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        else:
            # Try with sampling for other models
            outputs = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=256,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decode prediction
        prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        self.log(f"Raw model output: {prediction[:100]}...")
        
        # Try to parse as JSON
        result = {"raw_output": prediction}
        try:
            parsed = json.loads(prediction)
            self.log("Successfully parsed as JSON")
            result.update(parsed)
        except json.JSONDecodeError:
            self.log("Could not parse as JSON, returning raw output")
            
            # Try to extract structured info with regex as fallback
            import re
            patterns = [
                r'([A-Za-z\s]+):\s*([^:]+)',  # field: value
                r'([A-Za-z\s]+)\s*-\s*([^-]+)'  # field - value
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, prediction)
                for match in matches:
                    key = match[0].strip()
                    value = match[1].strip()
                    if key and value:
                        result[key] = value
        
        return result

    def display_result(self, result):
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        if "error" in result:
            self.result_text.insert(tk.END, f"Error: {result['error']}")
            return
        
        # Format the result
        if "raw_text" in result:
            raw_text = result.pop("raw_text")
            self.result_text.insert(tk.END, "Extracted Information:\n\n")
            
            for key, value in result.items():
                self.result_text.insert(tk.END, f"{key}: {value}\n")
            
            self.result_text.insert(tk.END, "\n\nRaw OCR Text:\n")
            self.result_text.insert(tk.END, raw_text)
        
        elif "raw_output" in result:
            raw_output = result.pop("raw_output")
            
            # Display structured fields first
            self.result_text.insert(tk.END, "Extracted Information:\n\n")
            
            for key, value in result.items():
                if key != "raw_output":
                    self.result_text.insert(tk.END, f"{key}: {value}\n")
            
            self.result_text.insert(tk.END, "\n\nRaw Model Output:\n")
            self.result_text.insert(tk.END, raw_output)
        
        else:
            # Just display everything as JSON
            formatted_json = json.dumps(result, indent=2)
            self.result_text.insert(tk.END, formatted_json)

def main():
    root = tk.Tk()
    app = PassportOCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()