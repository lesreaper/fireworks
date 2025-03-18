import os
import json
import torch
import logging
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DonutProcessor, 
    VisionEncoderDecoderModel, 
    Trainer, 
    TrainingArguments
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "naver-clova-ix/donut-base"  # Document understanding transformer
DATASET_PATH = "passports_dataset.jsonl"
OUTPUT_DIR = "./fine-tuned-donut-ocr"
CHECKPOINT_DIR = "./donut-checkpoints"  # Custom checkpoint directory
MAX_LENGTH = 128  # Sequence length
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
CHECKPOINT_INTERVAL = 100  # Save custom checkpoint every N steps
ENABLE_RESUME = True  # Set to True to resume from checkpoint if available

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Function to find latest checkpoint
def find_latest_checkpoint():
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-step-*.pt")
    if not checkpoints:
        return None
    
    # Extract step numbers and find the max
    steps = [int(chkpt.split('-step-')[1].split('.pt')[0]) for chkpt in checkpoints]
    latest_step = max(steps)
    latest_checkpoint = f"{CHECKPOINT_DIR}/checkpoint-step-{latest_step}.pt"
    
    logger.info(f"Found checkpoint at step {latest_step}: {latest_checkpoint}")
    return latest_checkpoint, latest_step

# Custom checkpoint saver
def save_custom_checkpoint(model, optimizer, scheduler, step, loss):
    checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-step-{step}.pt"
    
    # Save both the wrapper and the original model
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    # Create checkpoint with enough info to resume training
    checkpoint = {
        'step': step,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }
    
    # Save using PyTorch's save to avoid safetensors issues
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved custom checkpoint at step {step} to {checkpoint_path}")
    
    # Cleanup old checkpoints - keep only the 3 most recent
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-step-*.pt")
    if len(checkpoints) > 3:
        # Sort by modification time (oldest first)
        checkpoints.sort(key=os.path.getmtime)
        # Remove the oldest ones
        for old_checkpoint in checkpoints[:-3]:
            logger.info(f"Removing old checkpoint: {old_checkpoint}")
            os.remove(old_checkpoint)

# Load model and processor
logger.info(f"Loading {MODEL_ID}...")
processor = DonutProcessor.from_pretrained(MODEL_ID)

# Check for checkpoint to resume from
start_step = 0
if ENABLE_RESUME:
    checkpoint_info = find_latest_checkpoint()
    if checkpoint_info:
        checkpoint_path, step = checkpoint_info
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize the model
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint['step']
        logger.info(f"Resumed training from step {start_step}")
    else:
        logger.info("No checkpoint found, starting from scratch")
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
else:
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

# Move model to device
model = model.to(device)
logger.info(f"Model loaded and moved to {device}")

# Dataset class for document OCR
class PassportOCRDataset(Dataset):
    def __init__(self, dataset_path, processor, max_length=128):
        self.processor = processor
        self.max_length = max_length
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        self.examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    self.examples.append(example)
                except Exception as e:
                    logger.warning(f"Error parsing line: {e}")
        
        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            
            # Get image path and verify it exists
            image_path = example["image"]
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return {"skip": True}
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Create text prompt
            prompt = "extract information:"
            response = json.dumps(example["response"])
            
            # Log examples
            if idx < 3:
                logger.info(f"Example {idx}:")
                logger.info(f"  Response: {response[:100]}...")
            
            # Process image with legacy=False to address the warning
            pixel_values = self.processor(image, return_tensors="pt", legacy=False).pixel_values.squeeze()
            
            # Create decoder input ids - limit to max_length
            decoder_input_ids = self.processor.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).input_ids.squeeze()
            
            # Process the target labels - limit to max_length
            labels = self.processor.tokenizer(
                response,
                add_special_tokens=False,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).input_ids.squeeze()
            
            # Return the processed features
            return {
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids,
                "labels": labels
            }
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return {"skip": True}

# Custom data collator
def data_collator(batch):
    # Filter out skipped items
    batch = [b for b in batch if "skip" not in b]
    if not batch:
        return None
    
    # Prepare batch
    collated = {}
    
    # Stack pixel values
    if "pixel_values" in batch[0]:
        collated["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    
    # Process decoder input ids
    if "decoder_input_ids" in batch[0]:
        max_length = max(b["decoder_input_ids"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_length), 0, dtype=torch.long)
        for i, b in enumerate(batch):
            length = min(b["decoder_input_ids"].shape[0], max_length)  # Prevent overflow
            padded[i, :length] = b["decoder_input_ids"][:length]
        collated["decoder_input_ids"] = padded
    
    # Process labels - ensure consistent dimensions with the model's expected output
    if "labels" in batch[0]:
        max_length = max(b["labels"].shape[0] for b in batch)
        padded = torch.full((len(batch), max_length), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            length = min(b["labels"].shape[0], max_length)  # Prevent overflow
            padded[i, :length] = b["labels"][:length]
        collated["labels"] = padded
    
    return collated

# Custom model wrapper to handle dimension mismatches
class DonutModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values, decoder_input_ids, labels=None):
        # Log shapes for debugging (less frequent to reduce log clutter)
        if torch.distributed.is_initialized():
            is_main_process = torch.distributed.get_rank() == 0
        else:
            is_main_process = True
            
        if is_main_process and torch.rand(1).item() < 0.01:  # Only log occasionally
            logger.info(f"decoder_input_ids shape: {decoder_input_ids.shape}, labels shape: {labels.shape if labels is not None else 'None'}")
        
        # Extend decoder_input_ids to match the length of labels
        if labels is not None:
            batch_size = decoder_input_ids.size(0)
            input_seq_len = decoder_input_ids.size(1)
            target_seq_len = labels.size(1)
            
            # Create padding
            padding_len = target_seq_len - input_seq_len
            if padding_len > 0:
                # Create padding (filled with 0, which is typically the pad token id)
                padding = torch.zeros((batch_size, padding_len), 
                                    dtype=decoder_input_ids.dtype, 
                                    device=decoder_input_ids.device)
                
                # Concatenate to extend decoder_input_ids
                extended_decoder_input_ids = torch.cat([decoder_input_ids, padding], dim=1)
                
                if is_main_process and torch.rand(1).item() < 0.01:  # Only log occasionally
                    logger.info(f"Extended decoder_input_ids shape: {extended_decoder_input_ids.shape}")
                
                # Forward pass with extended input ids
                outputs = self.model(
                    pixel_values=pixel_values,
                    decoder_input_ids=extended_decoder_input_ids,
                    labels=labels
                )
                return outputs
        
        # Default forward pass
        return self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

# Custom Trainer subclass to handle compatibility issues and safe saving
class DonutTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_step = start_step  # Initialize with the checkpoint step if resuming
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # We're accepting the num_items_in_batch parameter but not using it
        # This ensures compatibility with the parent Trainer class
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get loss
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _save(self, output_dir: str):
        """
        Override the _save method to handle shared tensors properly
        """
        # Get the actual model (not the wrapper)
        if hasattr(self.model, 'model'):
            actual_model = self.model.model
        else:
            actual_model = self.model
            
        actual_model.save_pretrained(
            output_dir,
            is_main_process=self.args.should_save,
            state_dict=actual_model.state_dict(),
            save_function=torch.save,  # Use torch.save instead of safetensors
        )
        
        if self.args.should_save:
            # Save the tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
        # Also save our custom checkpoint
        if self.args.should_save:
            save_custom_checkpoint(
                self.model, 
                self.optimizer, 
                self.lr_scheduler, 
                self.state.global_step,
                self.state.log_history[-1]['loss'] if self.state.log_history else 0
            )
    
    def training_step(self, model, inputs):
        """Override training_step to add custom checkpoint saving"""
        # Call the parent training_step
        loss = super().training_step(model, inputs)
        
        # Increment our custom step counter
        self.custom_step += 1
        
        # Save custom checkpoint at specified intervals
        if self.custom_step % CHECKPOINT_INTERVAL == 0:
            save_custom_checkpoint(
                model, 
                self.optimizer, 
                self.lr_scheduler, 
                self.custom_step,
                loss.item()
            )
            
        return loss

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    # No mixed precision (can enable if needed)
    fp16=False,
    bf16=False,
    # Logging and saving
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    # Other
    dataloader_drop_last=True,
    dataloader_num_workers=0,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    # Add debug mode
    debug="underflow_overflow",
    # Add label smoothing to improve stability
    label_smoothing_factor=0.1,
    # Use PyTorch save instead of safetensors
    save_safetensors=False,
)

# Main execution
try:
    # Clear cache
    torch.cuda.empty_cache()
    
    # Create dataset
    dataset = PassportOCRDataset(DATASET_PATH, processor, max_length=MAX_LENGTH)
    
    # Wrap the model to handle dimension mismatches
    wrapped_model = DonutModelWrapper(model)
    
    # Initialize custom trainer
    trainer = DonutTrainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,  # Pass tokenizer for saving
    )
    
    # Start training
    logger.info(f"Starting training from step {start_step}...")
    trainer.train()
    
    # Save final model (the original model, not the wrapper) using PyTorch save
    logger.info(f"Saving final model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR, save_function=torch.save)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info("Training completed successfully.")
    
except Exception as e:
    logger.error(f"Training failed: {e}")
    import traceback
    logger.error(traceback.format_exc())
    
    # Save checkpoint on error to ensure we don't lose progress
    try:
        error_checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-error.pt"
        
        # Get the actual model if it's wrapped
        if hasattr(trainer.model, 'model'):
            actual_model = trainer.model.model
        else:
            actual_model = trainer.model
            
        # Create error checkpoint
        error_checkpoint = {
            'step': trainer.state.global_step,
            'model_state_dict': actual_model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict() if hasattr(trainer, 'optimizer') else None,
            'scheduler_state_dict': trainer.lr_scheduler.state_dict() if hasattr(trainer, 'lr_scheduler') else None,
            'loss': trainer.state.log_history[-1]['loss'] if trainer.state.log_history else 0
        }
        
        # Save using PyTorch's save
        torch.save(error_checkpoint, error_checkpoint_path)
        logger.info(f"Saved error checkpoint to {error_checkpoint_path}")
    except Exception as save_e:
        logger.error(f"Failed to save error checkpoint: {save_e}")
    
    # Try to save partial model
    try:
        partial_path = f"{OUTPUT_DIR}_partial"
        model.save_pretrained(partial_path, save_function=torch.save)
        processor.save_pretrained(partial_path)
        logger.info(f"Saved partial model to {partial_path}")
    except Exception as save_e:
        logger.error(f"Failed to save partial model: {save_e}")

# Example inference code
"""
# How to use the fine-tuned model:
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load fine-tuned model
processor = DonutProcessor.from_pretrained("./fine-tuned-donut-ocr")
model = VisionEncoderDecoderModel.from_pretrained("./fine-tuned-donut-ocr")
model = model.to("cuda")

# Load image
image = Image.open("passport.jpg").convert("RGB")

# Process image
pixel_values = processor(image, return_tensors="pt").pixel_values.to("cuda")

# Generate output
prompt = "extract information:"
decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=5,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# Decode prediction
prediction = processor.decode(outputs.sequences[0], skip_special_tokens=True)
print(prediction)
"""