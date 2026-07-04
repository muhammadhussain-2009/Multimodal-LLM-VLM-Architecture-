import os
import yaml
import logging
from typing import Dict, Any

# Configure logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LoRATrainer")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    import wandb
    from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAS_TRAINING_LIBS = True
except ImportError as e:
    logger.warning(f"ML packages not fully installed. Training script will run in simulation mode: {str(e)}")
    HAS_TRAINING_LIBS = False

class STEMDiagramDataset(object):
    """Placeholder dataset mapping to Hugging Face datasets formatting."""
    def __init__(self, data_path: str, processor: Any, transform: Any):
        self.data_path = data_path
        self.processor = processor
        self.transform = transform
        # Simulation dataset size
        self.size = 10

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return mock structures compatible with LLaVA model forward pass
        return {
            "input_ids": torch.randint(0, 10000, (1, 128)),
            "pixel_values": torch.randn(1, 3, 336, 336),
            "labels": torch.randint(0, 10000, (1, 128))
        }


class LoRAFinetuningPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
    def get_transforms(self) -> Any:
        """Sets up the image augmentation pipeline targeting messy student drawings."""
        if not HAS_TRAINING_LIBS:
            return None
            
        aug_cfg = self.config.get("image_augmentation", {})
        
        # Build transform sequence based on yaml parameters
        transform_list = [
            transforms.Resize(tuple(aug_cfg["resize"]["size"])),
            transforms.RandomResizedCrop(
                tuple(aug_cfg["resize"]["size"]),
                scale=tuple(aug_cfg["random_resized_crop"]["scale"]),
                ratio=tuple(aug_cfg["random_resized_crop"]["ratio"])
            ),
            transforms.ColorJitter(
                brightness=aug_cfg["color_jitter"]["brightness"],
                contrast=aug_cfg["color_jitter"]["contrast"],
                saturation=aug_cfg["color_jitter"]["saturation"]
            ),
            transforms.RandomAffine(
                degrees=aug_cfg["random_affine"]["degrees"],
                translate=tuple(aug_cfg["random_affine"]["translate"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(transform_list)

    def run_training(self) -> None:
        """Executes the training loop incorporating quantization and PEFT LoRA weights."""
        if not HAS_TRAINING_LIBS:
            logger.info("[SIMULATION MODE] Running mock training steps based on training_config.yaml details.")
            logger.info(f"Model selection: {self.config['model']['pretrained_model_name_or_path']}")
            logger.info(f"LoRA Rank: {self.config['model']['lora']['r']}, Alpha: {self.config['model']['lora']['lora_alpha']}")
            logger.info(f"Optimizer: {self.config['training']['optimizer']}, learning rate: {self.config['training']['learning_rate']}")
            logger.info("Training simulation completed successfully.")
            return

        # Initialize Weights & Biases
        wandb.init(
            project="multimodal-stem-vlm",
            config=self.config,
            name="llava-stem-lora-run"
        )
        
        model_name = self.config["model"]["pretrained_model_name_or_path"]
        logger.info(f"Loading pretrained vision-language model: {model_name}...")
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model in 8-bit or 4-bit depending on settings (optimized for budget limits)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )
        
        # Prepare for quantized training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA Config
        lora_cfg = self.config["model"]["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"]
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Prepare datasets
        img_transforms = self.get_transforms()
        train_dataset = STEMDiagramDataset("./data/processed/dataset_merged.json", processor, img_transforms)
        
        # Set training arguments
        train_args = self.config["training"]
        training_args = TrainingArguments(
            output_dir=train_args["output_dir"],
            num_train_epochs=train_args["num_train_epochs"],
            per_device_train_batch_size=train_args["per_device_train_batch_size"],
            gradient_accumulation_steps=train_args["gradient_accumulation_steps"],
            learning_rate=float(train_args["learning_rate"]),
            lr_scheduler_type=train_args["lr_scheduler_type"],
            warmup_ratio=train_args["warmup_ratio"],
            weight_decay=train_args["weight_decay"],
            optim=train_args["optimizer"],
            logging_steps=train_args["logging_steps"],
            save_strategy=train_args["save_strategy"],
            evaluation_strategy=train_args["evaluation_strategy"],
            fp16=train_args["fp16"],
            report_to=["wandb"]
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset, # Evaluating on train just for sample pipeline validation
        )
        
        logger.info("Starting VLM fine-tuning execution...")
        trainer.train()
        
        # Save checkpoints
        model.save_pretrained(os.path.join(train_args["output_dir"], "final_peft_adapter"))
        logger.info("Training complete and adapter weights saved.")
        wandb.finish()


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "./training_config.yaml"
    trainer = LoRAFinetuningPipeline(config_file)
    trainer.run_training()
