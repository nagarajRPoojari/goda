import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


model_id = "meta-llama/Llama-3.2-1B"  

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normalized Float 4 (optimized for LLM weights)
    bnb_4bit_compute_dtype=torch.bfloat16, # Gradients are computed in 16-bit precision
    bnb_4bit_use_double_quant=True       # Quantizes the quantization constants for extra savings
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Standardize padding token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config, #
    device_map="auto"
)

peft_config = LoraConfig(
    r=16,                    # Rank dimension (higher means more capacity, typical: 8, 16, 32, 64)
    lora_alpha=32,           # Scaling factor (rule of thumb: 2x the rank)
    lora_dropout=0.05,       # Dropout probability for LoRA layers
    bias="none",
    task_type="CAUSAL_LM",   # Task type for auto-regressive language models
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target attention layers
)

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1%]") 

training_args = SFTConfig(
    output_dir="./qlora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, # Simulates a larger batch size
    learning_rate=2e-4,            # PEFT usually requires a higher LR than full fine-tuning
    logging_steps=10,
    max_seq_length=512,            # Max token length handled by SFTTrainer
    packing=True,                  # TRL feature: packs multiple small examples into one sequence
    bf16=True,                     # Use bfloat16 training if supported by hardware (else fp16=True)
    num_train_epochs=1,
    report_to="none"               # Can change to "wandb" or "tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("./final_qlora_adapter")
print("🎉 Fine-tuning complete and adapter weights saved successfully!")