import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Use CPU
device = 'cpu'
print(f'Using device: {device}')

# 1. Load Model (TinyLlama)
model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
print(f'Loading {model_id}...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
print('Model loaded')

# 2. Prepare Dummy Dataset
data = [
    {'text': 'User: How do I learn RAG? Assistant: Start with the RAG Lab! '},
    {'text': 'User: What is Docker? Assistant: A tool to containerize apps. '},
    {'text': 'User: Who is Antigravity? Assistant: An agentic AI coding assistant. '}
] * 5

dataset = Dataset.from_list(data)
print(f'Dataset size: {len(dataset)}')

# 3. Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=4,            # Rank
    lora_alpha=16, 
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()

# 4. Train (1 Step Demo)
sft_config = SFTConfig(
    dataset_text_field='text',
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    max_steps=500, # Increased to 500 steps
    logging_steps=50,
    use_cpu=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)

print('Starting training...')
trainer.train()
print('Training complete!')

# 5. Save
output_adapter_dir = './tinyllama_lora_adapter'
trainer.save_model(output_adapter_dir)
print(f'Adapter saved to {output_adapter_dir}')

# Clean up memory
import gc
del model
del trainer
gc.collect()
print('Memory cleaned. Reloading base model for inference...')
base_model = AutoModelForCausalLM.from_pretrained(model_id)

print('Loading adapter...')
finetuned_model = PeftModel.from_pretrained(base_model, output_adapter_dir)

test_prompt = 'User: Who is Antigravity? Assistant: '
inputs = tokenizer(test_prompt, return_tensors='pt')

print('Generating response...')
outputs = finetuned_model.generate(**inputs, max_new_tokens=30)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print('-'*20)
print(result)
print('-'*20)
