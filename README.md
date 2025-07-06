# PRODIGY_GA_01
import torch
import warnings
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Suppress deprecation warnings (optional, remove if debugging)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Safely set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded")

# Load dataset from file
try:
    with open('sample_dataset.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    # Split into paragraphs and remove empty ones
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    dataset = Dataset.from_dict({"text": paragraphs})
    print(f"Loaded {len(dataset)} examples from sample_dataset.txt")
except FileNotFoundError:
    print("Error: sample_dataset.txt not found. Please create the file with your training data.")
    raise

# Split dataset into train and validation (10% for validation if dataset is large enough)
if len(dataset) > 10:  # Only split if enough examples
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"Split dataset: {len(train_dataset)} train examples, {len(eval_dataset)} validation examples")
else:
    train_dataset = dataset
    eval_dataset = dataset  # Use same dataset for small datasets
    print("Dataset too small for splitting, using same data for train and eval")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
print("Datasets tokenized")

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling for GPT-2
)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
print("Model loaded")

# Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)
print("Starting training...")

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained('fine_tuned_gpt2')
tokenizer.save_pretrained('fine_tuned_gpt2')
print("Model and tokenizer saved to fine_tuned_gpt2")

# Evaluate model (perplexity)
eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results['eval_loss'])).item()
print(f"Perplexity: {perplexity:.2f}")

# Text generation
input_text = "In the bustling city"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

sample_output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=150,
    num_beams=5,
    no_repeat_ngram_size=3,
    early_stopping=True,
    pad_token_id=tokenizer.eos_token_id,
    top_p=0.9
)

generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
print(f"\nPrompt: {input_text}")
print(f"Generated: {generated_text}")
