+++
title = 'Implement and Understand PEFT Fine-tune Bloom-560m-tagger'
date = 2024-02-02T18:31:22+05:30
draft = false
+++

Fantastic! Let's dive right in.

---

Welcome to your next adventure in machine learning! In this guide, we'll explore the PEFT Fine-tune Bloom-560m-tagger project, a cutting-edge endeavor that combines the power of language models with efficient fine-tuning techniques. Whether you're a seasoned data scientist or a curious enthusiast, this tutorial will equip you with the knowledge and tools to implement and understand the intricacies of this project.

## Setting Up the Environment
Before we dive into the code, let's ensure our workspace is ready. Here’s how we kick things off:

### Installing Necessary Libraries
```bash
!pip install -q bitsandbytes datasets accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
```

Here, we install the essential libraries needed for our project:
- `bitsandbytes`: Optimizes deep learning models for faster training and reduced memory usage.
- `datasets`: Provides easy access to a vast range of datasets, simplifying data loading and preprocessing.
- `accelerate`: A library by Hugging Face to speed up computations on your machine.
- `loralib`: Likely a specialized library, although its specific use isn't detailed here, it's presumably important for our project.
- We also fetch the latest versions of `transformers` and `peft` from their respective GitHub repositories.

### Initializing the Workspace
```python
from huggingface_hub import notebook_login

notebook_login()
```
This line imports and invokes the `notebook_login` function from the `huggingface_hub` library, which prompts you to log in to Hugging Face's Hub. This step is crucial for accessing models and pushing your trained model to the Hub.

```bash
!nvidia-smi -L
```
We use `nvidia-smi -L` to list NVIDIA GPUs available in our environment, ensuring we have the necessary hardware acceleration for training our model.

### Preparing the Model and Tokenizer
Now, let's set up our model and tokenizer:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloomz-560m",
    load_in_8bit=True,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
```

- We ensure that our script uses the first GPU by setting `CUDA_VISIBLE_DEVICES="0"`.
- `torch` and `torch.nn` are imported for neural network operations.
- `bitsandbytes` is also imported, likely to enhance model training efficiency.
- We then load a pre-trained model and tokenizer using `AutoModelForCausalLM` and `AutoTokenizer` from the `bigscience/bloomz-560m` repository. The `load_in_8bit` option optimizes memory usage by loading the model in 8-bit precision, and `device_map='auto'` automatically places the model's layers onto available devices for optimal performance.

### Freezing the Model Parameters
```python
for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
```
Here, we freeze all parameters of the model to prevent them from being updated during training. This is essential for fine-tuning specific parts of the model, like adapters, without altering the entire network. Parameters with one dimension are cast to 32-bit floating point for numerical stability.

### Enabling Gradient Checkpointing and Input Gradients
```python
model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()
```
Gradient checkpointing is activated to reduce memory usage during training, and input gradients are enabled, allowing the model to compute gradients with respect to its inputs.

### Adjusting the Model’s Output
```python
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)
```
This custom class ensures the model's output is cast to 32-bit floating point, which is necessary for maintaining numerical stability and performance.

### Integrating PEFT with LoRA
Now, we integrate the Parameter Efficient Fine-tuning (PEFT) with LoRA (Low-Rank Adaptation) techniques:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, # attention heads
    lora_alpha=32, # alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
```
- `LoraConfig` is used to define the configuration for LoRA, specifying the number of attention heads (`r`), the alpha scaling factor (`lora_alpha`), dropout rate, and task type.
- `get_peft_model` integrates these settings into our model, preparing it for fine-tuning with both PEFT and LoRA mechanisms.

### Print Trainable Parameters Function
```python
print_trainable_parameters(model)
```
This function, presumably defined earlier in the code, outputs the number of trainable parameters, helping us understand the model’s complexity and the scale of the fine-tuning process.

### Loading the Dataset
```python
from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")
```
We load a dataset of English quotes, which will be used for training the model. The `datasets` library simplifies the process of fetching and preparing data for training.

### Preprocessing the Data
```python
def merge_columns(example):
    example["prediction"] = example["quote"] + " ->: " + str(example["tags"])
    return example

data['train'] = data['train'].map(merge_columns)
data['train'][0]
```
In this section, we preprocess the data by merging the `quote` and `tags` columns, formatting it for the model's expected input structure.

### Tokenizing the Data
```python
data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)
```
The dataset is tokenized using our previously loaded tokenizer, converting text data into a format suitable for model training.

### Setting Up the Trainer
```python
trainer = transformers.Trainer(
    model=model,
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=25,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model

.config.use_cache = False  # silence the warnings. Please re-enable for inference!
```
Here, we set up the `Trainer` object from the `transformers` library, specifying training arguments like batch size, learning rate, and output directory. The `data_collator` is used to batch and prepare the data for training the language model.

### Training the Model
```python
trainer.train()
```
With everything set up, we kick off the training process using the `train` method of our `Trainer` object.

### Sharing the Model on Hugging Face Hub
```python
model.push_to_hub("<your_username>/bloomz-560m-tagger",
                  use_auth_token=True,
                  commit_message="basic training",
                  private=True)
```
After training, we push the model to the Hugging Face Hub, making it accessible for further use and sharing.

### Loading and Using the Trained Model
```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "<your_username>/bloomz-560m-tagger"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id)

batch = tokenizer("“Training models with PEFT and LoRa is cool” ->: ", return_tensors='pt')

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
```
In the final section, we demonstrate how to load and use the fine-tuned model. We tokenize a sample prompt and generate text using the model, showcasing the practical application of our fine-tuned language model.

## Conclusion
Congratulations! You've just navigated through the comprehensive setup and execution of the PEFT Fine-tune Bloom-560m-tagger project. By breaking down each step and understanding the code, you've gained valuable insights into the world of machine learning and model fine-tuning. This hands-on project not only enhances your coding skills but also deepens your understanding of advanced AI methodologies. Now, it's your turn to experiment, explore, and excel in your machine learning journey!