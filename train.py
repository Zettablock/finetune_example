import torch
import transformers
from datasets import load_dataset
from transformers import TrainingArguments
from dataclasses import dataclass, field
from unsloth import FastLanguageModel, is_bfloat16_supported
from utils import format_prompt
from trl import SFTTrainer
# import evaluate


load_in_4bit = True


@dataclass
class ModelArguments():
    model_id: str = field(
        default="unsloth/Meta-Llama-3.1-8B",
        metadata={'help': 'HF model id'}
    )

@dataclass
class DataArguments():
    data_id: str = field(
        default = "ruslanmv/ai-medical-chatbot",
        metadata={'help': 'HF dataset id'}
    )

# Train using unsloth
def train():
    # parser = transformers.HfArgumentParser(
    #     (ModelArguments, DataArguments, TrainingArguments)
    # )

    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments)
    )

    model_args, data_args = parser.parse_args_into_dataclasses()

    model_id = model_args.model_id
    max_seq_length = 512

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    config = {
        "lr": 5e-5,
        "epochs": 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        "lora_alpha": 64,
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "batch_size": 16,
        "bias": "none",
    }

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["r"],
        target_modules= config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,  
        loftq_config = None, 
    )

    EOS_TOKEN = tokenizer.eos_token
    print(f"EOS_TOKEN: {EOS_TOKEN}")

    dataset = load_dataset(data_args.data_id, split='train').select(range(20000))
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    train_dataset = train_dataset.map(lambda examples: format_prompt(examples, EOS_TOKEN=EOS_TOKEN), batched = True,)
    test_dataset = test_dataset.map(lambda examples: format_prompt(examples, EOS_TOKEN=EOS_TOKEN), batched = True,)

    print(train_dataset)

    # # Evaluate before fine-tuning
    # initial_metrics = evaluate_model(model, test_dataset, tokenizer, max_seq_length)
    # print("Initial metrics:", initial_metrics)


    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            
            # Use num_train_epochs = 1, warmup_ratio for full training runs!
            warmup_steps = 5,
            max_steps = 60,

            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "health-model",
            push_to_hub=True
        ),
    )

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Evaluate before fine-tuning
    # final_metrics = evaluate_model(model, test_dataset, tokenizer, max_seq_length)
    # print("Final metrics:", final_metrics)


    trainer.push_to_hub()

    # Merge to 16bit
    model.save_pretrained_merged("health-model", tokenizer, save_method = "merged_16bit",)
    # model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
    




def main():
    train()

if __name__=="__main__":
    main()






