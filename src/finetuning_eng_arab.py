import sys
import argparse

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model, PrefixTuningConfig
from datasets import load_dataset


def data_arrange(data, tokenizer, input_column_name="sentence_eng_Latn", output_column_name='sentence_arz_Arab', max_token=128):

    input_q = 'Translate this sentence into Arabic:' + data[input_column_name] + '\n'
    tokenized_q = tokenizer(input_q, return_tensors='pt', padding='max_length', truncation=True, max_length=max_token)
    input_qa = 'Translate this sentence into Arabic:' + data[input_column_name] + '\n' + data[output_column_name] + tokenizer.special_tokens_map['eos_token']
    tokenized_qa = tokenizer(input_qa, return_tensors='pt', padding='max_length', truncation=True, max_length=max_token)
    labels = tokenized_qa['input_ids'][0].clone()
    for i in range(0, max_token):
        if not tokenized_q['input_ids'][0][i] == 1:
            labels[i] = -100
    return {
        'input': input_q,
        'input_ids': tokenized_q['input_ids'][0],
        'attention_mask': tokenized_q['attention_mask'][0],
        'labels': labels
    }

def fine_tuning(model_name:str,
                tokenizer_name:str, 
                checkpoint_dir_path:str,
                saved_model_name:str,
                learning_rate:float,
                epochs:int,
                save_step:int,
                seed:int):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if(tokenizer.pad_token == None):
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 quantization_config=bnb_config)

    # LoRA setting
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    # https://github.com/facebookresearch/flores/blob/main/flores200/README.md
    dataset = load_dataset("facebook/flores", "eng_Latn-arz_Arab")
    dataset_tokenized_train = dataset["dev"].map(
        data_arrange,
        fn_kwargs={"tokenizer":tokenizer},
        remove_columns=['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink']
    )
    dataset_tokenized_test = dataset["devtest"].map(
        data_arrange,
        fn_kwargs={"tokenizer":tokenizer},
        remove_columns=['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink']
    )


    training_args = TrainingArguments(
        output_dir=checkpoint_dir_path,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        evaluation_strategy='steps',
        per_device_train_batch_size=16,
        logging_steps=10,
        eval_steps=10,
        seed=seed,
        save_steps=save_step
    )

    trainer=Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_tokenized_train,
        eval_dataset=dataset_tokenized_test
    )

    train_result = trainer.train()
    trainer.evaluate()

    model.save_pretrained(checkpoint_dir_path + "/" + saved_model_name)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    # parser.add_argument("--dataset_dir_path", type=str)
    parser.add_argument("--checkpoint_dir_path", type=str, default="./output")
    parser.add_argument("--saved_model_name", type=str, default="saved_model")
    parser.add_argument('--train_size',  type=float, default=0.8)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--validation_size', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--save_steps',  type=int, default=200)
    parser.add_argument('--seed',  type=int, default=2023)


    args = parser.parse_args()

    return(args)

if __name__ == '__main__':
    args = get_args()

    model = args.model
    tokenizer = args.tokenizer
    checkpoint_dir_path = args.checkpoint_dir_path
    saved_model_name = args.saved_model_name
    learning_rate = args.learning_rate
    epochs = args.num_train_epochs
    save_step = args.save_steps
    seed = args.seed

    fine_tuning(model_name=model, 
                tokenizer_name=tokenizer, 
                checkpoint_dir_path=checkpoint_dir_path,
                saved_model_name=saved_model_name,
                learning_rate=learning_rate,
                epochs=epochs,
                save_step=save_step,
                seed=seed)
