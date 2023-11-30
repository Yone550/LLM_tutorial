import time
import argparse
import torch

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

def generate_text(checkpoint_dir:str,
                  input_text:str,
                  fp16:bool,
                  bnb_4bit_quant:bool,
                  max_new_tokens:int,
                  top_k:float,
                  top_p:float,
                  temperature:float):
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    config = PeftConfig.from_pretrained(checkpoint_dir)
    
    if(fp16):
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
        model.eval()
        model = model.to(device)
    elif(bnb_4bit_quant):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model.eval()
        model = model.to(device)


    model = PeftModel.from_pretrained(model, checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


    print("Model Information")
    print(model)
    print(model.generation_config)
    print("Tokenizer Information")
    print(tokenizer)
    print(tokenizer.special_tokens_map)


    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True)
        print(inputs)

        inputs = inputs.to(device)
        outputs = model.generate(**inputs, 
                                 max_new_tokens=max_new_tokens, 
                                 top_k=top_k, 
                                 top_p=top_p, 
                                 temperature=temperature, 
                                 do_sample=True,
                                #  num_beams=4,
                                #  pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id,
                                 )
    
    # https://github.com/huggingface/transformers/issues/19764
    # model.generate() method outputs the tensor including inputs tensor
    #
    print(type(outputs))
    print(outputs.size())
    print(outputs)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return


    

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_text', type=str, default="Translate this sentence into Arabic:How are you today?\n")
    parser.add_argument("--checkpoint_dir_path", type=str, default="./output/saved_model")
    parser.add_argument('--seed', default=2023)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--bnb_4bit_quant', type=bool, default=False)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--top_p', type=float, default=0.875)
    parser.add_argument('--temperature', type=float, default=0.25)
    parser.add_argument('--show_generation_time', type=bool, default=True)


    args = parser.parse_args()

    return(args)

if __name__ == '__main__':

    args = get_args()

    input_text = args.input_text
    checkpoint_dir = args.checkpoint_dir_path
    fp16 = args.fp16
    bnb_4bit_quant = args.bnb_4bit_quant
    max_new_tokens = args.max_new_tokens
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature

    start = time.perf_counter()
    generate_text(input_text=input_text,
                  checkpoint_dir=checkpoint_dir,
                  fp16=fp16,
                  bnb_4bit_quant=bnb_4bit_quant,
                  max_new_tokens=max_new_tokens,
                  top_k=top_k,
                  top_p=top_p,
                  temperature=temperature)
    end = time.perf_counter()
    if(args.show_generation_time):
        print("generation time: {} seconds.".format(end - start))
