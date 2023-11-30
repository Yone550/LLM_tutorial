import argparse
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import LogitsProcessor,LogitsProcessorList


# https://towardsdatascience.com/challenges-in-stop-generation-within-llama-2-25f5fea8dea2
# To generate EOS token 
#
class EosTokenRewardLogitProcess(LogitsProcessor):
  def __init__(self, eos_token_id: int, max_length: int):

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        if not isinstance(max_length, int) or max_length < 1:
          raise ValueError(f"`max_length` has to be a integer bigger than 1, but is {max_length}")

        self.eos_token_id = eos_token_id
        self.max_length=max_length

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    cur_len = input_ids.shape[-1]
    # start to increese the reward of the eos_token from 60% max length progressively on length
    for cur_len in (max(0,int(self.max_length*0.60)), self.max_length ):
      ratio = cur_len/self.max_length
      num_tokens = scores.shape[1] # size of vocab
      scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] =\
       scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]*ratio*10*torch.exp(-torch.sign(scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]))
      scores[:, self.eos_token_id] = 1e2 * ratio
    return scores


def generate_text(model_name:str,
                  tokenizer_name:str, 
                  input_text:str,
                  fp16:bool,
                  bnb_4bit_quant:bool,
                  max_new_tokens:int,
                  top_k:float,
                  top_p:float,
                  temperature:float,
                  eos_logits_processor:bool = False):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if(tokenizer.pad_token == None):
        tokenizer.pad_token = tokenizer.eos_token
    if(fp16):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.eval()
        model = model.to(device)
    elif(bnb_4bit_quant):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model = model.to(device)

    print("Model Information")
    print(model)
    print(model.generation_config)
    print("Tokenizer Information")
    print(tokenizer)
    print(tokenizer.special_tokens_map)


    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True)
        print(inputs)


        if(eos_logits_processor):
            logits_process_list = LogitsProcessorList([EosTokenRewardLogitProcess(eos_token_id=tokenizer.eos_token_id, 
                                                                                  max_length=(len(inputs) + max_new_tokens))])
        else:
            # logits_process_list = LogitsProcessorList([LogitsProcessor()])
            logits_process_list = None

        print(logits_process_list)

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
                                 logits_processor=logits_process_list,
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

    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--input_text', type=str, default="Translate this sentence into Arabic:How are you today?\n")
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--bnb_4bit_quant', type=bool, default=False)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=4)
    parser.add_argument('--top_p', type=float, default=0.875)
    parser.add_argument('--temperature', type=float, default=0.25)
    parser.add_argument('--show_generation_time', type=bool, default=True)
    parser.add_argument('--eos_logits_processor', type=bool, default=False)

    args = parser.parse_args()

    return(args)

if __name__ == '__main__':
    args = get_args()

    model = args.model
    tokenizer = args.tokenizer
    input_text = args.input_text
    fp16 = args.fp16
    bnb_4bit_quant = args.bnb_4bit_quant
    max_new_tokens = args.max_new_tokens
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    eos_logits_processor = args.eos_logits_processor

    start = time.perf_counter()
    generate_text(model_name=model, 
                  tokenizer_name=tokenizer, 
                  input_text=input_text,
                  fp16=fp16,
                  bnb_4bit_quant=bnb_4bit_quant,
                  max_new_tokens=max_new_tokens,
                  top_k=top_k,
                  top_p=top_p,
                  temperature=temperature,
                  eos_logits_processor=eos_logits_processor)
    end = time.perf_counter()
    if(args.show_generation_time):
        print("generation time: {} seconds.".format(end - start))