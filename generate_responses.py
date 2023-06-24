import sys
import random
SEED=42
random.seed(SEED)

import os
import openai
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import BlenderbotTokenizerFast,BlenderbotForConditionalGeneration
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
from torch import Tensor,nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import notebook_launcher
from torch import nn, Tensor
import json
import os
from torch.utils.data.dataset import IterableDataset,Dataset

from prompt import *
from datautils import *


def to_device(batch_or_tensor, device, non_blocking: bool = False) :
   
    if isinstance(batch_or_tensor, Tensor):
        return batch_or_tensor.to(device, non_blocking=non_blocking)
    elif isinstance(batch_or_tensor, dict):
        for key, value in batch_or_tensor.items():
            if isinstance(value, Tensor):
                batch_or_tensor[key] = value.to(device, non_blocking=non_blocking)
        return batch_or_tensor
    elif isinstance(batch_or_tensor, (list, tuple)):
        for key, value in enumerate(batch_or_tensor):
            if isinstance(value, Tensor):
                batch_or_tensor[key] = value.to(device, non_blocking=non_blocking)
        return batch_or_tensor
    else:
        raise NotImplementedError(f"Not supported type: {type(batch_or_tensor)}")

def generate_responses(tokenizer,model,dataloader,precision_mode,write_to_file=None):
    #print("Write to file")
    #print(write_to_file)
    accelerator=Accelerator(mixed_precision=precision_mode)
    model,dataloader=accelerator.prepare(model,dataloader)
    model.eval()
    model.to(accelerator.device)
    #model.to(torch.device("cuda:0"))
    response_list=[]
    for batch in tqdm(dataloader):
        
        batch["input_ids"]=batch["input_ids"].squeeze(1)
        batch["attention_mask"]=batch["attention_mask"].squeeze(1)
        sub_batch={}
        sub_batch["input_ids"]=batch["input_ids"]
        sub_batch["attention_mask"]=batch["attention_mask"]
        sub_batch=to_device(sub_batch,accelerator.device)
        with torch.no_grad():

            if accelerator.state.num_processes == 1:
                preds=model.generate(**sub_batch,min_length=24,max_length=128)
            else:
                preds=model.module.generate(**sub_batch,min_length=24,max_length=128)
            responses=tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        prompts=[x for x in batch["text"]]      
        responses=[x for x in responses]
        if "gold_response" in batch:
            gold_response=[x.strip() for x in batch["gold_response"]]
        if "summary" in batch:
            summaries=[x.strip() for x in batch["summary"]]
        if "current_utterance" in batch:
            current_utterance=[x.strip() for x in batch["current_utterance"]]
        if "history" in batch:
            history=[x.strip() for x in batch["history"]]
        if "id" in batch:
            ids=batch["id"].tolist()
        for i,response in enumerate(responses):
            response_list.append({"prompts":prompts[i],"current_utterance":current_utterance[i],"predicted_response":responses[i],"gold_response":gold_response[i],"history":history[i],"summary":summaries[i]})

    for entry in response_list:
        print(json.dumps(entry)+"\n")

def gpt3_generate(prompts, openai_key):
    openai.api_key = openai_key
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompts,
        temperature=1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response 

def delayed_batched_gpt3_generate(prompts, openai_key, batch_size=8):
    import time
    rate_limit = 50 # per minute
    batched_prompts = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    responses = []
    prev_time = time.time()

    for batch in tqdm(batched_prompts):
        max_exceptions = 3
        exception_delay = 5

        subset = None
        while subset is None:
            try:
                prompt_batch = [x["prompt"] for x in batch]
                subset = gpt3_generate(prompt_batch, openai_key)
                break
            except Exception as e:
                max_exceptions -= 1
                time.sleep(exception_delay)
                if max_exceptions == 0:
                    print("Too many exceptions, skipping batch")
                    print(e)
                    break
                else:
                    print("Exception encountered, retrying")
                    continue
        if subset is not None:
            responses.extend(subset["choices"])
        else:
            dummy_openai_response = {
                "choices": [{"text": ""} for _ in batch]
            }
            responses.extend(dummy_openai_response["choices"])
        print(responses[-1])
        curr_time = time.time()
        delay_needed = 60 / rate_limit
        if curr_time - prev_time < delay_needed:
            time.sleep(delay_needed - (curr_time - prev_time))
        prev_time = curr_time
    
    return responses


def helper(input_file,
           bs,
           outfile,
           prompt_template,
           precision_mode,
           use_shorter_template=False,
           has_persona_only=False,
           has_knowledge_only=False,
           has_persona_and_summary=False,
           has_knowledge_and_summary=False,
           current_utterance_only=False,
           model_path="facebook/blenderbot-3B",
           bart_summary=False,
           use_fsb_prompt=False,
           segment_utt=False,
           num_processes=8):
    
    prompts=generate_prompts_from_json(input_file,
                                       prompt_template,
                                       current_utterance_only=current_utterance_only,
                                       use_shorter_template=use_shorter_template,
                                       bart_summary=bart_summary,
                                       has_persona_only=has_persona_only,
                                       has_persona_and_summary=has_persona_and_summary,
                                       has_knowledge_only=has_knowledge_only,
                                       has_knowledge_and_summary=has_knowledge_and_summary,
                                       use_fsb_prompt=use_fsb_prompt,
                                       segment_utt=segment_utt)
    if current_utterance_only:
        max_seq_length=256
    elif use_shorter_template:        
        max_seq_length=384
    else:
        max_seq_length=1024
    #print("Model path")
    #print(model_path)
    #print(model_path=="bigscience/T0_3B")
    if model_path=="facebook/blenderbot-3B":
        tokenizer = BlenderbotTokenizerFast.from_pretrained(model_path)
        tokenizer.add_prefix_space=False
        model = BlenderbotForConditionalGeneration.from_pretrained(model_path)
    elif model_path == "google/flan-t5-xl":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer=AutoTokenizer.from_pretrained(model_path)
    elif model_path == "bigscience/T0_3B":
        #print("Bigscience")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_path == "allenai/tk-instruct-3b-def-pos":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_path== "allenai/tk-instruct-3b-def":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_path=="EleutherAI/gpt-neo-2.7B":        
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        model = GPTNeoForCausalLM.from_pretrained(model_path)

    if model_path != "dv3":
        dataset=GenericDataset(model_path,prompts,max_seq_length=max_seq_length)
        dataloader = DataLoader(dataset,  batch_size=bs)
        args=(tokenizer,model,dataloader,precision_mode,outfile)
        #print(outfile)
        notebook_launcher(generate_responses,args,num_processes=num_processes)
    else:
        # OpenAI text-davinci-03
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        api_key = os.getenv("OPENAI_API_KEY")
        response_list=[]
        
        responses = delayed_batched_gpt3_generate(prompts, api_key, batch_size=bs)
        for i, sample in tqdm(enumerate(prompts), total=len(prompts)):
            # print(sample)
            response = responses[i]
            # if i==0:
            #     break
            response_list.append(sample)
            response_list[-1]["predicted_response"]=response["text"]

        # for i, sample in tqdm(enumerate(prompts), total=len(prompts)):
        #     # print(sample)
        #     response = gpt3_generate(sample['prompt'], api_key)
        #     # if i==0:
        #     #     break
        #     response_list.append(sample)
        #     response_list[-1]["predicted_response"]=response["choices"][0]["text"]
        
        for entry in response_list:
            print(json.dumps(entry)+"\n")

    

def compute_preds(output):
    with open(output,"r") as fp:
        content=fp.readlines()
    preds=list(set([x.strip() for x in content if len(x.strip())>0]))
    cnt=0
    filtered_preds=[]
    for entry in tqdm(preds):
        try:
            entry=json.loads(entry)
            #print(list(entry.keys()))
            if len(list(entry.keys())) in [6,7]:
                filtered_preds.append(entry)
                cnt+=1
        except:
            #print(entry)
            continue
    print(cnt)
    return filtered_preds


# Unit Test
if __name__ == "__main__":
    #Usage example when we do not use summary, persona or knowledge for zs prompt :
    input_dataset = "msc"
    output_folder="msc_GeneratedResponsesFromPreviousUtterances"
    outputfile_prefix="last4_previous_utt"
    inputfile="context_data/multi_session_chat/previous_utterances/msc_previous_utterances_last2.txt"
    prompt_template=pegasusft_template_pplbased_tk_instruct

    temp_output="intermediate_file.txt"
    stdout = sys.stdout
    sys.stdout = open(temp_output, 'w')
    helper(inputfile,
        24,
        temp_output,
        prompt_template=prompt_template,
        precision_mode="fp16",
        model_path="allenai/tk-instruct-3b-def",
        use_shorter_template=True)
    sys.stdout = stdout
    filtered_preds=compute_preds(temp_output)
    path = "outputs/{2}/{0}/"
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    fp=open("outputs/{2}/{0}/{1}_optimalppltemplate_tk_instruct.txt".format(output_folder,outputfile_prefix,input_dataset),"w")
    for entry in tqdm(filtered_preds):
        fp.write(json.dumps(entry)+"\n")
    fp.close()

