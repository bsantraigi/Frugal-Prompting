import random
import re
import wandb
import argparse
import uuid
import os
import sys
from tqdm import tqdm
import json
import pandas as pd

import multiprocessing

def cmdline_args(x=None):
    """Argument Parser
    --dataset: Dataset to use (MSC or TC)
    --model: Model to use (flan-t5, tk-instruct, T0)
    --prompt_type: "manual" or "ppl"
    --few_shot: Use 1 example per along with each query, otherwise zero_shot
    --background_knowledge: Use background knowledge or not (in summarized format)
    --history_signal_type: Type of history signal (full, peg, bart, recent-k, semantic-k, none)
    --history_k: Number of utterances to use in recent-k or semantic-k
    --wandb: Use wandb or not
    --num_gpus: Number of GPUs to use
    """
    parser = argparse.ArgumentParser(description='Launcher')
    parser.add_argument('-d', '--dataset', type=str, default='MSC', help='Dataset to use (MSC or TC)', choices=["MSC", "TC"])
    parser.add_argument('-m', '--model', type=str, default='flan-t5', help='Model to use (flan-t5, tk-instruct, T0)', choices=["flan-t5", "T0", "tk-instruct", "dv3"])
    parser.add_argument('-bs', '--batch_size', type=int, default=12, help='Batch size for inference')
    parser.add_argument('-pt','--prompt_type',  type=str, default='ppl', help='manual or ppl', choices=["manual", "ppl"])
    parser.add_argument('-fs','--few_shot',  action="store_true", help='Use 1 example per along with each query, otherwise zero_shot')
    parser.add_argument('-bw','--background_knowledge',  action="store_true", help='Use background knowledge or not (in summarized format)')
    parser.add_argument('-hst','--history_signal_type',  type=str, default='peg', choices=['full', 'peg_cd', 'peg', 'bart', 'recent-k', 'semantic-k', 'none'],
                        help='Type of history signal (full, peg_cd, peg, bart, recent-k, semantic-k, none)')
    parser.add_argument('-hk','--history_k',  type=int, default=4, help='Number of utterances to use in recent-k or semantic-k', choices=[1,2,4,8,10])
    parser.add_argument('-w','--wandb',  action="store_true", help='Use wandb or not')
    parser.add_argument('-ngpu', '--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('-e', '--eval_mode', action="store_true", help='Evaluation mode')
    if x is not None:
        return parser.parse_args(x)
    else:
        return parser.parse_args()


def get_output_eval_filenames(args):
    ## Output file path generator
    path = f"{args.dataset}-evals/"
    # use all args to create a filename
    output_filename = f"{args.model}_" \
                        f"{args.prompt_type}_" \
                        f"{'fs' if args.few_shot else 'zs'}_" \
                        f"{'bk' if args.background_knowledge else 'nbk'}_" \
                        f"{args.history_signal_type}"
    if args.history_signal_type in ["recent-k", "semantic-k"]:
        output_filename += f"_{args.history_k}.jsonl"
    else:
        output_filename += f".jsonl"

    ## Eval file path generator
    eval_filename = output_filename.replace(".jsonl", ".eval.json")

    return path, output_filename, eval_filename

all_evals = []
if __name__ == "__main__":
    exp_list = open("data/cmds_superset.sh", "r").read().split("\n")
    print("Loading eval files")
    eval_files = []
    for exp in tqdm(exp_list):
        # merge spaces
        exp = re.sub(' +', ' ', exp)

        # Convert to args
        args = cmdline_args(exp.split(" ")[2:])
        path, output_filename, eval_filename = get_output_eval_filenames(args)
        # print(eval_filename)
        eval_files.append(eval_filename)

        try:
            # Load two json files from {dataset}-evals/evals and {dataset}-evals/evalsDEB
            evals = json.load(open(f"{path}evals/{eval_filename}", "r"))
            evalsDEB = json.load(open(f"{path}evalsDEB/{eval_filename}", "r"))

            # Merge the two json files
            evals.update(evalsDEB)
            all_evals.append(vars(args))
            all_evals[-1].update(evals)
        except FileNotFoundError as e:
            print(f"File not found: {eval_filename}")
            print(f"Command: {exp}")
    
    # Save all evals to a csv file
    df = pd.DataFrame(all_evals)


    # Rename columns. prompt_len -> prompt, output_len -> output, BLEURT -> Bleurt, model -> Model
    df = df.rename(columns={"prompt_len": "prompt", "output_len": "output", "BLEURT": "Bleurt", "model": "Model"})


    df['Persona Signal'] = "None"
    df['Persona Signal'][df['background_knowledge'] == True] = "Pegasus cnn/dm"

    # df['History Signal'] = ""
    # df['History Signal'][df['history_signal_type'] == "full"] = "Full"
    # df['History Signal'][df['history_signal_type'] == "peg_cd"] = "Pegasus-CD"
    # df['History Signal'][df['history_signal_type'] == "peg"] = "PegasusFT"
    # df['History Signal'][df['history_signal_type'] == "bart"] = "BART"
    # df['History Signal'][df['history_signal_type'] == "recent-k"] = "Recent-" + df['history_k'].astype(str)
    # df['History Signal'][df['history_signal_type'] == "semantic-k"] = "Semantic-" + df['history_k'].astype(str)
    # df['History Signal'][df['history_signal_type'] == "none"] = "None"
    df.rename(columns={"history_signal_type": "History Signal"}, inplace=True)
    df.replace({"History Signal": {
        "full": "Full", 
        "peg_cd": "Pegasus-CD", 
        "peg": "PegasusFT",
        "bart": "BART",
        "recent-k": "Recent",
        "semantic-k": "Semantic",
        "none": "None"
    }}, inplace=True)
    df.loc[df['History Signal'] == "Recent", "History Signal"] += "-" + df['history_k'].astype(str)
    df.loc[df['History Signal'] == "Semantic", "History Signal"] += "-" + df['history_k'].astype(str)

    # Few shot -> "Orig", Zero shot -> "Short"
    df['Prompt Type'] = ""
    df['Prompt Type'][df['few_shot'] == True] = "Orig"
    df['Prompt Type'][df['few_shot'] == False] = "Short"
    # Add (ppl), if so
    df['Prompt Type'][df['prompt_type'] == "ppl"] += " (ppl)"

    # Rename model -> Model
    df.rename(columns={"model": "Model"}, inplace=True)
    # flan-t5 -> flanT5-XL, tk-instruct -> Tk-Instruct, T0 -> T0, dv3 -> text-davinci-003
    df.replace({"Model": {"flan-t5": "flanT5-XL", "tk-instruct": "Tk-Instruct", "T0": "T0", "dv3": "text-davinci-003"}}, inplace=True)

    # Extra dummy cols
    used_cols = ['Model', 'Method', 'Persona Signal', 'History Signal', 'Prompt Type',
        'BLEU', 'METEOR', 'rouge1', 'rouge2', 'rougeL', 'Bert-p', 'Bert-r',
        'Bert-f1', 'DEB', 'Bleurt', 'output', 'prompt']
    for col in used_cols:
        if col not in df.columns:
            print(f"Adding dummy col: {col}")
            df[col] = 0.5
    
    # dv3 has prompt len missing (-1). fill it in from other models, with exact same (history, persona and prompt_type)
    query = df.loc[df['Model'] == "text-davinci-003", ['dataset', 'History Signal', 'Prompt Type', 'Persona Signal']]
    prompt_of_flant5 = pd.merge(query, df.loc[df['Model'] == "flanT5-XL", ['dataset', 'History Signal', 'Prompt Type', 'Persona Signal', 'prompt']], on=['dataset', 'History Signal', 'Prompt Type', 'Persona Signal'], how='left')['prompt']
    # df.loc[df['Model'] == "text-davinci-003", 'prompt'] = prompt_of_flant5    
    df.loc[query.index, 'prompt'] = prompt_of_flant5.values

    df.to_csv("data/evals.csv", index=False)