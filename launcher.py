import random
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

def load_dataset(dataset, prompt_type, few_shot, background_knowledge, history_signal_type, history_k):
    """Load dataset
    """
    return []    

def full_model_path(model_path):
    if model_path=="BB":
        full_path = "facebook/blenderbot-3B"
    elif model_path == "flan-t5":
        full_path = "google/flan-t5-xl"
    elif model_path == "T0":
        full_path = "bigscience/T0_3B"
        #print("Bigscience")
    elif model_path == "tk-instruct":
        raise Exception("Multiple tk-insruct models. Please specify the exact model path")
        full_path = "allenai/tk-instruct-3b-def-pos"
    elif model_path== "tk-instruct-2":
        full_path = "allenai/tk-instruct-3b-def"
    elif model_path=="EleutherAI/gpt-neo-2.7B": 
        full_path = "EleutherAI/gpt-neo-2.7B"       

    return full_path


def search_template(args):
    """Search for the best template
    """
    # dv3 + ppl combination is invalid
    assert not (args.model=="dv3" and args.prompt_type=="ppl"), "dv3 + ppl combination is invalid"

    template_df = pd.read_csv("templates.csv")
    # Create new rows by splitting 'history_signal_type' by a semicolon
    template_df = template_df.assign(history_signal_type=template_df['history_signal_type'].str.split(';'))

    # Explode the new rows
    template_df = template_df.explode('history_signal_type')

    # MODs
    ## MOD-1
    hist_type = args.history_signal_type
    hist_type = "bart" if hist_type in ["peg", "peg_cd"] else hist_type # same template for peg and bart

    # model should be used if prompt_type is ppl
    if args.prompt_type=="ppl":
        template_df = template_df[ \
            (template_df["model"]==args.model) & \
            (template_df["prompt_type"]==args.prompt_type) & \
            (template_df["few_shot"]==args.few_shot) & \
            (template_df["background_knowledge"]==args.background_knowledge) & \
            (template_df["history_signal_type"]==hist_type) \
            ]
    else:
        template_df = template_df[ \
            (template_df["prompt_type"]==args.prompt_type) & \
            (template_df["few_shot"]==args.few_shot) & \
            (template_df["background_knowledge"]==args.background_knowledge) & \
            (template_df["history_signal_type"]==hist_type) \
            ]

    # if bg knowledge, dataset should be considered
    if args.background_knowledge:
        template_df = template_df[template_df["dataset"]==args.dataset]
    assert len(template_df)==1, "Couldn't find a unique template for the given arguments"
    return template_df["value"].values[0]


def main():
    args = cmdline_args()
    print(args)
    arg_dict = vars(args)
    if args.wandb:
        wandb.init(project="frugal-prompts", config=arg_dict)

    prompt_template = search_template(args)

    dataset = load_dataset(
        dataset = args.dataset,
        prompt_type = args.prompt_type,
        few_shot = args.few_shot,
        background_knowledge = args.background_knowledge,
        history_signal_type = args.history_signal_type,
        history_k = args.history_k
    )

    # model = load_model(args.model)
    # responses = inference(model, dataset)

    # Generation Block
    if args.dataset == "MSC":
        total_instances = 16300
        input_dataset="MSC"
        data_path = "context_data/multi_session_chat/"
    elif args.dataset == "TC":
        total_instances = 22452
        input_dataset="TC"
        data_path = "context_data/topical_chat/"
    else:
        raise Exception("Invalid dataset")

    if args.history_signal_type == "recent-k":
        inputfile = os.path.join(data_path, "previous_utterances", f"last{args.history_k}.txt")
    elif args.history_signal_type == "semantic-k":
        inputfile = os.path.join(data_path, "semantically_similar_utterances", f"top{args.history_k}.txt")
    elif args.history_signal_type in ["full", "none"]:
        inputfile = os.path.join(data_path, "previous_utterances", "full.txt")
    elif args.history_signal_type == "peg":
        inputfile = os.path.join(data_path, "generated_summary", "test_summary_pegasusft_dialogdata.txt")
    elif args.history_signal_type == "bart":
        inputfile = os.path.join(data_path, "generated_summary", "test_summary_bartlarge_dialogdata.txt")
    elif args.history_signal_type == "peg_cd":
        inputfile = os.path.join(data_path, "generated_summary", "test_summary_pegasus_cnndailymail.txt")

    if args.background_knowledge:
        assert args.history_signal_type in ['bart', 'peg'], "This config is not part of the paper yet!"
        if args.dataset == "MSC":
            inputfile = os.path.join(data_path, "generated_summary", "test_pegasusft_summary_pegasuscnndm_persona_dialogdata.txt")
        elif args.dataset == "TC":
            inputfile = os.path.join(data_path, "generated_summary", "test_summary_and_knowledge_dialogdata.txt")

    if not os.path.exists(inputfile):
        raise Exception("Input file not found")
    else:
        print(f"Input file: {inputfile}")

    # TODO: Prompt template logic
    # To be read from a csv file, based on the arguments
    
    temp_dir = "./tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # UNIQ hash id for the run
    hash_id = uuid.uuid4().hex[:6].upper()
    temp_output = os.path.join(temp_dir, f"intermediate_file_{hash_id}.txt")
    # temp_output="intermediate_file.txt"
    print(f"Temp output file: {temp_output}")
    stdout = sys.stdout
    sys.stdout = open(temp_output, 'w')

    # helper(input_file,
    #        bs,
    #        outfile,
    #        prompt_template,
    #        precision_mode,
    #        use_shorter_template=False,
    #        has_persona_only=False,
    #        has_knowledge_only=False,
    #        has_persona_and_summary=False,
    #        has_knowledge_and_summary=False,
    #        current_utterance_only=False,
    #        model_path="facebook/blenderbot-3B",
    #        bart_summary=False,
    #        use_fsb_prompt=False,
    #        segment_utt=False,
    #        num_processes=8)

    # OUTPUT PATH
    path = f"outputs/{input_dataset}/"
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

    if not args.eval_mode:
        # Delay the import to avoid loading the model-libraries before the template is found (takes time)
        from generate_responses import helper, compute_preds
        has_persona_only = False
        has_knowledge_only = False
        has_persona_and_summary = False
        has_knowledge_and_summary = False
        if args.background_knowledge:
            has_persona_only = (args.history_signal_type == "none") and (args.dataset=="MSC")
            has_knowledge_only = (args.history_signal_type == "none") and (args.dataset=="TC")

            has_persona_and_summary = (args.history_signal_type != "none") and (args.dataset=="MSC")
            has_knowledge_and_summary = (args.history_signal_type != "none") and (args.dataset=="TC")
        
        # args.model to model_path
        model_path_translator = {
            'flan-t5': 'google/flan-t5-xl',
            'T0': 'bigscience/T0_3B',
            'tk-instruct': 'allenai/tk-instruct-3b-def',
            'dv3': "dv3"
        }
        model_path = model_path_translator[args.model]
 
        helper(inputfile,
            args.batch_size,
            temp_output,
            prompt_template=prompt_template,
            precision_mode="fp16",
            model_path=model_path,
            use_shorter_template=(not args.few_shot), 
            has_persona_only=has_persona_only,
            has_knowledge_only=has_knowledge_only,
            has_persona_and_summary=has_persona_and_summary,
            has_knowledge_and_summary=has_knowledge_and_summary,
            current_utterance_only=(args.history_signal_type == "none"),
            bart_summary=((args.history_signal_type == "bart") or (args.history_signal_type == "peg")),
            num_processes=args.num_gpus)

        sys.stdout = stdout

        filtered_preds=compute_preds(temp_output)
        
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)    
        print(f"Saving to {path}/{output_filename}")
        with open(f"{path}/{output_filename}", "w") as fp:
            for entry in tqdm(filtered_preds):
                fp.write(json.dumps(entry)+"\n")

        wandb.finish()

    else:
        # Evaluation

        import evaluate 
        import numpy as np
        from utils.deb.eval_deb import DEB
        from transformers import AutoTokenizer
        from bleurt import score as bleurt_scorer
        from utils.Evaluator import Evaluator


        my_eval = Evaluator()
        # TODO: Load deb model
        os.environ["DEB-PATH"] = "./utils/deb/data/deb_model/"
        os.environ["BLEURT-PATH"] = "./utils/bleurt/bleurt/BLEURT-20"


        # all_metrics = ["bleu", "meteor", "rouge", "bert", "deb", "bleurt", "length"]
        all_metrics = ["meteor", "deb", "bleurt", "length"]
        all_res, eval_instances, logs = my_eval.compute(os.path.join(path, output_filename), all_metrics)

        correction_factor = eval_instances / total_instances

        res = {}
        for metric in all_res:
            res[metric] = {}
            for submetric in all_res[metric]:
                res[metric][submetric] = np.mean(all_res[metric][submetric]) * correction_factor
        
        results_obj = {
                "BLEU": res["bleu"]["bleu"],
                "METEOR": res["meteor"]["meteor"],
                "ROUGE-1": res["rouge"]["rouge1"],
                "ROUGE-2": res["rouge"]["rouge2"],
                "ROUGE-L": res["rouge"]["rougeL"],
                "BERTScore-p": res["bert"]["precision"],
                "BERTScore-r": res["bert"]["recall"],
                "BERTScore-F1": res["bert"]["f1"],
                "DEB": res["deb"]["deb"],
                "BLEURT": res["bleurt"]["scores"],
                "prompt_len": res["length"]["prompt_length"] if "prompt_length" in res["length"] else -1,
                "output_len": res["length"]["response_length"]
            }
        
        print(results_obj)
        
        # Save results to file
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        eval_filename = output_filename.replace(".jsonl", ".eval.json")
        with open(f"{path}/{eval_filename}", "w") as fp:
            json.dump(results_obj, fp, indent=4)
        
        # Test: report some random results to wandb for now
        if args.wandb:        
            # BLEU	METEOR	ROUGE-1	ROUGE-2	ROUGE-L	BERTScore-p	BERTScore-r	BERTScore-F1	DEB	BLEURT	prompt_len output_len
            wandb.log(results_obj)
            wandb.finish()

if __name__ == "__main__":
    main()
