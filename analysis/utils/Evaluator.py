import os
import sys
import json
import time
import argparse
import traceback
import multiprocessing

import evaluate 
import numpy as np
from tqdm import tqdm
from deb.eval_deb import DEB
from ctc_score import DialogScorer
from transformers import AutoTokenizer
from bleurt import score as bleurt_scorer

class Evaluator:
    
    def __init__(self):
        # List function and the dependencies
        self.supported_metrics = {
            "bleu": (self.bleu_f, self.check_pred_gt),
            "meteor": (self.meteor_f, self.check_pred_gt),
            "rouge": (self.rouge_f, self.check_pred_gt),
            "bert": (self.bert_f, self.check_pred_gt),
            "bleurt": (self.bleurt_f, self.check_pred_gt),
            "ctc": (self.ctc_f, self.check_hist_pred),
            "deb": (self.deb_f, self.check_hist_pred),
            "length": (self.length_f, self.check_length)
        }

        self.bertscore = None
        self.deb = None
        self.e_bert = None
        self.e_roberta = None
        self.bleurt = None
        self.tokenizer = None

    def _parse(self, input_file):
        data = []
        with open(input_file, 'r') as f:
            cnt = 0
            lines = f.readlines()
            n_instances = len(lines)
            cntr = 0
            for line in lines:
                check = json.loads(line.strip())
                # For the ground truth file
                if 'predicted_response' not in check.keys():
                    if 'gold_response' in check.keys():
                        check['predicted_response'] = check['gold_response']

                prefices = ["Person1: ", "Person2: ", "Bot: ", "User: "]
                for pref in prefices:
                    if check["predicted_response"].strip().startswith(pref):
                        check["predicted_response"] = check["predicted_response"].strip()[len(pref):]
                        cntr += 1
                        break
                    elif check["predicted_response"].strip().startswith(pref[:-1]):
                        check["predicted_response"] = check["predicted_response"].strip()[len(pref[:-1]):]
                        cntr += 1
                        break


                if len(check["predicted_response"].strip()) == 0:
                    # ignore for now!
                    check["predicted_response"] = "NO RESPONSE"
                    cnt += 1
                else:
                    # Erroneous cases
                    if 'summary' not in check.keys():
                        check['summary'] = "NO SUMMARY"

                    data.append(check)

            print(f"Got rid of {cntr} prefices")
            print(f"Ignored {cnt} instances")

        return data, n_instances

    def _chunk(self, data, pieces):
        size = len(data) // pieces
        take = np.array([size] * pieces)
        left = len(data) % pieces
        if left:
            more = np.array([1] * (left) + [0] * (pieces - left))
            take = take + more

        idx = 0
        final_data = []
        for batch_size in take:
            final_data.append([data[idx + cur] for cur in range(batch_size)])
            idx += batch_size

        return final_data

    def check_pred_gt(self, data):
        required_keys = ["predicted_response", "gold_response"]
        for idx, ins in enumerate(data):
            for key in required_keys:
                if key not in ins:
                    return False, f"Key {key} not found in the data at {idx}"
        return True, "OK"

    def check_hist_pred(self, data):
        required_keys = ["predicted_response", "history"]
        message = "OK"
        
        for idx, ins in enumerate(data):
            if "current_utterance" not in ins:
                message = f"Key current_utterance not found in the data at {idx} please check if OK"
                
            for key in required_keys:
                if key not in ins:
                    return False, f"Key {key} not found in the data at {idx}"
        
        return True, message

    def check_length(self, data):
        required_keys = ["predicted_response"]
        message = "OK"
        for idx, ins in enumerate(data):
            for key in required_keys:
                if key not in ins:
                    return False, f"Key {key} not found in the data at {idx}"
            if ("prompts" not in ins) and ("prompt" not in ins):
                message = f"Key prompts not found in the data at {idx}, add them to the data"
        return True, message
    
    def bleu_f(self, data):
        bleu = evaluate.load("bleu")
        reqd_vals = ["bleu"]

        results = dict()
        for val in reqd_vals:
            results[val] = []

        for instance in tqdm(data, desc="bleu..."):
            total = 0
            # incorporate averages
            for max_order in range(4):
                ### Caution: total has to be a dict for multiple metrics
                for metric in reqd_vals:
                    computed_val = bleu.compute(predictions=[instance["predicted_response"]], references=[instance["gold_response"]], max_order=max_order + 1)
                    total += computed_val[metric]

            for metric in reqd_vals:
                results[metric].append(total / 4)

        print(np.mean(results["bleu"]) / 22484)
        return results

    def _compute_meteor(self, pid, data, results):
        meteor = evaluate.load("meteor")
        reqd_vals = ["meteor"]

        my_results = dict()
        for val in reqd_vals:
            my_results[val] = []

        if pid == 0:
            for instance in tqdm(data, desc="meteor..."):
                computed_val = meteor.compute(predictions=[instance["predicted_response"]], references=[instance["gold_response"]])
                for metric in reqd_vals:
                    my_results[metric].append(computed_val[metric])
        else:
            for instance in data:
                computed_val = meteor.compute(predictions=[instance["predicted_response"]], references=[instance["gold_response"]])
                for metric in reqd_vals:
                    my_results[metric].append(computed_val[metric])

        results[pid] = my_results
        
    def meteor_f(self, data):
        n_proc = 24
        if len(data) < 1000:
            n_proc = 6
        elif len(data) < 3000:
            n_proc = 12

        #n_proc = 1

        chunked_data = self._chunk(data, n_proc)

        manager = multiprocessing.Manager()
        results = manager.dict()
        fin_results = dict()

        reqd_vals = ["meteor"]

        for val in reqd_vals:
            fin_results[val] = []

        processes = []
        for i in range(n_proc):
            if len(chunked_data[i]) == 0:
                continue
            p = multiprocessing.Process(target=self._compute_meteor,args=(i, chunked_data[i], results))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

        for idx in results:
            for val in reqd_vals:
                fin_results[val].extend(results[idx][val])

        return fin_results


    def _compute_rouge(self, pid, data, results):
        rouge = evaluate.load("rouge")
        reqd_vals = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        my_res = dict()
        for val in reqd_vals:
            my_res[val] = []

        if pid == 0:
            for instance in tqdm(data, desc="Rouge..."):
                computed_val = rouge.compute(predictions=[instance["predicted_response"]], references=[instance["gold_response"]])
                for metric in reqd_vals:
                    my_res[metric].append(computed_val[metric])
        else:
            for instance in data:
                computed_val = rouge.compute(predictions=[instance["predicted_response"]], references=[instance["gold_response"]])
                for metric in reqd_vals:
                    my_res[metric].append(computed_val[metric])


        results[pid] = my_res


    def rouge_f(self, data):
        n_proc = 96
        if len(data) < 1000:
            n_proc = 32
        elif len(data) < 3000:
            n_proc = 64

        chunked_data = self._chunk(data, n_proc)

        manager = multiprocessing.Manager()
        results = manager.dict()
        fin_results = dict()

        reqd_vals = ["rouge1", "rouge2", "rougeL"]

        for val in reqd_vals:
            fin_results[val] = []

        processes = []
        for i in range(n_proc):
            if len(chunked_data[i]) == 0:
                continue
            p = multiprocessing.Process(target=self._compute_rouge,args=(i, chunked_data[i], results))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

        for idx in results:
            for val in reqd_vals:
                fin_results[val].extend(results[idx][val])

        return fin_results

    def _init_bert(self):
        self.bertscore = evaluate.load("bertscore")
        return self.bertscore

    def bert_f(self, data):
        if self.bertscore is None:
            self._init_bert()
        
        reqd_vals = ["precision", "recall", "f1"]
        results = dict()        
        for metric in reqd_vals:
            results[metric] = []
            
        batch_size = 64
        for i in tqdm(range(0, len(data), batch_size), desc="Bertscore..."):
            batch = data[i:i+batch_size]
            predictions = []
            references = []
            
            for instance in batch:
                predictions.append(instance["predicted_response"])
                references.append(instance["gold_response"])
            
            score = self.bertscore.compute(predictions=predictions, references=references, lang="en")
            for metric in reqd_vals:
                results[metric].extend(score[metric])

        # for instance in tqdm(data, desc="Bertscore..."):
        #     score = self.bertscore.compute(predictions=[instance["predicted_response"]], references=[instance["gold_response"]], lang="en")
        #     for metric in reqd_vals:
        #         results[metric].append(score[metric])

        return results

    def _init_bleurt(self):
        # self.bleurt = evaluate.load("bleurt", module_type="metric")
        self.bleurt = bleurt_scorer.BleurtScorer(os.getenv("BLEURT-PATH"))
                
    def bleurt_f(self, data):
        if self.bleurt is None:
            self._init_bleurt()

        results = {"scores" : []}
        batch_size = 32
        for i in tqdm(range(0, len(data), batch_size), desc="Bleurt..."):
            batch = data[i:i+batch_size]
            predictions = []
            references = []
            for instance in batch:
                predictions.append(instance["predicted_response"])
                references.append(instance["gold_response"])
            scores = self.bleurt.score(candidates=predictions, references=references)
            results["scores"].extend(list(scores))    
        
        return results

    def _init_e_bert(self, align):
        self.e_bert = DialogScorer(align=align)

    def _init_e_roberta(self, align):
        self.e_roberta = DialogScorer(align=align)

    def ctc_f(self, data):
        
        if self.e_bert is None:
            self._init_e_bert("E-bert")

        if self.e_roberta is None:
            self._init_e_roberta("E-roberta")

        aligns = {"E-bert":self.e_bert, "E-roberta":self.e_roberta}
        aspects = ["engagingness"]
        results = dict()

        for al in aligns.keys():
            for asp in aspects:
                results[f"{al}_{asp}"] = []

        for align in aligns:
            scorer = aligns[align]

            for instance in tqdm(data, desc=f"Ctc {align}..."):
                dialog_history = instance["history"]
                if "current_utterance" in instance.keys():
                    dialog_history = dialog_history + "\n" + instance["current_utterance"]

                for aspect in aspects:
                    cur_key = f"{align}_{aspect}"
                    score = scorer.score(fact="NO FACT", dialog_history=dialog_history, hypo=instance["predicted_response"], aspect=aspect)
                    results[cur_key].append(score)

        return results
    
    def _init_deb(self):
        self.deb = DEB(deb_ckpt_basepath=os.getenv("DEB-PATH"))

    def deb_f(self, data):
        if self.deb is None:
            self._init_deb()
        
        print("DEB...")
        results = {"deb":[]}
        batch_size = 32
        
        for i in tqdm(range(0, len(data), batch_size), desc="DEB..."):
            batch = data[i:i+batch_size]
            dialog_history = []
            predicted_response = []
            
            for instance in batch:
                dialog_history.append(instance["history"])
                if "current_utterance" in instance.keys():
                    dialog_history[-1] = dialog_history[-1] + "\n" + instance["current_utterance"]
                predicted_response.append(instance["predicted_response"])
            label, prob = self.deb.evaluate(dialog_history, predicted_response)
            results["deb"].extend(prob)
            
        return results

    def _init_length(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    def length_f(self, data):
        # counts no of tokens in response and prompt
        if self.tokenizer is None:
            self._init_length()

        prompt = False
        if len(data) != 0 and "prompts" in data[0].keys():
            prompt = True
            results = {"response_length":[], "prompt_length":[]}
        else:
            results = {"response_length":[]}
            
        for ins in tqdm(data, desc='length...'):
            results["response_length"].append(len(self.tokenizer(ins['predicted_response'])['input_ids']))
            if prompt:
                results["prompt_length"].append(len(self.tokenizer(ins['prompts'])['input_ids']))

        return results

    def get_key(self, my_dict):
        return my_dict["current_utterance"].strip() + "\n" + my_dict["gold_response"].strip()

    def fillin(self, gold_data, data):
        cur_data = []
        offset = 0
        
        cnt = 0
        histories = dict()
        for ins in gold_data:
            key = self.get_key(ins)
            histories[key] = ins["history"]

        for ins in data:
            key = self.get_key(ins)
            if key in histories:
                cur_data.append({**ins, "history": histories[key]})
            else:
                if cnt % 500 == 0:
                    print(ins["current_utterance"])
                cnt += 1
                # print("Oh nonono")
        
        print(f"Extra {cnt}")
        return cur_data

    # This is the entrypoint
    def compute(self, input_file, metrics=None):
        print("Parsing file...")
        data, length = self._parse(input_file)

        if "history" not in data[0]:
            print("Gotta fillin!")
            if "multi_session" in input_file:
                verification_file = "msc_previous_utterances.txt"
            elif "topical_chat" in input_file:
                verification_file = "tc_previous_utterances.txt"
            else:
                raise Exception("Unknown file")

            gold_data, gold_length = self._parse(verification_file)

            data = self.fillin(gold_data, data)

        required_metrics = []
        logs = []
        
        print("Basic checks...")
        if metrics is None:
            metrics = self.supported_metrics.keys()
            
        for metric in metrics:
            if not metric in self.supported_metrics.keys():
                raise Exception(f"Metric {metric} not supported yet")
            else:
                status, message = self.supported_metrics[metric][1](data)
                if "OK" not in message:
                    logs.append((f"{input_file.split('/')[-1]}_{metric}", message))
                    print(message)
                if status:
                    required_metrics.append(metric)

        print("Computing metrics...")
        res_data = dict()
        for metric in required_metrics:
            res_data[metric] = self.supported_metrics[metric][0](data)

        return res_data, length, logs

def cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", type=str, choices=["cpu", "gpu"], required=True, help="CPU/GPU based metrics")

    return (parser.parse_args())

if __name__ == "__main__":
    args = cmdline_args()

    my_eval = Evaluator()
    # TODO: Load deb model
    os.environ["DEB-PATH"] = "./deb/data/deb_model/"
    os.environ["BLEURT-PATH"] = "./bleurt/bleurt/BLEURT-20"

    # TODO: change base path
    base_path = "evaluate_final"
    # datasets = [("multi_session_chat_gt", 16300), ("topical_chat_gt", 1078), ("topical_chat_ss", 1078), ("topical_chat_prev", 1078), ("topical_chat", 1078)]

    datasets = [("test", 121212),   # to load all models on  gpu
                ("multi_session_chat_prev", 16300), 
                ("multi_session_chat_ss", 16300),
                ("multi_session_chat_summary", 16300),
                ("multi_session_chat_cur", 16300),
                # ("multi_session_chat_new_prompt", 16300),
                ("topical_chat_prev", 22452),
                ("topical_chat_ss", 22452),
                ("topical_chat_summary", 22452),
                ("topical_chat_cur", 22452)
                # ("topical_chat_gt", 22452)
                ]
    
    # TODO:remove after evaluation
    # datasets = [("multi_session_chat_pegasus_ft", 16300), ("topical_chat_summary", 22452), ("multi_session_chat_left", 16300)]
    # datasets = [("topical_chat_cur", 22452), ("topical_chat_summary", 22452), ("topical_chat_prev", 22452)]
    datasets = [("topical_chat_all", 22452), ("multi_session_chat_all", 16300)]
    
    # TODO: Uncomment for testing
    # datasets = [("msc", 16300), ("tc", 1078)]

    EXT = ".txt"
    all_logs = []
    for (dataset, total_instances) in datasets:
        dir_path = os.path.join(base_path, dataset)
        files = list(filter(lambda f: f.endswith(EXT), os.listdir(dir_path)))

        eval_path = os.path.join(base_path, dataset + "_eval")
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)

        for input_file in files:
            try:
                print(f"Processing file: {input_file.split('/')[-1]}", file=sys.stderr)

                if "gpu" in args.compute:
                    # res, eval_instances, logs = my_eval.compute(os.path.join(dir_path, input_file), metrics=["bert", "ctc", "deb", "length"])
                    # res, eval_instances, logs = my_eval.compute(os.path.join(dir_path, input_file), metrics=["deb", "bleurt", "length"])
                    # res, eval_instances, logs = my_eval.compute(os.path.join(dir_path, input_file), metrics=["bleu", "meteor", "rouge", "bert", "deb"])
                    res, eval_instances, logs = my_eval.compute(os.path.join(dir_path, input_file), metrics=["bleurt", "length"])
                elif "cpu" in args.compute:
                    res, eval_instances, logs = my_eval.compute(os.path.join(dir_path, input_file), metrics=["bleu", "meteor", "rouge", "bert", "deb"])
                else:
                    raise Exception("Compute method unknown, must be cpu or gpu")
                    
                all_logs.extend(logs)
                # File with scores to all individual responses
                output_file = os.path.join(eval_path, input_file.split('/')[-1][:-len(EXT)] + f"_{args.compute}.json")
                with open(output_file, 'w') as f:
                    json.dump(res, f, indent=4)

                # Average scores
                new_file = output_file.split('/')[-1][:-5] + "_reduced.json"

                # csv for pasting into sheets
                dump_file = output_file.split('/')[-1][:-5] + "_reduced.csv"
                
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                metrics = []
                values = []

                new_data = dict()
                for metric in data:
                    new_data[metric] = {}
                    for submetric in data[metric]:
                        new_data[metric][submetric] = np.mean(data[metric][submetric]) * (eval_instances / total_instances)
                        metrics.append(submetric)
                        values.append(str(np.mean(new_data[metric][submetric])))

                with open(os.path.join(eval_path, new_file), 'w') as f:
                    json.dump(new_data, f, indent=4)

                with open(os.path.join(eval_path, dump_file), 'w') as f:
                    f.write(f"{','.join(metrics)}\n")
                    f.write(f"{','.join(values)}\n")

                    print(dump_file)
                    print(f"{','.join(metrics)}")
                    print(f"{','.join(values)}")

            except Exception as e:
                # traceback.print_exc()
                print(f"Exception occured {e}")
    
    print("***"*15)
    print("All logs")
    for metric, message in all_logs:
        print("***"*15) 
        print(metric)
        print(message)
