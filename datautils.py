
from transformers import AutoTokenizer,BlenderbotTokenizerFast

from torch.utils.data.dataset import IterableDataset,Dataset
        
class GenericDataset(Dataset):

    def __init__(self, model_name_or_path: str, examples,max_seq_length=512):
        super(GenericDataset, self).__init__()
        
        if model_name_or_path=="facebook/blenderbot-3B":
            self.tokenizer = BlenderbotTokenizerFast.from_pretrained(model_name_or_path)
            self.tokenizer.add_prefix_space=False
        else:
            self.tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token=self.tokenizer.eos_token
        
        self.examples=examples
        self.max_seq_length=max_seq_length
    def __getitem__(self,idx):
        return self.parse_data(self.examples[idx])
    '''def __iter__(self):
        for example in self.examples:
            try:
                line=self.parse_data(example)
                yield line
            except Exception as e:  # noqa
                traceback.print_exc()'''
            
                

    
    def __len__(self):
        return len(self.examples)
    
    def parse_data(self,example):
        if type(example)==dict:
            input_sentence=example["prompt"]
            input_ids = self.tokenizer([input_sentence], return_tensors='pt',padding="max_length",truncation=True,max_length=self.max_seq_length)
            instance={}
            instance["text"]=input_sentence 
            instance["gold_response"]=example["gold_response"]
            instance["history"]=example["history"]
            instance["current_utterance"]=example["current_utterance"]
            
            instance["summary"]=example["summary"] if "summary" in example else ""
            instance["id"]=example["id"]
            
        else:
            input_sentence=example
            input_ids = self.tokenizer([input_sentence], return_tensors='pt',padding="max_length",truncation=True,max_length=self.max_seq_length)
            instance={}
            instance["text"]=example    
        
        instance["input_ids"]=input_ids["input_ids"]
        instance["attention_mask"]=input_ids["attention_mask"]
        return instance