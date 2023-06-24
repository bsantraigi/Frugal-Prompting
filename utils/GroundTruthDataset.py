import json
import os
import re

import torch
import numpy as np
from transformers import BertTokenizer
from abc import abstractmethod, ABC
import copy


from Dataset import BaseDataset


class GroundTruthPersona(BaseDataset):
    """
    General rules:
    1. Read the file (may be different txt, json, etc)
    2. Instantiate the parent class with tokenizer
    3. Define the construct class to parse and create the standard dataset

    Format expected for self.data: List of objects where
    Each object is a dict with three keys
    1. "history": The conversation history
    2. "current_utterance": The current turn to which response is reqd
    3. "response": The response for the utterance
    """

    def __init__(self, filename, tokenizer=None, length=None):
        with open(filename, 'r') as f:
            temp_data = f.readlines()

        super().__init__(tokenizer)
        self.data = self._construct(temp_data, length)

    # override
    def _construct(self, data, length=None):
        res_data = []
        history = []
        context = None
        response = None

        for example in data:
            if example[0] == "1" and int(example.split(" ")[0]) == 1 and history:
                base_obj = {}
                base_obj["history"] = self._custom_tok("\n".join(history))
                base_obj["current_utterance"] = self._custom_tok(context)
                base_obj["response"] = self._custom_tok(response)
                res_data.append(base_obj)
                history = []
                context = None
                response = None

            elif context:
                history.append(context)
                history.append(response)

            context, response = list(map(lambda x: self.normalize_reply(x), example.strip().split("\t")[:2]))
            # Get rid of the initial serial number
            context = " ".join(context.split(" ")[1:])

        # Last object
        if history:
            base_obj = {}
            base_obj["history"] = self._custom_tok(self.normalize_reply("\n".join(history)))
            base_obj["current_utterance"] = self._custom_tok(self.normalize_reply(context))
            base_obj["response"] = self._custom_tok(self.normalize_reply(response))
            res_data.append(base_obj)

        return res_data


class GroundTruthTopical(BaseDataset):
    """
    General rules:
    1. Read the file (may be different txt, json, etc)
    2. Instantiate the parent class with tokenizer
    3. Define the construct class to parse and create the standard dataset

    Format expected for self.data: List of objects where
    Each object is a dict with two keys
    1. "history": The conversation history
    2. "current_utterance": The current turn to which response is reqd
    2. "response": The response for the utterance
    """

    def __init__(self, filename, tokenizer=None, length=None):
        with open(filename, 'r') as f:
            temp_data = json.load(f)

        super().__init__(tokenizer)
        self.data = self._construct(temp_data, length)

    # override
    def _construct(self, data, length=None):
        # history for the first utterance
        DUMMY_TEXT = "__SILENCE__"
        res_data = []

        if not length:
            length = len(data)
        else:
            length = min(length, len(data))

        data = data.values()

        for example in data:
            # Normalize reply before appending
            utterances = [self.normalize_reply(instance["message"]) for instance in example["content"]]
            utterances = list(filter(lambda x: len(x.strip()) > 0, utterances))

            cur_history = [utterances[0]]
            for i in range(1, len(utterances)):
                base_obj = {}
                if len(cur_history) <= 1:
                    base_obj['history'] = DUMMY_TEXT
                else:
                    base_obj['history'] = "\n".join(cur_history)

                base_obj['current_utterance'] = utterances[i - 1]
                base_obj['response'] = utterances[i]
                cur_history.append(utterances[i])
                res_data.append(base_obj)

        return res_data


DUMMY_TEXT = "__SILENCE__"
"""
    As per the paper the sessions may span across several hours/days. To demarcate between the time gap we use __SILENCE_ token
    This implementation is picked from the the parlai repo
"""
max_length = 1024


class GroundTruthMultiSessionChat(BaseDataset):
    """
    General rules:
    1. Read the file (may be different txt, json, etc)
    2. Instantiate the parent class with tokenizer
    3. Define the construct class to parse and create the standard dataset

    Format expected for self.data: List of objects where
    Each object is a dict with three keys
    1. "history": The conversation history i.e. can be previous/current persona, raw conversation history, gold summary or predicted summary
    2. "current_utterance": The current turn to which response is read
    2. "response": The response for the utterance
    """

    def __init__(self, filename, version, tokenizer=None, length=None):
        with open(filename, "r") as fp:
            raw_data = [json.loads(line.strip()) for line in fp]

        super().__init__(tokenizer)

        self.version = version
        '''
            version is an important parameter since it decides whether or not any previous session context will be append to 
            the current conversation. If version=1 , then we do not have conversation history, otherwise we have histories
            aggregated across sessions for sessions 1/2/3
        '''
        self.persona_type = "current"
        '''
            persona changes across sessions https://arxiv.org/pdf/2107.07567.pdf. We can either keep previous persona or the current . 
            Choices are from ["previous","current"]
        '''
        self.previous_context = "raw_history"
        '''
            As per the paper and the implementation https://arxiv.org/pdf/2107.07567.pdf, we can have four types of contexts:
            1. Persona information
            2. Raw conversation history
            3. Gold summaries
            4. Summaries generated by a summary model
            We can choose from ["persona","raw_history","gold_summary"]
            Since we are generating summaries ourselves we are not using the 4th option
            We have also not implemented option 3 for the time being
        '''
        self.data = self._construct(raw_data, length)

    # override
    def compile_persona_dialog_input(self, dialog, personas, previous_dialogs):
        new_dialog = copy.deepcopy(dialog)
        new_previous_dialogs = copy.deepcopy(previous_dialogs)
        your_persona = ""
        partner_persona = ""
        your_persona = '\n'.join([f'your persona: {x}' for x in personas[0]])
        partner_persona = '\n'.join([f"partner's persona: {x}" for x in personas[1]])
        for prev_dialog in new_previous_dialogs:
            prev_dialog['dialog'].insert(0, {"text": DUMMY_TEXT})
            if len(prev_dialog['dialog']) % 2 == 1:
                prev_dialog['dialog'].append({"text": DUMMY_TEXT})
            new_dialog.insert(0, {"text": DUMMY_TEXT})
        return your_persona, partner_persona, new_dialog, new_previous_dialogs

    def normalize_replies(self, x):
        xs = [xt.strip() for xt in x.split('\n')]
        xs2 = []
        for x in xs:
            if 'your persona:' in x:
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: '):]
                x = self.normalize_reply(x)
                x = 'your persona: ' + x
            elif "partner's persona: " in x:
                x = x[len("partner's persona: "):]
                x = self.normalize_reply(x)
                x = "partner's persona: " + x
            elif x != DUMMY_TEXT:
                x = self.normalize_reply(x)
            xs2.append(x)
        return "\n".join(xs2)

    def _construct(self, data, length=None):
        res_data = []
        if not length:
            length = len(data)
        else:
            length = min(length, len(data))

        for dialog_dict in data:
            if self.persona_type == "current":
                personas = dialog_dict["personas"]
            else:
                personas = dialog_dict["init_personas"]
            (your_persona, partner_persona, new_dialog, new_previous_dialogs) = self.compile_persona_dialog_input(
                dialog_dict['dialog'],
                personas,
                dialog_dict['previous_dialogs'])
            previous_sessions_msgs = []
            if self.previous_context == 'raw_history':
                for d_id in range(len(new_previous_dialogs)):
                    previous_dialog_msg = [
                        x['text'] for x in new_previous_dialogs[d_id]['dialog']
                    ]

                    previous_sessions_msgs.append(
                        '\n'.join(previous_dialog_msg)
                    )

            episodes = []
            for i in range(0, len(new_dialog) - 1, 2):
                text = new_dialog[i]['text']
                action = {
                    'text': self.normalize_replies(text),
                    'labels': [self.normalize_replies(new_dialog[i + 1]['text'])]
                }

                episodes.append(action)

            if self.version == 1:

                utterances = []
                for episode in episodes:
                    if episode["text"] != DUMMY_TEXT:
                        base_obj = {}
                        base_obj['history'] = "\n".join(utterances) if len(utterances) >= 1 else DUMMY_TEXT
                        base_obj['current_utterance'] = episode["text"]
                        base_obj['response'] = episode["labels"]
                        res_data.append(base_obj)
                        if len(episode["text"]) > 0:
                            utterances.append(episode["text"])
                break

            persona_context_str = ""
            if self.previous_context == "persona":
                previous_context_str = ((partner_persona + '\n') if len(partner_persona) > 0 else "") + your_persona
            elif self.previous_context == 'raw_history':
                persona_context_str = previous_sessions_msgs

            if persona_context_str and len(persona_context_str) > 0:

                utterances = persona_context_str[0].split("\n")
                # print(len(utterances))
                for episode in episodes:
                    if episode["text"] != DUMMY_TEXT:
                        base_obj = {}
                        base_obj['history'] = "\n".join(utterances)
                        base_obj['current_utterance'] = episode["text"]
                        base_obj['response'] = episode["labels"]
                        res_data.append(base_obj)
                        if len(episode["text"]) > 0:
                            utterances.append(episode["text"])
                        # since we already have a long history we keep a fixed length history by continuously popping the first element
                        utterances.pop(0)

        return res_data


if __name__ == "__main__":
    my_dataset = GroundTruthTopical(os.path.join(os.getcwd(), "data/topical_chat/topical_chat_test_rare.json"))
    print(len(my_dataset))
    my_dataset2 = GroundTruthPersona(os.path.join(os.getcwd(), "data/persona_chat/persona_chat_test.txt"))
    print(len(my_dataset2))
    my_dataset3=GroundTruthMultiSessionChat(os.path.join(os.getcwd(),"data/multi_session_chat/msc_dialogue/session_2/train.txt"), version=0)
    print(len(my_dataset3))

    # import ipdb
    # ipdb.set_trace()
