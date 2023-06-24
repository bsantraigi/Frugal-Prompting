import torch
from abc import abstractmethod, ABC

# TODO: Review the max_len argument and the trim technique in the _custom_tok method
class BaseDataset(ABC):
    """
    This optionally has a tokenizer for tokenization and faster access.
    It also has all the magic methods defined for sequential/random access
    Just need to override the _construct method which populates self.data
    """
    def __init__(self, tokenizer=None):
        self.data = []
        self.tokenizer = tokenizer

        if not self.tokenizer:
            print("WARNING: Tokenizer not instantiated, only raw text will be available, manual tokenization is required!")

    @abstractmethod
    def _construct(self, length):
        pass

    def normalize_reply(self,text):
        """
        Standardize the capitalization and punctuation spacing of the input text.

        Version 1: Fix sentence start casing, and punctuation.

        Version 2: Add trailing period, if missing.
        """

        switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]

        # add spaces so that words and punctuation can be seaprated
        new_text = text.lower()

        # normalize in case of human:
        for new, old in switch_list:
            new_text = new_text.replace(old, new).replace('  ', ' ')

        # split on punctuation to find sentence boundaries
        # capitalize stuff
        tokens = new_text.split(' ')
        for i in range(len(tokens)):
            if i == 0:
                tokens[i] = str.upper(tokens[i])
            elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
                tokens[i] = str.upper(tokens[i])
            elif tokens[i] in '?.!' and i < len(tokens) - 1:
                tokens[i + 1] = str.upper(tokens[i + 1])
        new_text = ' '.join(tokens)
        new_text = ' ' + new_text + ' '

        for tup in switch_list:
            new_text = new_text.replace(tup[0], tup[1])

        # get rid of surrounding whitespace
        new_text = new_text.strip()
        new_text = new_text.replace('  ', ' ')

        if new_text and new_text[-1] not in '!.?)"\'':
            new_text += '.'

        return new_text

    def _custom_tok(self, utterance, max_len=128):
        """
        Inputs:
        utterance: The actual text
        max_len: Maximum length of the utterance (left-trims the utterance)
        fact: Additional dependency for persona and topical chat

        Every utterance will be a dictionary consisting of 3 fields
        1. text: The actual utterance.
        2. input_ids: the tokens generated.
        3. attention_mask: Attention mask for the model.
        """

        if self.tokenizer:
            tokens = dict()
            tokens["text"] = utterance

            temp_tokens = self.tokenizer.encode(utterance, return_tensors="pt")

            tokens["input_ids"] = temp_tokens.clone()
            tokens["attention_mask"] = torch.ones(temp_tokens.shape)

            # NOTE: Just keep the most recent part of the utterance
            if tokens["input_ids"].shape[1] >= max_len:
                tokens["input_ids"] = tokens["input_ids"][:,-max_len:]
                tokens["attention_mask"] = tokens["attention_mask"][:,-max_len:]
        else:
            # Return plain text if tokenizer not supplied
            tokens = utterance

        return tokens
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert index < len(self) and index >= 0, "Index out of range!"
        return self.data[index]

    def __iter__(self):
        for data in self.data:
            yield data
