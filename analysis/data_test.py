from utils.GroundTruthDataset import GroundTruthMultiSessionChat, GroundTruthTopical
from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dataset = GroundTruthMultiSessionChat("data/multi_session_chat/msc_dialogue/session_2/test.txt", tokenizer, 4)
    print(dataset[0])