import spacy
import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset

TOTAL_DATA_NUM = 5000

spacy_zh = spacy.load('zh_core_web_sm')
spacy_en = spacy.load('en_core_web_sm')

class WordDataset(Dataset):
    def __init__(self, src_file, trg_file): 
        # dataset = WordDataset(src_file, trg_file)
        with open(src_file, encoding='utf-8') as f:
            self.src_lines = f.readlines()
        with open(trg_file, encoding='utf-8') as f:
            self.trg_lines = f.readlines()

        self.src_lines = self.src_lines[:TOTAL_DATA_NUM]
        self.trg_lines = self.trg_lines[:TOTAL_DATA_NUM]

        self.tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
        self.tokenizer_zh = get_tokenizer('spacy', language='zh_core_web_sm')
    
    def __len__(self): # len(dataset)
        return len(self.src_lines)
    
    def __getitem__(self, idx): # dataset[i]
        src_line = self.src_lines[idx]
        trg_line = self.trg_lines[idx]

        src_tokens = [tok.text for tok in spacy_zh(src_line)]
        trg_tokens = [tok.text for tok in spacy_en(trg_line)]

        return src_tokens, trg_tokens # word dataset[idx]
    
class NumberDataset(Dataset):
    def __init__(self, src_file, trg_file, src_vocab, trg_vocab, max_len):
        with open(src_file, encoding='utf-8') as f:
            self.src_lines = f.readlines()
        with open(trg_file, encoding='utf-8') as f:
            self.trg_lines = f.readlines()

        self.src_lines = self.src_lines[:TOTAL_DATA_NUM]
        self.trg_lines = self.trg_lines[:TOTAL_DATA_NUM]

        self.tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
        self.tokenizer_zh = get_tokenizer('spacy', language='zh_core_web_sm')

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)
    
    def __getitem__(self, idx):
        src_line = self.src_lines[idx]
        trg_line = self.trg_lines[idx]

        src_tokens = [tok.text for tok in spacy_zh(src_line)]
        trg_tokens = [tok.text for tok in spacy_en(trg_line)]

        src_nums = []
        trg_nums = []

        for word in src_tokens:
            try:
                src_nums.append(self.src_vocab[word]) # is number not word
            except KeyError:
                pass
        
        src_nums.insert(0, 1) # <sos> = 1
        src_nums.append(2) # <eos> = 2

        src_res = [0] * self.max_len # <pad> = 0
        src_res[:len(src_nums)] = src_nums # 0~len(src_nums)
        src_ret = torch.tensor(src_res)

        for word in trg_tokens:
            try:
                trg_nums.append(self.trg_vocab[word])
            except KeyError:
                pass

        trg_nums.insert(0, 1)
        trg_nums.append(2)
        
        trg_res = [0] * self.max_len
        trg_res[:len(trg_nums)] = trg_nums
        trg_ret = torch.tensor(trg_res)

        return src_ret, trg_ret # all number dataset[idx]