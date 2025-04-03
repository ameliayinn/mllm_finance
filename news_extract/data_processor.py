# data_processor.py
from transformers import AutoTokenizer
from config import Config
import json

class DataProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    def slide_window_tokenize(self, text, max_length=Config.max_seq_length, stride=Config.sliding_window_stride):
        """滑动窗口编码"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + max_length]
            if len(chunk) < max_length:
                chunk += [self.tokenizer.pad_token_id] * (max_length - len(chunk))
            chunks.append(chunk)
        return chunks
    
    def load_train_data(self):
        """加载训练数据"""
        with open(Config.train_data_path) as f:
            return [json.loads(line) for line in f]
    
    def process_corpus(self):
        """处理新闻语料"""
        with open(Config.corpus_path) as f:
            return [json.loads(line) for line in f]