# model_wrapper.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import Config

class QwenSummarizer:
    def __init__(self, model_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.model_name if not model_path else model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            Config.model_name if not model_path else model_path)
        
    def train_step(self, batch):
        """单次训练步骤"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"]
        }
        outputs = self.model(**inputs)
        return outputs.loss
    
    def generate_summary(self, text, max_length=Config.max_summary_length):
        """生成摘要"""
        inputs = self.tokenizer(
            text, 
            truncation=True, 
            max_length=Config.max_seq_length,
            return_tensors="pt"
        )
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            repetition_penalty=Config.repetition_penalty,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)