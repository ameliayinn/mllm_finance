# summarizer.py
from model_wrapper import QwenSummarizer
from data_processor import DataProcessor
from config import Config
import json

class HierarchicalSummarizer:
    def __init__(self, model_path):
        self.model = QwenSummarizer(model_path)
        self.data_processor = DataProcessor()
    
    def chunk_summarize(self, text):
        """分块摘要生成"""
        chunks = self.data_processor.slide_window_tokenize(text)
        partial_summaries = []
        for chunk in chunks:
            chunk_text = self.model.tokenizer.decode(chunk, skip_special_tokens=True)
            summary = self.model.generate_summary(
                chunk_text,
                max_length=self._calculate_partial_length(len(chunks))
            )
            partial_summaries.append(summary)
        return " ".join(partial_summaries)
    
    def _calculate_partial_length(self, num_chunks):
        """计算部分摘要长度"""
        base_length = int(1.5 * Config.max_summary_length / num_chunks)
        return min(base_length, Config.max_seq_length)
    
    def summarize_corpus(self):
        """处理整个语料库"""
        corpus = self.data_processor.process_corpus()
        results = []
        for doc in corpus:
            # 层次化摘要
            combined = self.chunk_summarize(doc["text"])
            final_summary = self.model.generate_summary(combined)
            results.append({
                "id": doc["id"],
                "original": doc["text"],
                "summary": final_summary
            })
        return results