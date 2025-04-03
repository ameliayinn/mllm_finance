# trainer.py
import torch
from torch.utils.data import DataLoader
from data_processor import DataProcessor
from model_wrapper import QwenSummarizer
from config import Config

class Trainer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = QwenSummarizer()
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(), 
            lr=Config.learning_rate
        )
    
    def create_dataloader(self):
        """创建训练数据加载器"""
        data = self.data_processor.load_train_data()
        # 实现数据批处理逻辑
        # ...
        return DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    def train(self):
        """训练主循环"""
        dataloader = self.create_dataloader()
        self.model.model.train()
        
        for epoch in range(Config.num_epochs):
            for batch in dataloader:
                loss = self.model.train_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 保存检查点
            torch.save(self.model.state_dict(), 
                      f"{Config.output_dir}/checkpoint_epoch{epoch}.pt")