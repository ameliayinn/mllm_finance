# config.py
class Config:
    # 模型参数
    model_name = ""
    distilled_version = "r1"
    max_seq_length = 512
    sliding_window_stride = 256
    
    # 训练参数
    learning_rate = 3e-5
    batch_size = 32
    num_epochs = 10
    
    # 推理参数
    max_summary_length = 200
    min_summary_length = 50
    repetition_penalty = 2.0
    
    # 路径配置
    train_data_path = "data/train.jsonl"
    corpus_path = "data/sp500_news.jsonl"
    output_dir = "results/"