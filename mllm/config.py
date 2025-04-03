# Configuration file for model training
config = {
    'bert_name': 'bert-base-uncased',
    'cond_len': 50,
    'pred_len': 10,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dropout': 0.1,
    'dim_feedforward': 512,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_checkpoint': True,
    'checkpoint_dir': './checkpoints',
    'csv_file': 'path/to/your/csv_file.csv',
}
