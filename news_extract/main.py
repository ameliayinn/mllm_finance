# main.py
from trainer import Trainer
from summarizer import HierarchicalSummarizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--model_path", help="Path to trained model")
    args = parser.parse_args()
    
    if args.mode == "train":
        trainer = Trainer()
        trainer.train()
    else:
        if not args.model_path:
            raise ValueError("Model path required for inference")
        summarizer = HierarchicalSummarizer(args.model_path)
        results = summarizer.summarize_corpus()
        
        # 保存结果
        with open("results/summaries.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()