import os
import json
from transformers import BertTokenizer
from src.preprocess_func import create_movies_dataset

def main(config_file="config/preprocessing.json"):

    with open(config_file) as f:
        config = json.load(f)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    create_movies_dataset(tokenizer, **config["general"])

if __name__ == "__main__":
    main()
