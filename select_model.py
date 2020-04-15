import os
import json
import torch
from transformers import BertForTokenClassification
from src.predict_func import plot_loss_accuracy


curr_path = os.path.dirname(__file__)
file_path = 'results/plot1.png'

def main(config_file="config/select_model.json"):

    with open(config_file) as f:
        config = json.load(f)

    plot_loss_accuracy(**config["general"], path=os.path.join(curr_path, file_path))

if __name__ == "__main__":
    main()
