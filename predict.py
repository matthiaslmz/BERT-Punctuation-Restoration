import os
import torch
from transformers import BertForTokenClassification
from src.predict_func import plot_confusion_matrix, evaluate_model

curr_path = os.path.dirname(__file__)
file_path = 'results/results4.png'
test_loader = torch.load("../ghdata/BERTPunctuationRestoration/data/experiment4/cached_test.pt")
checkpoint_path = '../ghdata/BERTPunctuationRestoration/data/experiment4/checkpoints/checkpoint-15000'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = BertForTokenClassification.from_pretrained(checkpoint_path).to(DEVICE)

def main():

    #make sure id_2_punc matches the number of classes that is being evaluated.
    evaluate_model(test_loader, model, id_2_punc={"0": "\"\"", "1": ".", "4": "?"}, path=os.path.join(curr_path, file_path))

if __name__ == "__main__":
    main()
