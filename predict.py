# import os
# import torch
# from transformers import BertForTokenClassification
# from src.predict_func import plot_confusion_matrix, evaluate_model

# def main():
#     if not os.path.exists(os.path.join(curr_path, "results")):
#         try:
#             os.makedirs(os.path.join(curr_path,"results"))
#         except FileExistsError:
#             pass
        
#     #make sure id_2_punc matches the number of classes that is being evaluated.
#     evaluate_model(test_loader, model, id_2_punc={"0": "\"\"", "1": ".", "4": "?"}, path=os.path.join(curr_path, file_path))

# if __name__ == "__main__":

#     curr_path = os.path.dirname(__file__)
#     file_path = 'results/results4.png'
#     test_loader = torch.load("C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/test/cached_test.pt")
#     checkpoint_path = "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/checkpoints/checkpoint-0000/"
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     model = BertForTokenClassification.from_pretrained(checkpoint_path).to(DEVICE)

#     main()

import os
import json
import torch
from transformers import BertForTokenClassification
from src.predict_func import evaluate_model

def main(image_name, config_file="config/predict.json"):

    if not os.path.exists(os.path.join(curr_path, "results")):
        try:
            os.makedirs(os.path.join(curr_path,"results"))
        except FileExistsError:
            pass
    
    with open(config_file) as f:
        config = json.load(f)

    #make sure id_2_punc matches the number of classes that is being evaluated.
    evaluate_model(**config["general"], path=os.path.join(curr_path, "results", image_name))

if __name__ == "__main__":
    curr_path = os.path.dirname(__file__)
    main(image_name="results5.png")
