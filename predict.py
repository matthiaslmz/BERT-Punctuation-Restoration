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

def main(config_file="config/predict.json",
        data_path="/misc/labshare/datasets5/callme/experiments/punctuation_restoration/",
        save_model_eval_path = "results/experiment_4/",
        conf_mat_image_name= "confusion_mat1.png"):
    
    with open(config_file) as f:
        config = json.load(f)

    #make sure id_2_punc matches the number of classes that is being evaluated.
    evaluate_model(**config['general'],
                path=os.path.join(data_path, save_model_eval_path),
                image_name = conf_mat_image_name)

if __name__ == "__main__":
    main()


# {
#     "general":{
#         "test_loader": "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/test/cached_test.pt",
#         "model":"C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/checkpoints/checkpoint-0000/",
#         "id_2_punc": {"0": "\"\"", "1": ".", "4": "?"}
#     }
# }
