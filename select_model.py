import os
import json
from src.predict_func import get_train_loss_files, plot_loss_accuracy

def main(image_name, 
        data_path="/misc/labshare/datasets5/callme/experiments/punctuation_restoration/",
        save_model_eval_path = "results/experiment_4/",
        model_eval_image_name = "plot1.png",
        config_file="config/select_model.json"):

    with open(config_file) as f:
        config = json.load(f)

    train_loss, eval_loss, train_acc, eval_acc = get_train_loss_files(**config['general'])
    plot_loss_accuracy(train_loss=train_loss,
                    eval_loss=eval_loss,
                    train_acc=train_acc,
                    eval_acc=eval_acc,
                    path=os.path.join(data_path, save_model_eval_path),
                    image_name = model_eval_image_name)

if __name__ == "__main__":
    main(image_name="plot1.png")

#     {
#     "general":{
#         "train_loss": "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/out/checkpoint-41000/train_loss_41000.txt",
#         "eval_loss": "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/out/checkpoint-41000/eval_loss_41000.txt",
#         "train_acc": "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/out/checkpoint-41000/train_accuracy_41000.txt",
#         "eval_acc": "C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/out/checkpoint-41000/eval_accuracy_41000.txt"
#     }
# }
