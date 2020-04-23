import os
import json
from transformers import BertTokenizer
from src.preprocess_func import create_movies_dataset, read_files_in_dir_ext
from src.pipeline import BERTPuncResto
from src.predict_func import get_train_loss_files, plot_loss_accuracy, evaluate_model

def main(config_file="config/pipeline.json", 
        data_path="/misc/labshare/datasets5/callme/experiments/punctuation_restoration/",
        save_model_eval_path = "results/experiment_4/",
        model_eval_image_name = "plot1.png",
        conf_mat_image_name= "confusion_mat1.png"):

    with open(config_file) as f:
        config = json.load(f)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dset, eval_dset, test_dset = create_movies_dataset(tokenizer, **config["create_movies"])

    #Train model
    model = BERTPuncResto(**config["general"],
                        cached_train_path_or_dataset=train_dset,
                        cached_eval_path_or_dataset=eval_dset)
    model.train_model(**config["training"])

    # Select model 
    output_path = config["training"]['output_path'][:-15]
    train_loss, eval_loss, train_acc, eval_acc = get_train_loss_files(output_path, "txt")

    best_model_output = plot_loss_accuracy(train_loss=train_loss,
                                            eval_loss=eval_loss,
                                            train_acc=train_acc,
                                            eval_acc=eval_acc,
                                            path=os.path.join(data_path, save_model_eval_path),
                                            image_name = model_eval_image_name)

    # Select model OR show_accuracy_loss(**config["select_model"], path=os.path.join(data_path,"results/experiment_6"), image_name=image_name)
    checkpoint_path = config["training"]['ckpt_path'][:-15]
    model_checkpoint_path = os.path.join(checkpoint_path, best_model_output)

    # Make sure that the number of classes in id_2_punc is same as the # of classes being evaluated from the test_loader
    evaluate_model(**config['predict'],
                    model=model_checkpoint_path,
                    path=os.path.join(data_path, save_model_eval_path),
                    image_name = conf_mat_image_name)

if __name__ == "__main__":
    main()
