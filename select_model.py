import os
import json
from src.predict_func import plot_loss_accuracy

def main(image_name, config_file="config/select_model.json"):

    with open(config_file) as f:
        config = json.load(f)

    if not os.path.exists(os.path.join(curr_path, "results")):
        try:
            os.makedirs(os.path.join(curr_path,"results"))
        except FileExistsError:
            pass

    plot_loss_accuracy(**config["general"], path=os.path.join(curr_path, "results", image_name))

if __name__ == "__main__":

    curr_path = os.path.dirname(__file__)
    main(image_name="plot1.png")
