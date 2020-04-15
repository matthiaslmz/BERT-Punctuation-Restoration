from transformers import BertTokenizer
from src.preprocess_func import create_movies_dataset
import os

def main():

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    punc2id = {"": 0, ".": 1, ",": 2, "$": 3, "?": 4}

    create_movies_dataset(tokenizer,
                     kept_punctuation=".,?",
                     punc_2_id=punc2id,
                     raw_dataset_path="C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/cornell_movie-dialogs_corpus/",
                     preprocessed_transcription_path="C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/",
                     cached_path="C:/Users/MatthiasL/Desktop/DATA/ghdata/BERTPunctuationRestoration/data/experiment4/%s/"
    )

if __name__ == "__main__":
    main()
