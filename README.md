# Punctuation Restoration with BERT

The aim of this experiment is to see how BERT performs on punctuation restoration. The purpose of this mini-experiment is to see how well the pre-trained (on English Wiki and BookCorpus with MLM and NSP as pre-training tasks) BERT performs on conversational data hence why I will be using movie transcripts.

### Dataset
We will be using Download the dataset @ [Cornell Movie Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

### Setup & Installation

Clone the repo:

    git clone https://github.com/matthiaslmz/BERT-Punctuation-Restoration.git

Next, create a new conda environment:
    conda env create -f enviroment.yml
    conda activate whatisbert

Make sure to download `bert-base-uncased` model, vocab and config file @:

1. [pytorch_model](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin)
2. [vocab](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt)
3. [config](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json)

### CUDA
1. If an error occurs, first make sure that you have a GPU that is available.
2. If error persists, it could be that the defined training batch size is too large for the GPU, decrease if necessary
3. By default in `pipeline.py`, if a GPU is available, `torch.cuda.is_available()` defaults to the first GPU, indexed `0`. If you need to use the $X^{th}$ GPU, make sure to edit like the following: `DEVICE = "cuda:X-1" if torch.cuda.is_available() else cpu`

