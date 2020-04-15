import ast
import difflib
import codecs
import json
import os
import re
import string
import torch
import itertools
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader

punc_2_id = {"": 0, ".": 1, ",": 2, "$": 3, "?": 4}

def read_files_in_dir_ext(dir_route, extension):
    """
    Retrieves all files in a dir with a specific extension
    :param dir_route: (str) the path where to look files with the specific extension
    :param extension: (str) the extension to look for. E.g.: ".txt".
    :return files_ext: (list) A list of files that contains all files with said extension.
    """
    files = os.listdir(dir_route)
    files_ext = [file for file in files if file.endswith(extension)]

    return files_ext


def read_file_in_string(file_name):
    """
    Reads the contents of a file and save them into a string.
    :param file_name: (str) the file/path to read.
    :return file_in_string: (str) the contents of the file in string form.
    """
    file_in_string = ""

    with codecs.open(file_name, "r", "utf-8") as f:
        file_in_string = f.read()
        f.close()

    return file_in_string


def save_string_in_file(string_text, file_name, mode="w"):
    """
    Saves the contents of a string into a file
    :param string_text: (str) the string to save
    :param file_name: (str) the file/path to save the string
    :param mode: (str) the writing mode for the file ("r", "r+", "w", "w+", "a", "a+")
           'r'   Open text file for reading.  The stream is positioned at the
                     beginning of the file.
           'r+'  Open for reading and writing.  The stream is positioned at the
                     beginning of the file.
           'w'   Truncate file to zero length or create text file for writing.
                     The stream is positioned at the beginning of the file.
           'w+'  Open for reading and writing.  The file is created if it does not
                     exist, otherwise it is truncated.  The stream is positioned at
                     the beginning of the file.
           'a'   Open for writing.  The file is created if it does not exist.  The
                     stream is positioned at the end of the file.  Subsequent writes
                     to the file will always end up at the then current end of file,
                     irrespective of any intervening fseek(3) or similar.
           'a+'  Open for reading and writing.  The file is created if it does not
                     exist.  The stream is positioned at the end of the file.  Subse-
                     quent writes to the file will always end up at the then current
                     end of file, irrespective of any intervening fseek(3) or similar.
    :return: None.
    """
    with codecs.open(file_name, mode=mode, encoding="utf-8") as f:
        f.write(string_text)
        f.close()

    return None
    
def create_punctuation_labels(transcription, tokenizer, kept_punctuation):
    """
    Tokenizes sentences containing punctuations and then extracts word as inputs and punctuation as labels.
    
    Parameters
    ----------
    transcription: str
        The directory path the pre-processed text file
    
    tokenizer: 
        A Huggingface model's tokenizer. e.g., `tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`

    kept_punctuation: str 
        A string containing punctuations that are to be kept as labels. e.g., ".,!?" Those punctuations not included in this parameter will be considered as inputs 
        
    Returns
    ----------
    List of inputs, List of labels
    """
    # tokenize
    tokenized_transcription = tokenizer.tokenize(transcription)

    inputs = []
    labels = []
    # loop over all tokens
    for i in range(len(tokenized_transcription)):
        curr_token = tokenized_transcription[i]
        if i == len(tokenized_transcription) - 1:
            next_token = ""
        else:
            next_token = tokenized_transcription[i+1]

        # add to inputs if not a punctuation
        if curr_token not in kept_punctuation:
            inputs.append(curr_token)
            # add punctuation to labels if next token is punctuation
            if next_token in kept_punctuation:
                labels.append(next_token)
            else:
                labels.append("")

    reconstructed_tokenized = []
    for token, punc in zip(inputs, labels):
        if token != "":
            reconstructed_tokenized.append(token)
        if punc != "":
            reconstructed_tokenized.append(punc)

    return inputs, labels

def create_encoded_labels(labels, punc_2_id, max_seq_len=512):
    """
    Creates encoded labels similar to the `encode_plus` function from huggingface with an exception that [CLS] and [SEP] tokens are padded with [0] instead.  
    
    Parameters
    ----------
    labels: list
       A list of labels

    max_seq_len: int 
       If set to a number, will limit the total sequence returned so that it has a maximum length. Make sure that the assigned value is the same as the `max_length` parameter
       used to encode inputs with the `encode_plus()` function (from huggingface) by referring to `encoded_inputs` variable in the `create_calls_dataset()` function. 
        
    Returns
    ----------
    List of encoded labels
    """

    encoded_labels = []
    for label in tqdm(labels):
        encoded_label = [punc_2_id[lbl] for lbl in label]
        # add no punctuation flags for special tokens
        encoded_label = [0] + encoded_label[:(max_seq_len - 2)] + [0]
        # pad
        encoded_label += [0 for _ in range(max_seq_len - len(encoded_label))]
        encoded_labels.append(encoded_label)
        
    return encoded_labels


def create_tensordataset(encoded_inputs, encoded_labels):
    """
    Creates a tensor dataset with the ENCODED inputs, labels and attention tensors.  
    
    Parameters
    ----------
    encoded_inputs: list
       A list of dictonaries containing 'input_ids' tensors and 'attention_mask' tensors. Refer to `encoded_inputs` variable in the `create_calls_dataset()` function.
       Or check out the `encode_plus` function at https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html#PreTrainedTokenizer.encode_plus

    encoded_labels: list 
       A list of encoded labels tensors from the `create_encoded_labels()` function.
        
    Returns
    ----------
    A tensor dataset containing inputs, labels, and attention tensors
    """
    inputs_tensor = torch.cat([encoded_inputs[i]['input_ids']for i in range(0,len(encoded_inputs))])
    labels_tensor = torch.LongTensor(encoded_labels)
    attn_tensor = torch.cat([encoded_inputs[i]['attention_mask']for i in range(0,len(encoded_inputs))])

    return TensorDataset(inputs_tensor, labels_tensor, attn_tensor)

def create_movies_transcriptions(dataset_path,
                                  lines_file="movie_lines.txt",
                                  conversations_file="movie_conversations.txt",
                                  sep=" +++$+++ ",
                                  encoding='iso-8859-1'):
    """
    This function is used to create transcription from the raw Cornell movie corpus dataset 
    
    Parameters
    ----------
    dataset_path:
    A string containing the path to the raw Cornell movie corpus dataset. 
    
    lines_file:
    A string that describes the file name of the movie lines.
    
    conversations_file:
    A string that describes the file name of the movie conversations.

    sep:
    The string of the seperator used in the movie lines and movie conversations file. 
    
    encoding:
    Encoding used to read the movie lines and movie conversations file.
    
    Returns
    ----------
    A list of strings containing the movie conversations that has been seperated and combined using the movie lines file. 
    """

    lines_file = os.path.join(dataset_path, lines_file)
    conversations_file = os.path.join(dataset_path, conversations_file)

    line_dict = {}
    with open(lines_file, encoding=encoding) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split(sep)
        line_id = line[0]
        utt = line[-1]
        line_dict[line_id] = utt

    transcriptions = []
    with open(conversations_file, encoding=encoding) as f:
        conversations = f.readlines()

    for convo in conversations:
        convo = convo.strip().split(sep)[-1]
        list_of_utt = ast.literal_eval(convo)
        transcription = " ".join(line_dict[elem] for elem in list_of_utt)
        transcriptions.append(transcription)

    return transcriptions

def pre_process_transcription_mov(string):
    """
    This function is used to pre-process a string that has been transformed by the `create_movies_transcriptions()` function. 
    
    Parameters
    ----------
   string: str
       A string to be pre-processed
        
    Returns
    ----------
    A string that has been pre-processed
    """
    
    # Convert to lower case
    string = string.lower()

    # Change all space-like characters to a standard space
    string = re.sub(r'\t', ' ', string)
    
    # Remove carriage returns:    
    string = re.sub(r'\r', '', string)
    
    #Standardize "okay":
    string = re.sub(r'\bo\.k\.', ' okay ', string)
    string = re.sub(r'\bok\b', ' okay ', string)
    
    # replace "yep" and "yeah" with yes
    string = string.replace("yeah", "yes")
    string = string.replace("yep", "yes")

    string = string.replace("uh-uh", "no")
    string = string.replace("nuh-uh", "no")
    
    replacement_patterns = {
    "patts":
        [   
            #replace all variations of mm-hmm with yes
            (r'\bmm-\w+\b', 'yes'),
            (r'\bm+hm+\b', 'yes'),
            
            #remove anything with chevron
            (r'(\<\w+\>(\.+|\?+|\!+|\,+))|(\<.+?\>)', ' '),
            
            #replace ...? . . .? ... ? with ? 
            (r'(\.{2,}\?)|(\. {2,}\.\?)|(\.{2,} \?)', '?'),
             
            # Remove interjections:
            (r'(um-hum(\.+|\?+|\!+|\,+|\$+))|(um-hum)', ' '),
            (r'(\bumhm+\b(\.+|\?+|\!+|\,+|\$+))|(\bumhm+\b)', ' '),
            (r'(\baw+\b(\.+|\?+|\!+|\,+|\$+))|(\baw+\b)', ' '),
            (r'(\bmmm+\b(\.+|\?+|\!+|\,+|\$+))|(\bmmm+\b)', ' '),
            (r'(\baha+\b(\.+|\?+|\!+|\,+|\$+))|(\baha+\b)', ' '),
            (r'(\ber+\b(\.+|\?+|\!+|\,+|\$+))|(\ber+\\b)', ' '), (r'(\buh+\b(\.+|\?+|\!+|\,+|\$+))|(\buh+\b)', ' '),
            (r'(\buhm+\b(\.+|\?+|\!+|\,+|\$+))|(\buhm+\b)', ' '), (r'\behm\b', ' '),
            (r'(\bhuh\b(\.+|\?+|\!+|\,+|\$+))|(\bhuh\b)', ' '), (r'\bhum\b', ' '),
            (r'(\bhm+\b(\.+|\?+|\!+|\,+|\$+))|(\bhm+\b)', ' '),
            (r'(\bum+\b(\.+|\?+|\!+|\,+|\$+))|(\bum+\b)', ' '), 
            (r'(\boh+\b(\.+|\?+|\!+|\,+|\$+))|(\boh+\b)', ' '),
            (r'(\bah+\b(\.+|\?+|\!+|\,+|\$+))|(\bah+\b)', ' '), (r'(\b\w+ahh+\b(\.+|\?+|\!+|\,+|\$+))|(\b\w+ahh+\b)', ' '),
            (r'(\baa+nn+\b(\.+|\?+|\!+|\,+|\$+))|(\baa+nn+\b)', ' '),
            (r'(\beh+\b(\.+|\?+|\!+|\,+|\$+))|(\beh+\b)', ' '),
            (r'aaa+', 'a'), (r'eee+', 'e'), (r'iii+', 'i'),  (r'ooo+', 'o'), (r'uuu+', 'u'),
            (r'fff+', 'f'), (r'hhh+', 'h'), (r'sss+', 's'), (r'rrr+', 'r'), (r'ooo+', 'o'), 
            (r'(\bahowwwo\b)', ' '), (r'(\barrrgggaahh\b)', ' '),
            
            #replace '!!!' with '!'
            (r'\!{3,}', '!'),
            
            #replace '??' with '?'
            (r'\?{2,}', '?'),
        
            #replace '...' with '$'
            (r'\.{3,}', '$'),
            
            #replace '. . .' with '$'
            (r'\. \. \.', '$ '),
            
            #replace ".." with "$"
            (r'\.\.', '$'),
            
            #replace". ." with ""
            (r'\. \.', ''),
            
            (r'\~\.\?', ''),
            
            (r'\.\-\.', ''),
            
            # Espaciates punctuation:
            (r'([\;\:])', ' \1'),
            
            (r'( +\,)', ', '),
            (r'( +\.)', '. '),
            (r'( +\!)', '. '), 
            (r'( +\:)', ', '),
            (r'( +\;)', ', '),
            (r'( +\?)', '? '),
            (r'( +\$)', '$ '),

            # Remove "-" to reduce variability:
            (r'\-', ' '),
            
            #Remove unwanted punctuations: # keep "'" unlike patrice which he removed
            #Added "$" to this in experiment 3 to get rid of "$" 
            ("[\(\)%+\\*<>{};/`:&\"^\[\]|@_=~#$]", "")
        ]
    }
    
    for pattern in replacement_patterns["patts"]:
            string = re.sub(
                pattern[0], pattern[1], string)
    
    #simplify long pauses ("---" or "--" -> "-")
    string = string.replace("---", "").replace("--","").replace("-", "")
    string = string.replace("!", ".")
    string = string.replace(":", "")
    string = string.replace(";", "")
    string = string.replace(",", "")
    
    # Remove double spaces:
    string = re.sub(r'  +', ' ', string)
    string = re.sub(r'^ +', '', string)
    string = re.sub(r'^ +', '', string, flags=re.MULTILINE)
    string = re.sub(r'(\$\?)|(\?{2,})', '\?', string)
    string = re.sub(r'(\?{2,}\$)', '$', string)
    string = string.strip("")
    string = string.strip()

    # Remove empty lines
    string = re.sub(r'(^$)', '', string, flags=re.MULTILINE)
    
    # Remove empty lines with spaces
    string = re.sub(r'(^\s*$)', '', string, flags=re.MULTILINE)
    
    return string

def create_movies_dataset(tokenizer,
                          raw_dataset_path,
                          preprocessed_transcription_path,
                          cached_path,
                          kept_punctuation,
                          punc_2_id,
                          transcription_path_filename="transcription.json",
                          inputs_path_filename="inputs.json",
                          labels_path_filename="labels.json",
                          cached_path_filename = "cached_%s.pt",
                          max_seq_len = 512,
                          eval_ratio = 0.1,
                          test_ratio = 0.1):
    """
    Converts a raw movie corpus to a tensor dataset that is ready to be fed into the BERT model.
    
    This function assumes that the "train", "eval" and "test" tensor dataset paths are structured as like the following:
        Train tensordataset path: /experiments/punctuation_restoration/data/train/...
        Evaluation tensordataset path: /experiments/punctuation_restoration/data/eval/...
        Testing tensordataset path: /experiments/punctuation_restoration/data/test/...
        
    This function also assumes that the file name for the "train", "eval" and "test" tensor dataset have the following format:
        /.../cached_train.pt
        /.../cached_eval.pt
        /.../cached_test.pt
    
    Parameters
    ----------
    tokenizer: 
    A Huggingface model's tokenizer. e.g., `tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`
    
    raw_dataset_path:
    A string that contains the path of where the raw Cornell movie conversations and lines file are stored. 
    
    preprocessed_transcription_path:
    A string that contains the path of where the pre-processed transcriptions processed by `create_movies_transcriptions()` are stored.  
    
    cached_path:
    A string that contains the path of where the encoded labels, inputs and attention masks that has been converted into a tensordataset are stored. 
    This string must contain '%s' for 4 different directory paths to store the "full", "train", "eval", and "test" dataset.  

    kept_punctuation:
    A string that includes the punctuations that you'd like to keep for the `create_punctuation_labels()` function. For example, if you like to keep 
    the null class, period, comma, and question mark, you can do so like this: ".,?" . You do not need to specify the null class as it's created
    automatically. 

    punc_2_id:
    A dictionary where the keys are numerical encoding on the punctuations while the value is it's associated punctuation. For example, if you have the
    null class, period, comma, and question mark to be encoded, you can do so with the following: {"0": "\"\"", "1": ".", "2": ",", "4": "?"}
    
    transcription_path_filename:
    A string describing the name of the transcription file.
    
    inputs_path_filename:
    A string describing the name of the inputs file.
    
    labels_path_filename:
    A string describing the name of the labels file.
    
    cached_path_filename:
    A string describing the name of the tensor dataset (contains the encoded inputs, labels, and attention masks) to be saved.

    max_seq_len:
    If set to a number, will limit the total sequence returned so that it has a maximum length. Make sure that the assigned value is the same as the `max_length` parameter
    used to encode inputs with the `encode_plus()` function (from huggingface).
    
    eval_ratio:
    The ratio of the evaluation dataset to be split by the `random_split()` function from torch.
    
    test_ratio:
    The ratio of the test dataset to be split by the `random_split()` function from torch.
        
    Returns
    ----------
    Train and evaluation dataset containing inputs, labels, and attention tensors
    """
    
    # create preprocessed transcriptions from cornell movie dialogs dataset
    if not os.path.exists(preprocessed_transcription_path + transcription_path_filename):
        try:
            os.makedirs(preprocessed_transcription_path)
        except FileExistsError:
            pass
        print("Create transcriptions and preprocessing from Cornell Movie Dialogs Dataset...")
        raw_transcriptions = create_movies_transcriptions(dataset_path=raw_dataset_path)
        
        # use list(filter(None,...)) to remove any empty strings. Important to do this for encoding later on.  
        transcriptions = list(filter(None,[pre_process_transcription_mov(text) for text in tqdm(raw_transcriptions)]))
        
        with open(preprocessed_transcription_path+transcription_path_filename, "w", encoding='utf8') as f:
            json.dump(transcriptions, f)
    else:
        print(f"Loading movies transcriptions from:",preprocessed_transcription_path + transcription_path_filename)
        with open(preprocessed_transcription_path + transcription_path_filename, encoding='utf8') as f:
            transcriptions = json.load(f)
    
    inputs_path = preprocessed_transcription_path + inputs_path_filename
    labels_path = preprocessed_transcription_path + labels_path_filename
    if not os.path.exists(inputs_path) and not os.path.exists(labels_path):
        print(f"Creating movies inputs and labels from movies transcriptions...")
        tokenized_inputs = []
        tokenized_labels = []
        for transcription in tqdm(transcriptions):
            inputs, labels = create_punctuation_labels(
                transcription, tokenizer,kept_punctuation)
            tokenized_inputs.append(inputs)
            tokenized_labels.append(labels)

        with open(inputs_path, "w") as f:
            json.dump(tokenized_inputs, f)
        with open(labels_path, "w") as f:
            json.dump(tokenized_labels, f)
    else:
        print("Loading inputs from:",inputs_path, "\n", "Loading labels from:",labels_path)
        with open(inputs_path) as f:
            tokenized_inputs = json.load(f)
        with open(labels_path) as f:
            tokenized_labels = json.load(f)
    
    cached_all_path = cached_path % ("all") + cached_path_filename % ("all")
    if not os.path.exists(cached_all_path):
        try:
            os.makedirs(cached_path % ("all"))
        except FileExistsError:
            pass

        #Adding ["CLS"], ["SEP"] tokens to each utterance, padding the utterances, and return torch tensors
        print("Converting tokens to ids, adding special tokens, padding and finally return encoded labels and inputs as torch tensors...")
        encoded_inputs = [tokenizer.encode_plus(inputs, add_special_tokens = True, max_length=512, pad_to_max_length=True, return_tensors='pt') for inputs in tqdm(tokenized_inputs)]
        encoded_labels = create_encoded_labels(tokenized_labels, punc_2_id, max_seq_len=512, )
        
        print("Creating TensorDataset")
        all_dset = create_tensordataset(encoded_inputs,encoded_labels)

        print(f"Saving movies TensorDataset to:", cached_all_path)
        torch.save(all_dset, cached_all_path)
    else:
        print(f"Loading movies TensorDataset from:", cached_all_path)
        with open(cached_all_path) as f:
            all_dset = torch.load(cached_all_path)
    
    modes = ["train","eval","test"]
    for mode in modes:
        if not os.path.exists(cached_path % mode):
            try:
                os.makedirs(cached_path % mode)
            except FileExistsError:
                pass
            
    cached_train_path = cached_path % ("train") + cached_path_filename % ("train")
    cached_eval_path = cached_path % ("eval") + cached_path_filename % ("eval")
    cached_test_path = cached_path % ("test") + cached_path_filename % ("test")

    if not os.path.exists(cached_train_path) or not os.path.exists(cached_eval_path) or not os.path.exists(cached_test_path):
        eval_size = int(eval_ratio * len(all_dset))
        test_size = int(test_ratio * len(all_dset))
        train_size = len(all_dset) - eval_size - test_size

        train_dset, eval_dset, test_dset = random_split(all_dset, [train_size, eval_size, test_size])
        
        print("Saving train dataset to:", cached_train_path)
        torch.save(train_dset, cached_train_path)
        print("Saving eval dataset to:", cached_eval_path)
        torch.save(eval_dset, cached_eval_path)
        print("Saving test dataset to:", cached_test_path)
        torch.save(test_dset, cached_test_path)

    else:
        print("Loading movies train torch dataset from:",cached_train_path)
        train_dset = torch.load(cached_train_path)
        print("Loading movies eval torch dataset from:",cached_eval_path)
        eval_dset = torch.load(cached_eval_path)

    return train_dset, eval_dset