import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

def plot_loss_accuracy(train_loss, eval_loss, train_acc, eval_acc, path):
    """
    Plots the accuracy and loss values for each checkpoint accumulated over all epochs
    Parameters
    ----------
    train_loss: str
        Text file path containing the train loss

    eval_loss: str
        Text file path containing the eval loss
   
    train_acc: str
        Text file pathcontaining the train accuracy

    eval_acc: str
        Text file path containing the eval accuracy

    path: str
        File path to save the image
    """

    t_loss = pd.read_csv(train_loss, sep="\n", names=["train_loss"])
    e_loss = pd.read_csv(eval_loss, sep="\n", names=["eval_loss"])
    t_acc = pd.read_csv(train_acc, sep="\n", names=["train_acc"])
    e_acc = pd.read_csv(eval_acc, sep="\n", names=["eval_acc"])

    df_loss = pd.concat([t_loss, e_loss], axis=1)
    df_acc = pd.concat([t_acc, e_acc], axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    min_loss_y = df_loss['eval_loss'].min()
    min_loss_x = df_loss['eval_loss'].idxmin()

    max_acc_y = df_acc['eval_acc'].max()
    max_acc_x = df_acc['eval_acc'].idxmax()

    sns.lineplot(data=df_loss, ax=ax)
    ax.set(xlabel='Checkpoints', ylabel='Loss')
    ax.legend(bbox_to_anchor=[0.5, 0.5], loc='center', labels=['train','eval'])
    ax.annotate("x={:.3f}, min validation loss={:.3f}".format(min_loss_x, min_loss_y), xy=(min_loss_x, min_loss_y+0.2))
    ax2 = ax.twinx()
    ax2.set(ylabel='Accuracy')
    ax2.annotate("x={:.3f}, max validation acc={:.3f}".format(max_acc_x, max_acc_y), xy=(max_acc_x-10, max_acc_y-10))

    sns.lineplot(data=df_acc, ax=ax2, legend=False)

    print("Saving image..")
    plt.savefig(path)
    print("Image saved at:", path)

    plt.show()

    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          path=None,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Parameters
    ----------
    cm: Confusion matrix provided by sklearn

    classes: list of punctuations
    
    normalize: 'True' to normalize confusion matrix. "False" by default. 

    path: String that contains the path to save the confusion matrix image

    test_loader: A torch dataset that consist of inputs, labels, and attention tensors

    model: Model that is used for test. 

    title: title of the plot

    cmap: A matplotlib color map for the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(9,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)

    print("Saving image..")
    plt.savefig(path)
    print("Image saved at:", path)

    plt.show()

def evaluate_model(test_loader_path, model_path, id_2_punc, path=None):
    """
    Function to test the trained BERT model. Input, labels and attention tensors are loaded to GPU by default.
    If using CPU, please remove `..to(DEVICE)` after unsqueezing tensors.

    Parameters
    ----------
    test_loader_path: Path of the test torch dataset that consist of inputs, labels, and attention tensors

    model_path: Path of trained model used for test. 

    id_2_punc: mapping of the encoded punctuation back to the original punctuation. e.g.,{"0": "\"\"", "1": "."}

    path: String that contains the path to save the confusion matrix image

    Example:
    ---------
    model =  BertForTokenClassification.from_pretrained('/path/to/trained_model')
    test_loader = torch.load('path/to/test_tensor_dataset')

    evaluate_model(test_loader, model, id_2_punc={"0": "\"\"", "1": ".", "2": "?"} )
    """
    test_loader = torch.load(test_loader_path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForTokenClassification.from_pretrained(model_path).to(DEVICE)

    total_test_loss = 0
    total_num_correct = 0
    total_num_examples = 0
    all_labels = []
    all_predictions = []
    all_attns = []

    model.eval()

    with torch.no_grad():
        for (inputs, labels, attns) in tqdm(test_loader, desc="Evaluate"):

            inputs = inputs.unsqueeze(0).to(DEVICE)
            labels = labels.unsqueeze(0).to(DEVICE)
            attns = attns.unsqueeze(0).to(DEVICE)

            outputs = model(inputs, labels=labels, attention_mask=attns)
            
            loss, scores = outputs[:2]

            predictions = torch.argmax(scores, dim=-1)

             # add loss to total loss
            total_test_loss += loss.item()

            # compute accuracy with two number (important to remove tokens with attention mask!)
            total_num_correct += ((predictions ==
                                   labels).long() * attns).sum().item()
            total_num_examples += attns.sum().item()

            # used to create confusion matrix
            all_labels += labels.cpu().flatten().tolist()
            all_predictions += predictions.cpu().flatten().tolist()
            all_attns += attns.cpu().flatten().tolist()

        # remove elements where attn mask is 0
        all_labels = [all_labels[i] for i in range(len(all_labels)) if all_attns[i] != 0]
        all_predictions = [all_predictions[i] for i in range(len(all_predictions)) if all_attns[i] != 0]

        # compute loss, accuracy and confusion matrix
        eval_loss = total_test_loss / len(test_loader)
        eval_accuracy = (total_num_correct / total_num_examples) * 100
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions)
        accuracy = (conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]).diagonal()
        result = pd.DataFrame(
            np.array([accuracy, precision, recall, f1]),
            columns=list(id_2_punc.values())[0:],
            index=['Accuracy', 'Precision', 'Recall', 'F1'])
        
        plot_confusion_matrix(conf_matrix, classes=list(id_2_punc.values())[0:], path=path)
        print("eval loss:", eval_loss,"\n")
        print("eval_accuracy:", eval_accuracy,"\n")

    return result