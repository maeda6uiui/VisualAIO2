import argparse
import logging
import os
import sys
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import (
    BertModel,
    BertForMultipleChoice,
)

import hashing

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Options(object):
    def __init__(self):
        self.options=[]

    def append(self,option):
        self.options.append(option)

    def get(self,index):
        return self.options[index]

def load_options_list(filepath):
    logger.info("Load a list of options. {}".format(filepath))

    with open(filepath,"r",encoding="UTF-8") as r:
        lines=r.read().splitlines()

    options=[]
    ops=None
    for line in lines:
        if ops is None:
            ops=Options()

        if line=="":
            options.append(ops)
            ops=None
        else:
            ops.append(line)

    return options

def create_dataloader(input_dir,batch_size,num_options=4,shuffle=True,drop_last=True):
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))[:,:num_options,:]
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))[:,:num_options,:]
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))[:,:num_options,:]
    labels=torch.load(os.path.join(input_dir,"labels.pt"))[:]
    indices=torch.empty(input_ids.size()[0],dtype=torch.long)
    for i in range(input_ids.size()[0]):
        indices[i]=i

    dataset=TensorDataset(indices,input_ids,attention_mask,token_type_ids,labels)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)

    return dataloader

def create_text_embeddings(bert_model,options_ids):
    """
    Creates text embeddings.

    Args:
        bert_model (transformers.BertModel): BERT model
        options_ids (torch.tensor (x,512)): Encoded text

    Returns:
        torch.tensor (x,512,768): Text embeddings
    """
    bert_model.eval()

    num_options=options_ids.size()[0]

    ret=torch.empty(num_options,512,768).to(device)

    for i in range(num_options):
        input_ids=options_ids[i].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs=bert_model(input_ids)
            embeddings=bert_model.get_input_embeddings()

            ret[i]=embeddings(input_ids)

    return ret

def create_option_embedding(text_embedding,im_embedding):
    """
    Creates an option embedding.

    Args:
        text_embedding (torch.tensor (512,768)): Text embedding
        im_embedding (torch.tensor (x,768)): Image embedding

    Returns:
        torch.tensor (512,768): Option embedding
        torch.tensor (512): Token type IDs
    """
    im_embedding_length=im_embedding.size(0)
    text_embedding=text_embedding[:512-im_embedding_length]
    option_embedding=torch.cat([text_embedding,im_embedding],dim=0)

    token_type_ids=torch.zeros(512,dtype=torch.long).to(device)
    for i in range(512-im_embedding_length,512):
        token_type_ids[i]=1

    return option_embedding,token_type_ids

def create_inputs_embeds_and_token_type_ids(bert_model,input_ids,indices,options,im_features_dir):
    batch_size=input_ids.size()[0]
    num_options=input_ids.size()[1]

    inputs_embeds=torch.empty(batch_size,num_options,512,768).to(device)
    inputs_token_type_ids=torch.empty(batch_size,num_options,512,dtype=torch.long).to(device)

    for i in range(batch_size):
        text_embeddings=create_text_embeddings(bert_model,input_ids[i])

        ops=options[indices[i]]
        for j in range(num_options):
            article_name=ops.get(j)
            article_hash=hashing.get_md5_hash(article_name)

            option_embedding=None
            inputs_token_type_ids_tmp=None
            im_features_filepath=os.path.join(im_features_dir,article_hash+".pt")

            if os.path.exists(im_features_filepath):
                if torch.cuda.is_available():
                    im_embedding=torch.load(im_features_filepath).to(device)
                else:
                    im_embedding=torch.load(im_features_filepath,map_location=torch.device("cpu")).to(device)

                option_embedding,inputs_token_type_ids_tmp=create_option_embedding(text_embeddings[j],im_embedding)
            else:
                option_embedding=text_embeddings[j]
                inputs_token_type_ids_tmp=torch.zeros(512,dtype=torch.long).to(device)

            inputs_embeds[i,j]=option_embedding
            inputs_token_type_ids[i,j]=inputs_token_type_ids_tmp

    return inputs_embeds,inputs_token_type_ids

def simple_accuracy(preds, labels):
    """
    Calculates accuracy.

    Parameters
    ----------
    preds: numpy.ndarray
        Predicted labels
    labels: numpy.ndarray
        Correct labels

    Returns
    ----------
    accuracy: float
        Accuracy
    """
    return (preds == labels).mean()

def test(bert_model,bfmc_model,options,im_features_dir,dataloader):
    bert_model.eval()
    bfmc_model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    correct_labels = None

    for batch_idx,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        with torch.no_grad():
            batch_size=len(batch)
            batch = tuple(t for t in batch)

            inputs = {
                "indices":batch[0].to(device),
                "input_ids": batch[1].to(device),
                "attention_mask": batch[2].to(device),
                "token_type_ids": batch[3].to(device),
                "labels": batch[4].to(device)
            }

            inputs_embeds,inputs_token_type_ids=create_inputs_embeds_and_token_type_ids(
                bert_model,inputs["input_ids"],inputs["indices"],options,im_features_dir
            )

            bfmc_inputs={
                "inputs_embeds":inputs_embeds,
                "attention_mask":inputs["attention_mask"],
                "token_type_ids":inputs_token_type_ids,
                "labels":inputs["labels"]
            }

            outputs = bfmc_model(**bfmc_inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            correct_labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            correct_labels = np.append(
                correct_labels, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    pred_labels = np.argmax(preds, axis=1)

    accuracy = simple_accuracy(pred_labels, correct_labels)

    return pred_labels,correct_labels,accuracy

def main(test_input_dir,im_features_dir,test_upper_bound,result_save_dir):
    #Load a list of options.
    logger.info("Load a list of options.")
    test_options=load_options_list(os.path.join(test_input_dir,"options_list.txt"))

    #Create a dataloader.
    logger.info("Create a test dataloader from {}.".format(test_input_dir))
    test_dataloader=create_dataloader(test_input_dir,4,num_options=20,shuffle=False,drop_last=False)

    #Load a pre-trained BERT model.
    logger.info("Load a pre-trained BERT model.")
    bert_model=BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    bert_model.to(device)

    #Create a BertForMultipleChoice model.
    logger.info("Create a BertForMultipleChoice model.")
    bfmc_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    bfmc_model.to(device)

    #Create a directory to save the results in.
    #os.makedirs(result_save_dir,exist_ok=True)

    logger.info("Start test.")
    for i in range(test_upper_bound):
        parameters_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(i+1))
        logger.info("Load model parameters from {}.".format(parameters_filepath))
        if torch.cuda.is_available():
            bfmc_model.load_state_dict(torch.load(parameters_filepath))
        else:
            bfmc_model.load_state_dict(torch.load(parameters_filepath,map_location=torch.device("cpu")))

        pred_labels,correct_labels,accuracy=test(bert_model,bfmc_model,test_options,im_features_dir,test_dataloader)

        logger.info("Accuracy: {}".format(accuracy))

        #Save results as text files.
        res_filepath=os.path.join(result_save_dir,"result_test_{}.txt".format(i+1))
        labels_filepath=os.path.join(result_save_dir,"labels_test_{}.txt".format(i+1))

        with open(res_filepath,"w") as w:
            w.write("Accuracy: {}\n".format(accuracy))

        with open(labels_filepath,"w") as w:
            for pred_label,correct_label in zip(pred_labels,correct_labels):
                w.write("{} {}\n".format(pred_label,correct_label))

    logger.info("Finished model test.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="VisualAIO2")

    parser.add_argument("--test_input_dir",type=str,default="~/EncodedCache/Dev2")
    parser.add_argument("--im_features_dir",type=str,default="~/VGG16Features")
    parser.add_argument("--test_upper_bound",type=int,default=10)
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir")

    args=parser.parse_args()

    main(
        args.test_input_dir,
        args.im_features_dir,
        args.test_upper_bound,
        args.result_save_dir
    )
