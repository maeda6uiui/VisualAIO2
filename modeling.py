import argparse
import logging
import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import (
    BertModel,
    BertForMultipleChoice,
    AdamW,
    get_linear_schedule_with_warmup,
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

def create_dataloader(input_dir,batch_size,shuffle=True,drop_last=True):
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))
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
        options_ids (torch.tensor (20,512)): Encoded text

    Returns:
        torch.tensor (20,512,768): Text embeddings
    """
    bert_model.eval()

    ret=torch.empty(20,512,768).to(device)

    for i in range(20):
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

    inputs_embeds=torch.empty(batch_size,20,512,768).to(device)
    inputs_token_type_ids=torch.empty(batch_size,20,512,dtype=torch.long).to(device)

    for i in range(batch_size):
        text_embeddings=create_text_embeddings(bert_model,input_ids[i])

        ops=options[indices[i]]
        for j in range(20):
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

def train(bert_model,bfmc_model,options,im_features_dir,optimizer,scheduler,dataloader):
    bert_model.eval()
    bfmc_model.train()

    for batch_idx,batch in enumerate(dataloader):
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

        # Initialize gradiants
        optimizer.zero_grad()
        # Forward propagation
        outputs = bfmc_model(**bfmc_inputs)
        loss = outputs[0]
        # Backward propagation
        loss.backward()
        nn.utils.clip_grad_norm_(bfmc_model.parameters(), 1.0)
        # Update parameters
        optimizer.step()
        scheduler.step()

        bfmc_model.zero_grad()

        if batch_idx % 10 == 0:
            logger.info("Current step: {}\tLoss: {}".format(batch_idx,loss.item()))

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

def evaluate(bert_model,bfmc_model,options,im_features_dir,dataloader):
    bert_model.eval()
    bfmc_model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    correct_labels = None

    for batch_idx,batch in enumerate(dataloader):
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

def main(batch_size,num_epochs,lr,train_input_dir,dev1_input_dir,im_features_dir,result_save_dir):
    logger.info("batch_size: {} num_epochs: {}".format(batch_size,num_epochs))

    #Load lists of options.
    logger.info("Load lists of options.")

    train_options=load_options_list(os.path.join(train_input_dir,"options_list.txt"))
    dev1_options=load_options_list(os.path.join(dev1_input_dir,"options_list.txt"))

    #Create dataloaders.
    logger.info("Create a training dataloader from {}.".format(train_input_dir))
    train_dataloader=create_dataloader(train_input_dir,batch_size,shuffle=True,drop_last=True)

    logger.info("Create a dev1 dataloader from {}.".format(dev1_input_dir))
    dev1_dataloader=create_dataloader(dev1_input_dir,4,shuffle=False,drop_last=False)

    #Load a pre-trained BERT model.
    logger.info("Load a pre-trained BERT model.")
    bert_model=BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    bert_model.to(device)

    #Create a BertForMultipleChoice model.
    logger.info("Create a BertForMultipleChoice model.")
    bfmc_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    bfmc_model.to(device)

    #Create an optimizer and a scheduler.
    optimizer=AdamW(bfmc_model.parameters(),lr=lr,eps=1e-8)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    #Create a directory to save the results in.
    os.makedirs(result_save_dir,exist_ok=True)

    logger.info("Start model training.")
    for epoch in range(num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch+1,num_epochs))

        train(bert_model,bfmc_model,train_options,im_features_dir,optimizer,scheduler,train_dataloader)
        pred_labels,correct_labels,accuracy=evaluate(bert_model,bfmc_model,dev1_options,im_features_dir,dev1_dataloader)

        #Save model parameters.
        checkpoint_filepath=os.path.join(result_save_dir,"checkpoint_{}.pt".format(epoch+1))
        torch.save(bfmc_model.state_dict(),checkpoint_filepath)

        #Save results as text files.
        res_filepath=os.path.join(result_save_dir,"result_{}.txt".format(epoch+1))
        labels_filepath=os.path.join(result_save_dir,"labels_{}.txt".format(epoch+1))

        with open(res_filepath,"w") as w:
            w.write("Accuracy: {}\n".format(accuracy))

        with open(labels_filepath,"w") as w:
            for pred_label,correct_label in zip(pred_labels,correct_labels):
                w.write("{} {}\n".format(pred_label,correct_label))

    logger.info("Finished model training.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="VisualAIO2")

    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--num_epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=5e-5)
    parser.add_argument("--train_input_dir",type=str,default="~/EncodedCache/Train")
    parser.add_argument("--dev1_input_dir",type=str,default="~/EncodedCache/Dev1")
    parser.add_argument("--im_features_dir",type=str,default="~/VGG16Features")
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir")

    args=parser.parse_args()

    main(
        args.batch_size,
        args.num_epochs,
        args.lr,
        args.train_input_dir,
        args.dev1_input_dir,
        args.im_features_dir,
        args.result_save_dir
    )
