#ディスク容量が不足するためこのコードは使用しない。

import argparse
import logging
import os
import torch
from tqdm import tqdm
from transformers import BertModel

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
            last_hidden_states=outputs[0]   #(1,512,768)

            ret[i]=last_hidden_states[0]

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

def main(input_dir,im_features_dir,save_dir):
    #Load encoded text from a cached file.
    logger.info("Load encoded text from {}.".format(input_dir))

    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    #Load the list of options.
    list_filepath=os.path.join(input_dir,"options_list.txt")

    logger.info("Load the list of options. {}".format(list_filepath))

    with open(list_filepath,"r",encoding="UTF-8") as r:
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

    #Load a BERT model.
    logger.info("Load a pre-trained BERT model.")

    bert_model=BertModel.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    bert_model.to(device)

    #Create a directory to save the cache files in.
    os.makedirs(save_dir,exist_ok=True)

    #Create input embeddings.
    logger.info("Start creating input embeddings.")

    for i in tqdm(range(input_ids.size()[0])):
        inputs_embeds=torch.empty(20,512,768)
        inputs_token_type_ids=torch.empty(20,512)

        text_embeddings=create_text_embeddings(bert_model,input_ids[i])

        ops=options[i]
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

            inputs_embeds[j]=option_embedding
            inputs_token_type_ids[j]=inputs_token_type_ids_tmp

        inputs_save_dir=os.path.join(save_dir,str(i))
        os.makedirs(inputs_save_dir,exist_ok=True)
        torch.save(inputs_embeds,os.path.join(inputs_save_dir,"inputs_embeds.pt"))
        torch.save(inputs_token_type_ids,os.path.join(inputs_save_dir,"inputs_token_type_ids.pt"))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="VisualAIO2")

    parser.add_argument("--input_dir",type=str,default="~/EncodedCache/Train")
    parser.add_argument("--im_features_dir",type=str,default="~/VGG16Features")
    parser.add_argument("--save_dir",type=str,default="~/VisualAIO2InputCache/Train")

    args=parser.parse_args()

    main(args.input_dir,args.im_features_dir,args.save_dir)
