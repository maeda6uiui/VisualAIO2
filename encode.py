import argparse
import gzip
import json
import logging
import os
import sys
import torch
from tqdm import tqdm
from transformers import BertJapaneseTokenizer

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class InputExample(object):
    """
    Input example
    """
    def __init__(self, qid, question, endings, label=None):
        self.qid = qid
        self.question = question
        self.endings = endings
        self.label = label

def load_examples(example_filepath):
    """
    Loads examples from a JSON file.

    Args:
        example_filepath (str): Filename of a JSON file containing examples

    Returns:
        [InputExamples]: Examples
    """
    num_options=20

    examples = []

    with open(example_filepath, "r", encoding="UTF-8") as r:
        lines = r.read().splitlines()

    for line in lines:
        data = json.loads(line)

        qid = data["qid"]
        question = data["question"].replace("_", "")
        options = data["answer_candidates"][:num_options]
        answer = data["answer_entity"]

        label=0
        if answer!="":
            label=options.index(answer)

        example = InputExample(qid, question, options, label)
        examples.append(example)

    return examples

def load_contexts(context_filepath):
    """
    Loads contexts from a gzip file.

    Args:
        context_filepath (str): Filename of a gzip file containing contexts

    Returns:
        {str:str}: Dict containing titles (key) and contexts (value)
    """
    contexts={}

    with gzip.open(context_filepath,mode="rt",encoding="utf-8") as r:
        for line in r:
            data = json.loads(line)

            title=data["title"]
            text=data["text"]

            contexts[title]=text

    return contexts

def encode_examples(tokenizer,examples,contexts,max_seq_length):
    """
    Encodes examples.

    Args:
        tokenizer (transformers.BertJapaneseTokenizer): BERT tokenizer
        examples ([InputExample]): Input examples
        contexts ({str:str}): Dict of contexts
        max_seq_length (int): Max length of input sequence to BERT model
    
    Returns:
        torch.tensor: Input ids
        torch.tensor: Attention mask
        torch.tensor: Token type IDs
        torch.tensor: Labels
    """
    num_options=20

    input_ids=torch.empty(len(examples),num_options,max_seq_length,dtype=torch.long)
    attention_mask=torch.empty(len(examples),num_options,max_seq_length,dtype=torch.long)
    token_type_ids=torch.empty(len(examples),num_options,max_seq_length,dtype=torch.long)
    labels=torch.empty(len(examples),dtype=torch.long)

    for example_index,example in enumerate(tqdm(examples)):
        #Process every option.
        for option_index,ending in enumerate(example.endings):
            #Text features
            text_a=example.question+"[SEP]"+ending
            text_b=contexts[ending]

            encoding = tokenizer.encode_plus(
                text_a,
                text_b,
                return_tensors="pt",
                add_special_tokens=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                max_length=max_seq_length,
                truncation_strategy="only_second"   #Truncate the context.
            )

            input_ids_tmp=encoding["input_ids"].view(-1)
            token_type_ids_tmp=encoding["token_type_ids"].view(-1)
            attention_mask_tmp=encoding["attention_mask"].view(-1)

            input_ids[example_index,option_index]=input_ids_tmp
            token_type_ids[example_index,option_index]=token_type_ids_tmp
            attention_mask[example_index,option_index]=attention_mask_tmp

        labels[example_index]=example.label

    return input_ids,attention_mask,token_type_ids,labels

def main(example_filepath,context_filepath,cache_save_dir):
    """
    Main function

    Args:
        example_filepath (str): Filename of a JSON file containing examples
        context_filepath (str): Filename of a gzip file containing examples
        cache_save_dir (str): Directory name to save cache files in
    """
    #Tokenizer
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    logger.info("Start loading examples from {}.".format(example_filepath))
    examples=load_examples(example_filepath)
    logger.info("Finished loading examples.")
    logger.info("Number of examples: {}".format(len(examples)))

    logger.info("Start loading contexts from {}.".format(context_filepath))
    contexts=load_contexts(context_filepath)
    logger.info("Finished loading contexts.")

    logger.info("Start encoding examples.")
    input_ids,attention_mask,token_type_ids,labels=encode_examples(tokenizer,examples,contexts,512)
    logger.info("Finished encoding examples.")

    os.makedirs(cache_save_dir,exist_ok=True)
    torch.save(input_ids,os.path.join(cache_save_dir,"input_ids.pt"))
    torch.save(attention_mask,os.path.join(cache_save_dir,"attention_mask.pt"))
    torch.save(token_type_ids,os.path.join(cache_save_dir,"token_type_ids.pt"))
    torch.save(labels,os.path.join(cache_save_dir,"labels.pt"))
    logger.info("Saved cache files in {}.".format(cache_save_dir))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="VisualAIO2")

    parser.add_argument("--example_filepath",type=str,default="../Data/train_questions.json")
    parser.add_argument("--context_filepath",type=str,default="../Data/candidate_entities.json.gz")
    parser.add_argument("--cache_save_dir",type=str,default="~/EncodedCache/Train/")

    args=parser.parse_args()

    main(args.example_filepath,args.context_filepath,args.cache_save_dir)
