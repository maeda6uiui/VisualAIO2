import argparse
import json
import logging
import os
import sys

def main(json_filepath,save_filepath):
    with open(json_filepath, "r", encoding="UTF-8") as r:
        json_lines = r.read().splitlines()

    output_lines=[]
    for json_line in json_lines:
        data = json.loads(json_line)
        options = data["answer_candidates"]
        
        for option in options:
            output_lines.append(option)
            output_lines.append("\n")

        output_lines.append("\n")

    save_dir=os.path.dirname(save_filepath)
    os.makedirs(save_dir,exist_ok=True)

    with open(save_filepath,"w",encoding="UTF-8") as w:
        w.writelines(output_lines)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="VisualAIO2")

    parser.add_argument("--json_filepath",type=str,default="./Data/train_questions.json")
    parser.add_argument("--save_filepath",type=str,default="./Data/train_options_list.txt")

    args=parser.parse_args()

    main(args.json_filepath,args.save_filepath)
