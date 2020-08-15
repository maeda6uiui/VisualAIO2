import argparse
import logging
import os
from tqdm import tqdm
import pandas as pd
import torch
import torchvision

#Object detection
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import hashing

#Setup Detectron2
setup_logger()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
if torch.cuda.is_available()==False:
    cfg.MODEL.DEVICE="cpu"

predictor = DefaultPredictor(cfg)

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pred_regions_as_images(im_dir):
    """
    Returns predicted regions.

    Args:
        im_dir (str): Directory name of the image files
    
    Returns:
        [PIL.Image]: Predicted regions
    """
    regions=[]

    files = os.listdir(im_dir)
    for file in files:
        im_filepath=os.path.join(im_dir,file)

        try:
            image_pil=Image.open(im_filepath)
        except:
            logger.error("Image file open error: {}".format(im_filepath))
            continue

        image_cv2 = cv2.imread(im_filepath)
        outputs = predictor(image_cv2)

        pred_boxes_tmp = outputs["instances"].pred_boxes.tensor
        for i in range(pred_boxes_tmp.size()[0]):
            top_left_x=int(pred_boxes_tmp[i][0])
            top_left_y=int(pred_boxes_tmp[i][1])
            bottom_right_x=int(pred_boxes_tmp[i][2])
            bottom_right_y=int(pred_boxes_tmp[i][3])

            image_region=image_pil.crop((top_left_x,top_left_y,bottom_right_x,bottom_right_y))
            regions.append(image_region)

    return regions

def get_vgg16_output_from_region(region,vgg16):
    """
    Returns output from VGG-16.

    Args:
        region (PIL.Image): Region of an image
        vgg16 (torchvision.models.vgg16): VGG16 model

    Returns:
        torch.tensor: Output from VGG-16
    """
    region=region.convert("RGB")
    region_tensor = preprocess(region).unsqueeze(0).to(device)

    ret=vgg16(region_tensor).to(device)

    return ret

def get_vgg16_output_from_regions(regions,vgg16,feature_dim):
    """
    Returns output from VGG-16.

    Args:
        regions ([PIL.Image]): Regions of images
        vgg16 (torchvision.models.vgg16): VGG16 model
        feature_dim (int): Dimension of the image features

    Returns:
        torch.tensor: Output from VGG-16
    """
    ret=torch.empty(0,feature_dim,dtype=torch.float).to(device)

    for region in regions:
        tmp=get_vgg16_output_from_region(region,vgg16)
        ret=torch.cat([ret,tmp],dim=0)

    return ret

def main(im_base_dir,feature_dim,features_save_dir):
    """
    Main function

    Args:
        im_base_dir (str): Base directory of the image files
        feature_dim (int): Dimension of image features
        features_save_dir (str): Directory name to save the image features in
    """
    #Load the article list.
    article_list_filepath=os.path.join(im_base_dir,"article_list.txt")
    df = pd.read_table(article_list_filepath, header=None)

    articles={}
    for row in df.itertuples(name=None):
        article_name = row[1]
        dir_1 = row[2]
        dir_2 = row[3]

        article_hash=hashing.get_md5_hash(article_name)

        im_dir = os.path.join(im_base_dir,"Images",str(dir_1),str(dir_2))
        articles[article_hash]=im_dir

    #Create a directory to save the image features in.
    os.makedirs(features_save_dir,exist_ok=True)

    #Create a VGG16 model.
    vgg16=torchvision.models.vgg16(pretrained=True)
    vgg16.classifier[6]=torch.nn.Linear(4096,feature_dim)
    vgg16.to(device)
    vgg16.eval()

    #Create image features.
    for article_hash,im_dir in tqdm(articles.items()):
        regions=get_pred_regions_as_images(im_dir)
        features=get_vgg16_output_from_regions(regions,vgg16,feature_dim)

        features_save_filepath=os.path.join(features_save_dir,article_hash+".pt")
        torch.save(features,features_save_filepath)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="VisualAIO2")

    parser.add_argument("--im_base_dir",type=str,default="~/WikipediaImages")
    parser.add_argument("--feature_dim",type=int,default=768)
    parser.add_argument("--features_save_dir",type=str,default="~/VGG16Features")

    args=parser.parse_args()

    logger.info("im_base_dir: {}".format(args.im_base_dir))
    logger.info("feature_dim: {}".format(args.feature_dim))
    logger.info("features_save_dir: {}".format(args.features_save_dir))

    main(args.im_base_dir,args.feature_dim,args.features_save_dir)
