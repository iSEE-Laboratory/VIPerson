import os
import argparse
from tqdm import tqdm
import sys

import torch
import torchvision.transforms as transforms
import torchvision

from PIL import Image
from sklearn.preprocessing import normalize

import numpy as np

from utils.iresnet import iresnet100
from utils.irse import IR_101

import torchvision

import inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, parent_dir)

import sys
sys.path.insert(0, 'ID_generator/') 

from utils.iresnet import iresnet100, iresnet50
from utils.irse import IR_101
from utils.synface_resnet import LResNet50E_IR
from utils.moco import MoCo


from models.clipreid.make_model_clipreid_text import make_model
def load_img_paths(datadir):
    """load num_imgs many FFHQ images"""
    img_files = sorted(os.listdir(datadir))
    '''
    img_paths = []
    import glob
    for f_name in img_files:
        img_paths = img_paths + glob.glob(os.path.join(datadir, f_name) + '/*.png')
    
    return img_paths # [os.path.join(datadir, f_name) for f_name in img_files if f_name.endswith(".jpg") or f_name.endswith(".png")]
    '''
    return [os.path.join(datadir, f_name) for f_name in img_files if f_name.endswith(".jpg") or f_name.endswith(".png")]
    

class ImageInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir):
        self.img_paths = load_img_paths(datadir)
        print("Number of images:", len(self.img_paths))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([256, 128]), # ([384, 128]), # odzu
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def load_elasticface(device):
    print("loading ElasticFace model...")
    ckpt = torch.load(os.path.join("utils", "Elastic_R100_295672backbone.pth"), map_location=device)
    backbone = iresnet100(num_features=512).to(device)
    backbone.load_state_dict(ckpt)
    return backbone

def load_curricularface(device):
    print("loading CurricularFace model...")
    backbone = IR_101([112, 112]).to(device)
    ckpt = torch.load(os.path.join("utils", "CurricularFace_Backbone.pth"), map_location=device)
    backbone.load_state_dict(ckpt)
    return backbone


def main(args):
    device = torch.device(0)
    bs = 1
    json_path = args.data_dir
    print("Dataset:", args.data_dir)
    if args.frm_name == "clipreid":
        from config_clip import cfg
        if args.config_file != "":
            cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        model = make_model(cfg, num_class=1000, camera_num=1, view_num = 1, json_path=json_path)
        model.to(device)
        
    else:
        raise NotImplementedError

    model.eval()
    
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Starting encoding ...")
    content = {}
    # =====================================
    batch = bs
    import json
     
    with open(json_path, 'r') as file:
        all_prompts = json.load(file)
    num_classes = len(all_prompts)
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    
    
    for i in range(i_ter):
        print(i)
        if i+1 != i_ter:
            l_list = torch.arange(i*batch, (i+1)* batch)
        else:
            l_list = torch.arange(i*batch, num_classes)
        
        emb_batch = model(label = l_list, get_text = True)
        emb_batch = torch.nn.functional.normalize(emb_batch).detach().cpu().numpy()
        filename_batch = []
        for label in l_list:
            filename_batch.append(all_prompts[label]["image"])
        for emb, filename in zip(emb_batch, filename_batch):
            content[filename.split(".")[0]] = emb
            # odzu
            torch.save(emb, os.path.join(args.out_dir, f"embeddings_{args.frm_name}" + "/" + filename.split(".")[0] + ".npy"))

    # =====================================
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--frm_name",
        type=str,
        default="clipreid", 
        help="[clipreid]",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default= './../gener_data/randgener_description/captionattrilist_20000id.json',
        help="path to data directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default='./../gener_data/randgener_description/',
        
        help="directory to save embedding file to"
    )
    
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--config_file", default="./models/clipreid/vit_base.yml", help="path to config file", type=str
    ) 
    
    
    args = parser.parse_args()
    main(args)
