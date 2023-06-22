import os
from PIL import Image
import clip
import torch
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("/home/light/.cache/clip/ViT-B-32.pt", device=device)
model, preprocess = clip.load("ViT-B/32", device=device)
def dataload(rootpath,scene_name=None):
    """
    Args:
        rootpath:

    Returns:
        all_scene_patches: dict
        all_d3_patches:
    """
    train_region_floders_path = os.path.join(rootpath, 'region')
    train_d3_floders_path = os.path.join(rootpath, 'd3')
    region_floders_listdirs = os.listdir(train_region_floders_path)
    # ----------------------- #
    # 导入region image
    # ----------------------- #
    all_scene_patches = {}
    region_floders_listdirs_len = len(region_floders_listdirs)
    for i, scene_dir in enumerate(region_floders_listdirs):
        if scene_name == '{}{}{}'.format(scene_dir[5:9], scene_dir[10:12], scene_dir[13:]):
            image_patches = []
            in_listdirs = os.listdir(os.path.join(train_region_floders_path, scene_dir))
            in_listdirs.sort(key=lambda x: int(x.split('.')[0]))
            for in_dir in in_listdirs:
                image_path = os.path.join(os.path.join(train_region_floders_path, scene_dir), in_dir)
                image = Image.open(image_path)
                image_processed = preprocess(image).to(device)
                image_patches.append(image_processed)
            # image_patch_features shape=(n,512)
            image_patch_features = model.encode_image(torch.stack(image_patches)).to(device).detach().cpu().numpy()
            all_scene_patches['{}{}{}'.format(scene_dir[5:9],scene_dir[10:12],scene_dir[13:])] = image_patch_features
            print("feature extractor complete {}/{}".format(i + 1, region_floders_listdirs_len))
            break
        if scene_name == None:
            image_patches = []
            in_listdirs = os.listdir(os.path.join(train_region_floders_path, scene_dir))
            in_listdirs.sort(key=lambda x: int(x.split('.')[0]))
            for in_dir in in_listdirs:
                image_path = os.path.join(os.path.join(train_region_floders_path, scene_dir), in_dir)
                image = Image.open(image_path)
                image_processed = preprocess(image).to(device)
                image_patches.append(image_processed)
            # image_patch_features shape=(n,512)
            image_patch_features = model.encode_image(torch.stack(image_patches)).to(device).detach().cpu().numpy()
            all_scene_patches['{}{}{}'.format(scene_dir[5:9], scene_dir[10:12], scene_dir[13:])] = image_patch_features
            print("feature extractor complete {}/{}".format(i + 1, region_floders_listdirs_len))
    all_d3_patches = {}
    # --------------------- #
    # 导入region 3d
    # --------------------- #
    d3_out_listdirs = os.listdir(train_d3_floders_path)
    for scene_dir in d3_out_listdirs:
        d3txt = os.path.join(train_d3_floders_path, scene_dir)
        with open(d3txt, mode="r") as r:
            lines = r.readlines()
            lst = []
            for line in lines:
                lst.append(list(map(float, line.strip().split())))
            r.close()
        all_d3_patches['{}{}{}'.format(scene_dir.split('.')[0][5:9],scene_dir.split('.')[0][10:12],scene_dir.split('.')[0][13:])] = lst
    return all_scene_patches, all_d3_patches
