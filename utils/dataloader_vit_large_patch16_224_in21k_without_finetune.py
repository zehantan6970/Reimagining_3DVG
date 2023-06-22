import os
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor
from transformers import AutoImageProcessor, ViTModel
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
image_encoder_model = "google/vit-large-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k").to(device)


def dataload(rootpath, scene_name=None):
    """
    Args:
        rootpath: 存放region与d3文件夹的根目录

    Returns:
        all_scene_patches: dict
        all_d3_patches: dict
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
            # 按照图片id从小到大排序
            in_listdirs.sort(key=lambda x: int(x.split('.')[0]))
            for in_dir in in_listdirs:
                image_path = os.path.join(os.path.join(train_region_floders_path, scene_dir), in_dir)
                image = Image.open(image_path)
                image_patches.append(image)
            # 批量处理1个region文件夹中的图片
            inputs = image_processor(image_patches, return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_patch_features = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
            all_scene_patches['{}{}{}'.format(scene_dir[5:9], scene_dir[10:12], scene_dir[13:])] = image_patch_features
            print("feature extractor complete {}/{}".format(i + 1, region_floders_listdirs_len))
            break
        if scene_name == None:
            image_patches = []
            in_listdirs = os.listdir(os.path.join(train_region_floders_path, scene_dir))
            # 按照图片id从小到大排序
            in_listdirs.sort(key=lambda x: int(x.split('.')[0]))
            for in_dir in in_listdirs:
                image_path = os.path.join(os.path.join(train_region_floders_path, scene_dir), in_dir)
                image = Image.open(image_path)
                image_patches.append(image)
            # 批量处理1个region文件夹中的图片
            inputs = image_processor(image_patches, return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_patch_features = outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()
            all_scene_patches['{}{}{}'.format(scene_dir[5:9], scene_dir[10:12], scene_dir[13:])] = image_patch_features
            print("feature extractor complete {}/{}".format(i + 1, region_floders_listdirs_len))
    # --------------------- #
    # 导入region 3d
    # --------------------- #
    all_d3_patches = {}
    d3_out_listdirs = os.listdir(train_d3_floders_path)
    for scene_dir in d3_out_listdirs:
        if scene_name == '{}{}{}'.format(scene_dir.split('.')[0][5:9], scene_dir.split('.')[0][10:12],
                                         scene_dir.split('.')[0][13:]):
            d3_txt = os.path.join(train_d3_floders_path, scene_dir)
            with open(d3_txt, mode="r") as r:
                lines = r.readlines()
                d3_lst = []
                for line in lines:
                    d3_lst.append(list(map(float, line.strip().split())))
                r.close()
            all_d3_patches['{}{}{}'.format(scene_dir.split('.')[0][5:9], scene_dir.split('.')[0][10:12],
                                           scene_dir.split('.')[0][13:])] = d3_lst
            break
        if scene_name == None:
            d3_txt = os.path.join(train_d3_floders_path, scene_dir)
            with open(d3_txt, mode="r") as r:
                lines = r.readlines()
                d3_lst = []
                for line in lines:
                    d3_lst.append(list(map(float, line.strip().split())))
                r.close()
            all_d3_patches['{}{}{}'.format(scene_dir.split('.')[0][5:9], scene_dir.split('.')[0][10:12],
                                           scene_dir.split('.')[0][13:])] = d3_lst
    return all_scene_patches, all_d3_patches
