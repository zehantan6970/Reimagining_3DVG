import os
from PIL import Image
import numpy as np
import open3d as o3d
import cv2 as cv
def uv2xyz(camera_inter,scalingFactor,depth,uv):
    """

    Args:
        camera_inter: 相机内参
        scalingFactor: 缩放因子
        depth: 深度图像，PIL
        uv: 像素坐标

    Returns:
        XYZ:相机坐标系下的xyz

    """
    fx ,fy,centerX,centerY=camera_inter
    # -----------------------------------------------------------------
    # 像素坐标u，v
    # -----------------------------------------------------------------
    u,v=uv

    # -----------------------------------------------------------------
    # 相机坐标X，Y，Z
    # -----------------------------------------------------------------
    Z = depth.getpixel((u, v)) / scalingFactor
    if Z == 0:
        return False
    else:
        X = (u - centerX) * Z / fx
        Y = (v - centerY) * Z / fy
        return [X,Y,Z]
def getRt(fragmentsNum,pose_matrix_path,start_ids=1):
    """
    Args:

        fragmentsNum: fragmentsNum: 取第n个rt（0，5，10，...）
        pose_matrix_path: pose文件的路径

    Returns:
        rt

    """
    with open(pose_matrix_path, mode="r") as r:
        lines = r.readlines()
        rt = []
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids].strip().split())))
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids+1].strip().split())))
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids+2].strip().split())))
        rt.append(list(map(float, lines[fragmentsNum * 5 + start_ids+3].strip().split())))
        r.close()
    return np.array(rt)
def transform(rt, points):
    """

    Args:
        rt: 变换矩阵
        points: 预转换的点云坐标

    Returns:
        new_points: 转换后的点云坐标

    """
    points = np.asarray(points)
    R = rt[:, :3][:3]
    T = rt[:, 3][:3]
    points_rt = np.dot(R, points.transpose((1, 0)))
    new_points = points_rt.transpose((1, 0)) + T
    return new_points
def get_obb(points):
    """

    Args:
        points: object点云的xyz坐标

    Returns:
        obb包围盒的中心坐标
        vertex_set: 8个顶点坐标

    """
    pcd = o3d.geometry.PointCloud()
    if len(points)<4:
        print(len(points))
        points=points.repeat(4,axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=200,
                                             std_ratio=0.2)
    try:
        pcd = pcd.select_by_index(ind)
        # 这句话偶尔会出现bug，如果出现bug就忽略掉这个region，然后返回None
        obb = pcd.get_oriented_bounding_box()
        [center_x, center_y, center_z] = obb.get_center()
        # obb包围盒的顶点
        vertex_set = np.asarray(obb.get_box_points())
        # print("obb包围盒的中心坐标为：\n", [center_x, center_y, center_z])
        # obb.color = (0, 1, 0)  # obb包围盒为绿色
        # o3d.visualization.draw_geometries([pcd, obb], window_name="OBB包围盒",
        #                                   width=1024, height=768,
        #                                   left=50, top=50,
        #                                   mesh_show_back_face=False)
        return [center_x, center_y, center_z], vertex_set
    except:
        return None, None
def get_box3d_min_max(corner):
    """

    Args:
        corner: numpy array (8,3), assume up direction is Z (batch of N samples)

    Returns:
        an array for min and max coordinates of 3D bounding box IoU

    """

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]

    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    """

    Args:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z

    Returns:
        iou: 3D bounding box IoU

    """
    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def get_iou_obb(bb1, bb2):
    iou3d = box3d_iou(bb1, bb2)
    return iou3d

def class_filter(lst):
    """

    Args:
        lst: object可能属于的类别

    Returns:
        frequency_sort[0][0]: 出现频率最高的类别

    """
    if not len(lst):
        return None
    count={}
    for word in lst:
        if word in count:
            count[word]+=1
        else:
            count[word]=1
    frequency_sort=sorted(count.items(),key=lambda x:x[1],reverse=True)
    return frequency_sort[0][0]
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "/" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件，就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)
def find_indexes(arr1, arr2):
    indexes = []
    for val in arr1:
        if val in arr2:
            indexes.append(np.where(arr2 == val)[0][0])
    return indexes
def image_preprocess(image, target_shape):
    """目标resize尺寸"""
    iw, ih = target_shape
    """原始图像尺寸"""
    w,h= image.size

    """计算缩放后图像的尺寸"""
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = image.resize((nw, nh))
    """
       创建一张画布，画布的尺寸就是目标尺寸
       fill_value=120为灰色画布 
    """
    image_paded = np.full(shape=[ih, iw, 3], fill_value=125)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    # print("dw,dh", dw, dh)
    """将缩放后的图片放在画布中央"""
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = (image_paded).astype(np.uint8)
    # image_paded=Image.fromarray(image_paded)
    return image_paded
import pandas as pd
train=pd.read_csv('/home/light/gree/slam/D3VG/datas/scannetv2-labels.combined.tsv', sep='\t', header=0, usecols=[4,7])
id_label={}
for i in train.values:
    id_label[i[0]]=i[1]
print(id_label)
import imageio
def generate_scannet_seg(instance_png,rgb_png):
    boxes_class, segs, boxes=[],[],[]
    image = Image.open(rgb_png)
    if image.size[0] != 1296:
        image = image.resize((1296, 968))
    mask_image = imageio.v2.imread(instance_png)
    mask_image = np.array(mask_image)
    # rt=getRt(0,pose_path,start_ids=0)
    for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40]:
        # for i in range(40):
        for j in range(5):
            label = (i) * 1000 + j
            xy = np.column_stack(np.where(mask_image == label))

            if len(xy)>100:
                image_arr = np.array(image)
                print(mask_image.shape)
                print(image_arr.shape)
                print(min(xy[:, 0]))
                for s in xy:
                    image_arr[s[0], s[1]] = 0
                image_pil = Image.fromarray(image_arr)
                image_pil.show()
                try:
                    u = np.array(list(map(int, (xy[:, 1] ))))[::10]
                    v = np.array(list(map(int, (xy[:, 0] ))))[::10]
                    uv = np.column_stack((u, v))
                    ymin, ymax = np.maximum(min(xy[:, 0]), 0), np.minimum(max(xy[:, 0]), image.size[1])
                    xmin, xmax = np.maximum(min(xy[:, 1]), 0), np.minimum(max(xy[:, 1]), image.size[0])
                except:
                    print("error")
                    break
                boxes_class.append(id_label[i])
                segs.append(uv)
                boxes.append([xmin,ymin,xmax,ymax])
    return boxes_class, segs, boxes

if __name__ =="__main__":
    generate_scannet_seg("/media/light/light_t2/t2/DATA/scannet_frames_25k/scene0000_00/instance/000000.png","/media/light/light_t2/t2/DATA/scannet_frames_25k/scene0000_00/color/000000.jpg")
