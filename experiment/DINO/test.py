from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import platform
import time
import openpyxl
import os
import sys



model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
changjing=[]
object_id=[]
describe=[]
wb = openpyxl.load_workbook('/home/wzh/gree/GroundingDINO-main/test3.xlsx')
sheet = wb['sheet1']
for cell1 in list(sheet.columns)[1]:
    cell=cell1.value.replace(" ","")
    changjing.append(cell)
print(changjing)
for cell2 in list(sheet.columns)[3]:
    object_id.append(cell2.value)
print(object_id)
for cell3 in list(sheet.columns)[9]:
    describe.append(cell3.value)
print(describe)
curPath = "/home/wzh/gree/GroundingDINO-main/scannet_gt/"




BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

T1=time.perf_counter()
for h in range (0,len(changjing)):
    img=changjing[h]
    scene = curPath+img[0:12]+"/color/"
    IMAGE_PATH = scene+img[-6::]+".jpg"
    image_source, image = load_image(IMAGE_PATH)
    TEXT_PROMPT = str(describe[h])+"."
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("./test_result3/"+str(h)+"_"+img+".jpg", annotated_frame)
print("检测完成")
T2=time.perf_counter()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))

