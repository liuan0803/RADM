
import numpy as np
from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
# from CLIP import clip
from PIL import Image
# from saliency_det import get_mask_np
from torchvision import transforms
import json
from PIL import ImageDraw
import cv2
from tqdm import tqdm
import time
import glob
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

color=['blue','green','red', 'yellow','black']


def cal_R_ove(test_annotation):
    overlap = []
    for layout in test_annotation:
        for lay in layout:
            if lay['category_id'] != 3 and lay['category_id'] != 4: # Exclude underlay and embellishment
                for l in layout:
                    if l['category_id'] != 3 and l['category_id'] != 4 and l!=lay: 
                            x1 = lay['bbox'][0]
                            x2 = lay['bbox'][0] + lay['bbox'][2]
                            y1 = lay['bbox'][1]
                            y2 = lay['bbox'][1] + lay['bbox'][3]
                            x3 = l['bbox'][0]
                            x4 = l['bbox'][0] + l['bbox'][2]
                            y3 = l['bbox'][1]
                            y4 = l['bbox'][1] + l['bbox'][3]
                            x_over = max(min(x2, x4) - max(x1, x3), 0)
                            y_over = max(min(y2, y4) - max(y1, y3), 0)
                            overlap.append(x_over * y_over / (lay['bbox'][2] * lay['bbox'][3])) # S_i intersect S_j / S_i    
    return sum(overlap) / len(overlap) if len(overlap)!= 0 else 0




def cal_R_ali(result, imageID2filename, imgdir):
    R_ali=[]
   
    for layout in tqdm(result):
        if len(layout)<=1:
            continue
        temp=[]
        image_path= os.path.join(imgdir, imageID2filename[layout[0]['image_id']])
        img=Image.open(image_path)
        w, h = img.size
        epsilon = 10e-7
        for i, lay in enumerate(layout):
            min_x, min_y = w, h
            
            for j in range(0, len(layout)):
                if i == j:
                    continue
                min_x = min(abs(layout[j]['bbox'][0] - lay['bbox'][0]), min_x, 
                          abs(layout[j]['bbox'][0] + layout[j]['bbox'][2]/2 - lay['bbox'][0] - lay['bbox'][2]/2),
                          abs(layout[j]['bbox'][0] + layout[j]['bbox'][2] - lay['bbox'][0] - lay['bbox'][2]))
                min_y = min(abs(layout[j]['bbox'][1] - lay['bbox'][1]), min_y, 
                          abs(layout[j]['bbox'][1] + layout[j]['bbox'][3]/2 - lay['bbox'][1] - lay['bbox'][3] / 2), 
                          abs(layout[j]['bbox'][1] + layout[j]['bbox'][3] - lay['bbox'][1] - lay['bbox'][3]))
            min_xl = -np.log(1.0 - min_x / w + epsilon)
            min_yl = -np.log(1.0 - min_y / h + epsilon)
            temp.append(min(min_xl, min_yl))
            
        if len(temp)!=0:
            R_ali.append(sum(temp)/len(temp))
    return sum(R_ali)/len(R_ali) 


def cal_R_und(test_annotation):

    underlay_over=[]
    for layout in test_annotation:
        for lay in layout:
            if lay['category_id'] == 3: # For underlay
                max_over = 0
                for l in layout:
                    if l['category_id'] != 3: #For all elements in other categories
                        x1 = l['bbox'][0]
                        x2 = l['bbox'][0] + l['bbox'][2]
                        y1 = l['bbox'][1]
                        y2 = l['bbox'][1] + l['bbox'][3]
                        x3 = lay['bbox'][0]
                        x4 = lay['bbox'][0] + lay['bbox'][2]
                        y3 = lay['bbox'][1]
                        y4 = lay['bbox'][1] + lay['bbox'][3]
                        x_over = max(min(x2, x4) - max(x1, x3), 0)
                        y_over = max(min(y2, y4) - max(y1, y3), 0)
                        over = x_over * y_over / (l['bbox'][2] * l['bbox'][3])
                        max_over = max(max_over,over)
                
                underlay_over.append(max_over)
    return sum(underlay_over) / len(underlay_over) if len(underlay_over) != 0 else 0
                    




def cal_R_occ(test_annotation, test_label):
    return len(test_annotation)/len(test_label)
    

def cal_R_com(test_annotation, imageID2filename, imgdir):
    def nn_conv2d(im, sobel_kernel):
        conv_op = nn.Conv2d(1, 1, 3, bias=False)
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
        gradient = conv_op(Variable(im))
        return gradient
    
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype='float32')
    sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype='float32')
    
    R_coms=[]
    for res in tqdm(test_annotation):
        for r in res:
            if r['category_id'] == 1: #only for text area
                text_area = r['bbox']
                image_path = os.path.join(imgdir, imageID2filename[r['image_id']])
                gray = Image.open(image_path).convert('L')

                x1 = int(text_area[0])
                y1 = int(text_area[1])
                w = int(text_area[2])
                h = int(text_area[3])
                gray = gray.crop((x1, y1, x1+w, y1+h))  
                gray_array = np.array(gray, dtype='float32')
                gray_tensor = torch.from_numpy(gray_array.reshape((1, 1, gray_array.shape[0], gray_array.shape[1])))
                
                try: #The w or h can be 0
                    image_x = nn_conv2d(gray_tensor, sobel_x)
                    image_y = nn_conv2d(gray_tensor, sobel_y)
                    image_xy = torch.mean(torch.sqrt(image_x ** 2 + image_y ** 2)).detach().numpy()
                    R_coms.append(image_xy)
                except:
                    continue

    return sum(R_coms)/len(R_coms) if len(R_coms)!=0 else 0


def visualize(test_annotation, imageID2filename, imgdir, outputdir):
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    for lay in tqdm(test_annotation):

        image_path= os.path.join(imgdir, imageID2filename[lay[0]['image_id']])
        img = Image.open(image_path)  

        draw = ImageDraw.ImageDraw(img)  
        for l in lay:
            x1,y1,w,h=int(l['bbox'][0]),int(l['bbox'][1]),int(l['bbox'][2]),int(l['bbox'][3])
            draw.rectangle(((x1, y1),(x1+w, y1+h)), fill=None, outline=color[l['category_id']-1], width=5)  
        img.save(os.path.join(outputdir,imageID2filename[lay[0]['image_id']]))


def main(test_label, test_annotation, imgdir, output_imgdir):
    with open(test_label) as f:
        input=json.load(f)
    with open(test_annotation) as f:
        test_result=json.load(f)
    res={}
    for ann in test_result:
        if ann['image_id'] not in res:
            res[ann['image_id']]=[ann]
        else:
            res[ann['image_id']].append(ann)

    res = list(res.values())       
    imageID2filename ={i : img['file_name'] for i, img in enumerate(input['images'])}
    
    visualize(res, imageID2filename, imgdir, output_imgdir) 
    print('R_occ: ', cal_R_occ(res, imageID2filename))
    print('R_com: ', cal_R_com(res, imageID2filename, imgdir))
    print('R_ove: ', cal_R_ove(res))
    print('R_und: ', cal_R_und(res))
    print('R_ali: ', cal_R_ali(res , imageID2filename, imgdir))


if __name__ == "__main__":
    '''
    test_label: json file, map from image ID to image file name
    root: {}
        images: []
            0: {}
                url:"100000002065_16_mask002.png"
                file_name:"100000002065_16_mask002.png"
                id:0
                width:350
                height:520
    
    test_annotation: json file, the generated layout.
    root: []
        image1: []
            bbox1: {}
                image_id:0
                bbox:[]
                category_id:2
                score:0.3739081919193268
    
    test_imgdir: 
    '''
    test_imgdir = '/home/lifengheng6/RADM_dataset/images/test'
    test_label = '/home/lifengheng6/RADM_dataset/annotations/test.json'
    test_annotation = '/home/lifengheng6/outputs_notebook2/inference/coco_instances_results.json'
    vis_example = './vis_example/'
    main(test_label, test_annotation, test_imgdir, vis_example)
    