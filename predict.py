"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision.transforms as T

import numpy as np 
import onnxruntime as ort 
from PIL import Image, ImageDraw
import os
import json
from tqdm import tqdm

def resize_bounding_box(bbox, original_size, target_size=(640, 640)):
    """
    將基於 target_size 的 bounding box 轉換回基於原始圖片大小 original_size 的座標。

    參數:
        bbox (tuple): (xmin, ymin, xmax, ymax) 基於 target_size 的 bounding box 座標
        original_size (tuple): (h, w) 原始圖片的尺寸
        target_size (tuple): 目標尺寸 (通常是 (640, 640))

    返回:
        tuple: 調整後基於原始圖片的 bounding box (xmin, ymin, xmax, ymax)
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size

    # 計算寬度和高度的縮放比例
    scale_w = orig_w / target_w
    scale_h = orig_h / target_h

    # 根據比例縮放 bounding box
    xmin, ymin, xmax, ymax = bbox
    xmin = int(xmin * scale_w)
    ymin = int(ymin * scale_h)
    xmax = int(xmax * scale_w)
    ymax = int(ymax * scale_h)

    return xmin, ymin, xmax, ymax


outputDICT = {}
def getBbox(images, labels, boxes, scores, filename, origin_size, thrh = 0.6):

    outputDICT[filename] = {"boxes":[], "labels" : []}

    for i, im in enumerate(images):
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for idx in range(len(lab)):
            outputDICT[filename]["boxes"].append( [float(item) for item in box[idx] ] )
            outputDICT[filename]["labels"].append( int(lab[idx]) )
        


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for b in box:
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[i].item()), fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    sess = ort.InferenceSession(args.onnx_file)
    print(ort.get_device())
    
    img_list = os.listdir(args.im_dir)
    cnt = 0
    for img in tqdm(img_list):
        if "." not in img:
            continue
        im_pil = Image.open(args.im_dir + "/" + img).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None]

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None]

        output = sess.run(
            # output_names=['labels', 'boxes', 'scores'],
            output_names=None,
            input_feed={'images': im_data.data.numpy(), "orig_target_sizes": orig_size.data.numpy()}
        )

        labels, boxes, scores = output
        getBbox([im_pil], labels, boxes, scores, img, (w, h))
        if cnt % 100 == 0:
            with open("predict.json", 'w') as json_file:
                json.dump(outputDICT, json_file, indent=4)
        cnt += 1

    with open("predict.json", 'w') as json_file:
        json.dump(outputDICT, json_file, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-file', type=str, )
    parser.add_argument('--im-dir', type=str, )
    # parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
