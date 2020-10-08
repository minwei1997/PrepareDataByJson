import argparse
import base64
import json
import os
import os.path as osp
import yaml

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils

import sys
parent_dir = osp.abspath(osp.join(os.getcwd(), ""))
js_data_path = osp.join(parent_dir, 'data\\js_data\\')


''' input exaple:
        json_file -> 'filename.json'
        out_dir -> abs path
'''
def js_to_output(json_file, out_dir=None):
    # logger.warning(
    #     "This script is aimed to demonstrate how to convert the "
    #     "JSON file to a single image dataset."
    # )
    # logger.warning(
    #     "It won't handle multiple JSON files to generate a "
    #     "real-use dataset."
    # )
    

    # output directory
    if out_dir is None:
        out_dir = osp.basename(json_file).replace(".", "_")
        out_dir = osp.join(js_data_path, 'js_output', out_dir)
    else:
        out_dir = out_dir

    # create folder if not exist
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    data = json.load(open(osp.join(js_data_path, 'js_file', json_file)))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
    )

    PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))
    utils.lblsave(osp.join(out_dir, "label.png"), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))

    with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        for lbl_name in label_names:
            f.write(lbl_name + "\n")


    info = dict(label_names=label_names)
    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)

    logger.info("Saved to: {}".format(out_dir))


if __name__ == "__main__":

    js_filename = '000001.json'
    js_to_output(js_filename)
