# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import os
import numpy as np
import socket

from os import path
import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import pickle

from cutie.inference.data.vos_test_dataset import VOSTestDataset
from cutie.inference.data.burst_test_dataset import BURSTTestDataset
from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.results_utils import ResultSaver, make_zip
from cutie.inference.utils.burst_utils import BURSTResultHandler
from cutie.inference.utils.args_utils import get_dataset_cfg
from gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch, overlay_davis

class Segmenter():
    def __init__(self):
        FORMAT = '[%(filename)s:%(lineno)d] %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT)
        logging.info("Loading segmenter")
        with torch.inference_mode():
            initialize(version_base='1.3.2', config_path="cutie/config", job_name="eval_config")
            self.cfg = compose(config_name="eval_config")

            with open_dict(self.cfg):
                self.cfg['weights'] = './weights/cutie-base-mega.pth'

            data_cfg = get_dataset_cfg(self.cfg)

            # Load the network weights
            self.cutie = CUTIE(self.cfg).cuda().eval()
            self.model_weights = torch.load(self.cfg.weights)
            self.cutie.load_weights(self.model_weights)
            self.processor = InferenceCore(self.cutie, cfg=self.cfg)
        logging.info("Loading finished")
        
        self.host= "192.168.99.91"#'localhost'
        self.port= 15323
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen()

        self.is_first_mask_set = False

    def run(self, mask_file=None):
        return (cv2.imread(mask_file, -1)>0).astype(np.uint8)
    
    def run_server(self):
        while True:
            logging.info("Waiting for connection...")
            conn, addr = self.socket.accept()
            logging.info("connected with ", addr)
            self.is_first_mask_set = False
            self.img_count = 0
            with conn, conn.makefile('rb') as rfile:
                    while True:
                        try:
                            logging.info("waiting for packet...")
                            input_data = pickle.load(rfile)        
                        except EOFError: # when socket closes
                            break
                        ret_data = dict()
                        
                        if("mask" in input_data) and ( not self.is_first_mask_set):
                            mask_img = input_data["mask"]
                            color_img = input_data["rgb"]
                            self.setFirstMask(mask_img, color_img)
                            ret_data["mask"] = None
                            ret_data["success"] = True
                            self.is_first_mask_set = True
                        elif("rgb" in input_data):
                            color_img = input_data["rgb"]
                            mask_img = self.runFrame(color_img= color_img)
                            ret_data["mask"] = mask_img
                            ret_data["success"] = True
                        else:
                            ret_data["mask"] = None
                            ret_data["success"] = False
                        
                        logging.info(f"returning package with {ret_data['success']}")
                                                    
                        data = pickle.dumps(ret_data)
                        conn.send(data)




    def setFirstMask(self, first_mask, first_color_img):
        device = 'cuda'
        torch.cuda.empty_cache()
        first_mask = np.where(first_mask > 0, 1, 0)
        self.num_objects = len(np.unique(first_mask)) - 1
        mask_torch = index_numpy_to_one_hot_torch(first_mask, self.num_objects+1).to(device)
        # convert numpy array to pytorch tensor format
        frame_torch = image_to_torch(first_color_img, device=device)
        with torch.inference_mode():
            # the background mask is not fed into the model
            prediction = self.processor.step(frame_torch, mask_torch[1:], idx_mask=False)
        

    def runFrame(self, color_img):
        
        device = 'cuda'
        torch.cuda.empty_cache()
        mask_img = None
        with torch.inference_mode():

            # convert numpy array to pytorch tensor format
            frame_torch = image_to_torch(color_img, device=device)
            # propagate only
            prediction = self.processor.step(frame_torch)

            # argmax, convert to numpy
            mask_img = torch_prob_to_numpy_mask(prediction)
        return mask_img

if __name__ == "__main__":
    segmenter = Segmenter()
    segmenter.run_server()

    #manual inference
    first_mask_path = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/first_mask.png"
    first_color_path = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb/00000.png"
    first_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
    first_color = cv2.imread(first_color_path)
    segmenter.setFirstMask(first_mask, first_color)
    color_inf_img = cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb/00001.png")
    mask = segmenter.runFrame(color_inf_img)
    cv2.imwrite("out/mask_pred.jpg", mask * 255)