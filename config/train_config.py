import os
import argparse

import torch
from easydict import EasyDict as edict

def parse_train_configs():
    parser = argparse.ArgumentParser(description='The Implementation of Pyorch YOLOv3')
    
    """
    # Yolov3 : COCO
    parser.add_argument("--data_config" , type=str,   default="config/coco.data", help="path to data config file")
    parser.add_argument("--model_def"   , type=str,   default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/yolov3.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--save_path", type=str, default="checkpoints/Yolo_V3_coco.pth", help="if specified starts from checkpoint model")
        
    # Yolov3-tiny : COCO
    parser.add_argument("--data_config" , type=str,   default="config/coco.data", help="path to data config file")
    parser.add_argument("--model_def"   , type=str,   default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/yolov3-tiny.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--save_path", type=str, default="checkpoints/Yolo_V3_coco_tiny.pth", help="if specified starts from checkpoint model")
    
    # Yolov3 : VOC
    parser.add_argument("--data_config" , type=str,   default="config/VOC.data", help="path to data config file")
    parser.add_argument("--model_def"   , type=str,   default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--pretrained_path", type=str, default="checkpoints/yolov3.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/Yolo_V3_VOC.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--save_path", type=str, default="checkpoints/Yolo_V3_VOC.pth", help="if specified starts from checkpoint model")
    """
    # Yolov3-tiny : VOC
    parser.add_argument("--data_config" , type=str,   default="config/VOC.data", help="path to data config file")
    parser.add_argument("--model_def"   , type=str,   default="config/yolov3-tiny.cfg", help="path to model definition file")
    # parser.add_argument("--pretrained_path", type=str, default="checkpoints/yolov3-tiny.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/Yolo_V3_VOC_tiny.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--save_path", type=str, default="checkpoints/Yolo_V3_VOC_tiny.pth", help="if specified starts from checkpoint model")
    
    
    # parser.add_argument("--data_config" , type=str,   default="config/custom.data", help="path to data config file")
    parser.add_argument("--working-dir" , type=str, default='./', metavar='PATH', help='The ROOT working directory')
    parser.add_argument("--num_epochs"  , type=int, default=2, help="number of epochs")
    parser.add_argument("--batch_size"  , type=int, default=4, help="size of each image batch")
    parser.add_argument("--img_size"    , type=int, default=416, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--checkpoint_freq", type=int, default=2, metavar='N', help='frequency of saving checkpoints (default: 2)')

    configs = edict(vars(parser.parse_args()))
    
    configs.iou_thres  = 0.5
    configs.conf_thres = 0.5
    configs.nms_thres  = 0.5
                
    ############## Dataset, logs, Checkpoints dir ######################
    
    configs.ckpt_dir    = os.path.join(configs.working_dir, 'checkpoints')
    configs.logs_dir    = os.path.join(configs.working_dir, 'logs')

    print(configs)

    if not os.path.isdir(configs.ckpt_dir):
        os.makedirs(configs.ckpt_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    ############## Hardware configurations #############################    
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    return configs
