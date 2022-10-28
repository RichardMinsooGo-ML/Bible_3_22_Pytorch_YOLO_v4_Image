# python detect_images.py --model_def config/yolov3.cfg --save_path checkpoints/yolov3.weights --class_path data/COCO2017/coco.names
# python detect_images.py --model_def config/yolov3.cfg --save_path checkpoints/Yolo_V3_coco.pth --class_path data/COCO2017/coco.names

# python detect_images.py --model_def config/yolov3-tiny.cfg --save_path checkpoints/yolov3-tiny.weights --class_path data/COCO2017/coco.names
# python detect_images.py --model_def config/yolov3-tiny.cfg --save_path checkpoints/Yolo_V3_coco_tiny.pth --class_path data/COCO2017/coco.names

# python detect_images.py --model_def config/yolov3.cfg --save_path checkpoints/Yolo_V3_VOC.pth --class_path data/VOC2012/voc2012.names

# python detect_images.py --model_def config/yolov3-tiny.cfg --save_path checkpoints/Yolo_V3_VOC_tiny.pth --class_path data/VOC2012/voc2012.names

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from PIL import Image

import torch
from utils.utils import *
from utils.datasets import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

# from models.models import *
# from model.yolov3 import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from pathlib import Path
Path("pred_IMAGES/images").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder"   , type=str, default="data/custom/samples", help="path to dataset")
    # parser.add_argument("--image_folder"   , type=str, default="data/VOC2012/test_images"  , help="path to dataset")
    
    # Yolov3 : COCO
    parser.add_argument("--model_def"  , type=str,   default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--save_path"  , type=str,   default="checkpoints/Yolo_V3_coco.pth", help="path to weights file")
    parser.add_argument("--save_path"  , type=str,   default="checkpoints/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path" , type=str,   default="data/COCO2017/coco.names", help="path to class label file")
    
    """
    # Yolov3-tiny : COCO
    parser.add_argument("--model_def"  , type=str,   default="config/yolov3-tiny.cfg", help="path to model definition file")
    # parser.add_argument("--save_path"  , type=str,   default="checkpoints/Yolo_V3_coco_tiny.pth", help="path to weights file")
    parser.add_argument('--save_path'  , type=str,   default="checkpoints/yolov3-tiny.weights", help="path to weights file")
    parser.add_argument("--class_path" , type=str,   default="data/COCO2017/coco.names", help="path to class label file")
    
    # Yolov3 : VOC
    parser.add_argument("--model_def"  , type=str,   default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--save_path"  , type=str,   default="checkpoints/Yolo_V3_VOC.pth", help="path to weights file")
    parser.add_argument("--class_path" , type=str,   default="data/VOC2012/voc2012.names", help="path to class label file")
    
    # Yolov3-tiny : VOC
    parser.add_argument("--model_def"  , type=str,   default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--save_path"  , type=str,   default="checkpoints/Yolo_V3_VOC_tiny.pth", help="path to weights file")
    parser.add_argument("--class_path" , type=str,   default="data/VOC2012/voc2012.names", help="path to class label file")
    """
        
    parser.add_argument("--batch_size" , type=int  , default=4, help="size of each image batch")
    parser.add_argument("--n_cpu"      , type=int,   default=4, help="number of cpu threads to use during batch generation")
    
    parser.add_argument("--conf_thres"      , type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres"       , type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size"   , type=int,   default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str  , help="path to checkpoint model")
    configs = parser.parse_args()
    print(configs)

    ############## Hardware configurations #############################    
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initiate model
    # model.apply(weights_init_normal)
    
    if configs.save_path == "checkpoints/Yolo_V3_VOC.pth":
        from model.yolov3 import *
        model = Darknet(configs.img_size, num_classes=20).to(configs.device)
    else:
        from models.models import *
        model = Darknet(configs.model_def).to(configs.device)
    
    # model = Darknet(configs.img_size, num_classes=20)
    # model = Darknet(configs.model_def, img_size=configs.img_size)
    classes = load_classes(configs.class_path)

    # model.print_network()
    print("\n" + "___m__@@__m___" * 10 + "\n")
    
    print(configs.save_path)
    
    assert os.path.isfile(configs.save_path), "No file at {}".format(configs.save_path)

    # If specified we start from checkpoint
    if configs.save_path:
        if configs.save_path.endswith(".pth"):
            model.load_state_dict(torch.load(configs.save_path))
            print("Trained pytorch weight loaded!")
        else:
            model.load_darknet_weights(configs.save_path)
            print("Darknet weight loaded!")
    # torch.save(model.state_dict(), "checkpoints/Yolo_V3_coco.pth")
    # torch.save(model.state_dict(), "checkpoints/Yolo_V3_coco_tiny.pth")
    # sys.exit()
    
    # Eval mode
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configs.device = torch.device("cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx))
    # model = model.to(device = device)
    model.eval()
    os.makedirs("pred_IMAGES", exist_ok=True)

    dataloader = DataLoader(
        ImageFolder(configs.image_folder, img_size=configs.img_size),
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=configs.n_cpu,
    )


    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_prediction = []  # Stores prediction for each image index

    print("\nPerforming object detection:")
    start_time = time.time()
    for batch_idx, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        
        # print(input_imgs)
        # sys.exit()
        
        input_imgs = Variable(input_imgs.type(Tensor))
        
        # Get prediction 
        with torch.no_grad():
            prediction = model(input_imgs)
            prediction = non_max_suppression(prediction, configs.conf_thres, configs.nms_thres)

        # Log progress
        end_time = time.time()
        inference_time = datetime.timedelta(seconds=end_time - start_time)
        start_time = end_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_idx, inference_time))

        # Save image and prediction
        imgs.extend(img_paths)
        img_prediction.extend(prediction)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    
    # Iterate through images and save plot of prediction
    for img_i, (path, prediction) in enumerate(zip(imgs, img_prediction)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of prediction
        if prediction is not None:
            # Rescale boxes to original image
            prediction = rescale_boxes(prediction, configs.img_size, img.shape[:2])
            unique_labels = prediction[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in prediction:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1,y1,s=classes[int(cls_pred)],color="white",verticalalignment="top",bbox={"color": color, "pad": 0},
                )

        # Save generated image with prediction
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"pred_IMAGES/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
