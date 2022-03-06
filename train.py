from terminaltables import AsciiTable

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.train_utils import *
from torch.autograd import Variable
import torch.optim as optim
from eval_mAP import evaluate_mAP

from models.models import *
# from model.yolov3 import *

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pretrained_path', type=str, default="./checkpoints/Yolo_V3_coco.pth", help="if specified starts from checkpoint model")
    parser.add_argument('--pretrained_path', type=str, default="checkpoints/yolov4.weights", help="if specified starts from checkpoint model")
    parser.add_argument('--save_path', type=str, default="checkpoints/Yolo_V4_VOC_model.pth", help="if specified starts from checkpoint model")
    
    parser.add_argument('--working-dir' , type=str, default='./', metavar='PATH', help='The ROOT working directory')
    parser.add_argument('--num_epochs'  , type=int, default=4, help="number of epochs")
    parser.add_argument('--batch_size'  , type=int, default=2, help="size of each image batch")
    
    parser.add_argument('--gradient_accumulations', type=int, default=2, help="number of gradient accums before step")
    parser.add_argument('--img_size'    , type=int, default=416, help="size of each image dimension")
    parser.add_argument('--model_def'   , type=str,   default="config/yolov4_VOC.cfg", help="path to model definition file")
    parser.add_argument('--n_cpu', type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument('--evaluation_interval', type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument('--multiscale_training', default=True, help="allow for multi-scale training")
    parser.add_argument('--checkpoint_freq', type=int, default=2, metavar='N', help='frequency of saving checkpoints (default: 2)')
    # parser.add_argument('--data_config' , type=str,   default="config/custom.data", help="path to data config file")
    parser.add_argument('--data_config' , type=str,   default="config/VOC.data", help="path to data config file")
    # parser.add_argument('--data_config' , type=str,   default="config/coco.data", help="path to data config file")
    parser.add_argument('--use_giou_loss', action='store_true', help='If true, use GIoU loss during training. If false, use MSE loss for training')
    
    configs = parser.parse_args()
    print(configs)

    configs.iou_thres  = 0.5
    configs.conf_thres = 0.5
    configs.nms_thres  = 0.5
                
    ############## Dataset, logs, Checkpoints dir ######################
    
    configs.ckpt_dir    = os.path.join(configs.working_dir, 'checkpoints')
    configs.logs_dir    = os.path.join(configs.working_dir, 'logs')

    if not os.path.isdir(configs.ckpt_dir):
        os.makedirs(configs.ckpt_dir)
    if not os.path.isdir(configs.logs_dir):
        os.makedirs(configs.logs_dir)

    ############## Hardware configurations #############################    
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initiate model
    # Darknet(args.image_size, num_classes)
    # model = Darknet(configs.img_size, num_classes=20).to(configs.device)
    model = Darknet(cfgfile=configs.model_def, use_giou_loss = configs.use_giou_loss)
    model = model.to(configs.device)
    # model.apply(weights_init_normal)
    model.print_network()
    
    # Get data configuration
    data_config = parse_data_config(configs.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    
    # If specified we start from checkpoint
    
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            model.load_state_dict(torch.load(configs.pretrained_path))
            print("Trained pytorch weight loaded!")
        else:
            model.load_darknet_weights(configs.pretrained_path)
            print("Darknet weight loaded!")
    class_names = load_classes(data_config["names"])
    
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "loss_x",
        "loss_y",
        "loss_w",
        "loss_h",
        "loss_obj",
        "loss_cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    
    # learning rate scheduler config
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # Create dataloader
    # dataset = ListDataset(train_path, augment=True, multiscale=configs.multiscale_training)
    dataset = ListDataset(valid_path, augment=False, multiscale=False)

    train_dataloader = DataLoader(
        dataset,
        configs.batch_size,
        shuffle=True,
        num_workers=configs.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    max_mAP = 0.0
    for epoch in range(0, configs.num_epochs, 1):

        num_iters_per_epoch = len(train_dataloader)        

        # print(num_iters_per_epoch)

        # switch to train mode
        model.train()
        start_time = time.time()
        
        epoch_loss = 0
        # Training        
        for batch_idx, batch_data in enumerate(tqdm.tqdm(train_dataloader)):
            """
            # print(batch_data)
            
            print(batch_data[0])
            print(batch_data[1])
            print(batch_data[1].shape)
            print(batch_data[2])
            
            imgs = batch_data[1]
            
            from PIL import Image
            import numpy as np

            w, h = imgs[0].shape[1], imgs[0].shape[2]
            src = imgs[0]
            # data = np.zeros((h, w, 3), dtype=np.uint8)
            # data[256, 256] = [255, 0, 0]
            
            data = np.zeros((h, w, 3), dtype=np.uint8)
            data[:,:,0] = src[0,:,:]*255
            data[:,:,1] = src[1,:,:]*255
            data[:,:,2] = src[2,:,:]*255
            # img = Image.fromarray(data, 'RGB')
            img = Image.fromarray(data)
            img.save('my_img.png')
            img.show()

            import sys
            sys.exit()
            """
            
            # data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            global_step = num_iters_per_epoch * epoch + batch_idx + 1
            
            targets = Variable(targets.to(configs.device), requires_grad=False)
            imgs = Variable(imgs.to(configs.device))

            total_loss, outputs = model(imgs, targets)
            
            epoch_loss += float(total_loss.item())
            # compute gradient and perform backpropagation
            total_loss.backward()

            if global_step % configs.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                # Adjust learning rate
                lr_scheduler.step()

                # zero the parameter gradients
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            """
            if (batch_idx+1)%int((len(train_dataloader)/8)) == 0:

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % ((epoch+1), configs.num_epochs, (batch_idx+1), len(train_dataloader))

                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", total_loss.item())]
                    # logger.list_of_scalars_summary(tensorboard_log, global_step)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {total_loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(train_dataloader) - (batch_idx + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_idx + 1))
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

            # model.seen += imgs.size(0)
        """
        crnt_epoch_loss = epoch_loss/num_iters_per_epoch
        
        torch.save(model.state_dict(), configs.save_path)
        # global_epoch += 1
        
        # print("Global_epoch :",global_epoch, "Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        print(epoch+1,"epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        

        # Evaulation        
        #-------------------------------------------------------------------------------------
        """        
        if (epoch+1)%8 == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate_mAP(model, valid_path, configs,
                batch_size=4)

            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

            max_mAP = AP.mean()
            #-------------------------------------------------------------------------------------
        # Save checkpoint
        if (epoch+1) % configs.checkpoint_freq == 0:
            torch.save(model.state_dict(), configs.save_path)
            print('save a checkpoint at {}'.format(configs.save_path))
        """            
            
if __name__ == '__main__':
    main()
    