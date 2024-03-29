import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')

from utils.utils import bbox_wh_iou, bbox_iou, to_cpu

class YoloLayer(nn.Module):
    """Yolo layer"""

    def __init__(self, anchors, num_classes, stride, ignore_thresh):
        super(YoloLayer, self).__init__()
        # Update the attributions when parsing the cfg during create the darknet
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.stride = stride
        # self.scale_x_y = scale_x_y
        # self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCELoss()
        self.ignore_thres = ignore_thresh
        self.obj_scale = 1
        self.noobj_scale = 100
        self.lgiou_scale = 3.54
        
        self.grid_size = 0  # grid size
        self.img_size = 0
        self.metrics = {}
        self.num_b_b_attr = 5

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g, device=self.device, dtype=torch.float).repeat(g, 1).view([1, 1, g, g])
        self.grid_y = torch.arange(g, device=self.device, dtype=torch.float).repeat(g, 1).t().view([1, 1, g, g])
        self.scaled_anchors = torch.tensor(
            [(a_w / self.stride, a_h / self.stride) 
             for a_w, a_h in 
             self.anchors], device=self.device, dtype=torch.float)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
    def build_targets(self, pred_boxes, pred_cls, target, anchors, ignore_thres):
        """ Built yolo targets to compute loss
        :param out_boxes: [num_samples or batch, num_anchors, grid_size, grid_size, 6]
        :param pred_cls: [num_samples or batch, num_anchors, grid_size, grid_size, num_classes]
        :param target: [num_boxes, 6]
        :param anchors: [num_anchors, 4]
        :return:
        """
        
        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)
        
        n_target_boxes = target.size(0)

        # Create output tensors on "device"
        obj_mask   = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.uint8)
        noobj_mask = torch.full(size=(nB, nA, nG, nG), fill_value=1, device=self.device, dtype=torch.uint8)
        class_mask = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        iou_scores = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tx   = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        ty   = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tw   = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        th   = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tcls = torch.full(size=(nB, nA, nG, nG, nC), fill_value=0, device=self.device, dtype=torch.float)
        giou_loss = torch.tensor([0.], device=self.device, dtype=torch.float)

        if n_target_boxes > 0:  # Make sure that there is at least 1 box
            # Convert to position relative to box
            target_boxes = target[:, 2:6] * nG
            
            gxy = target_boxes[:, :2]
            gwh = target_boxes[:, 2:]

            # Get anchors with best iou
            ious_a_tg = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
            
            best_ious, best_n = ious_a_tg.max(0)
            b, target_labels = target[:, :2].long().t()

            gx, gy = gxy.t()
            gw, gh = gwh.t()
            gi, gj = gxy.long().t()

            # Set masks
            obj_mask[  b, best_n, gj, gi] = 1
            noobj_mask[b, best_n, gj, gi] = 0

            # Set noobj mask to zero where iou exceeds ignore threshold
            for i, anchor_ious in enumerate(ious_a_tg.t()):
                noobj_mask[b[i], anchor_ious > self.ignore_thres, gj[i], gi[i]] = 0

            # Coordinates
            tx[b, best_n, gj, gi] = gx - gx.floor()
            ty[b, best_n, gj, gi] = gy - gy.floor()
            # Width and height
            tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
            th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

            # One-hot encoding of label
            tcls[b, best_n, gj, gi, target_labels] = 1
            
            # Compute label correctness and iou at best anchor
            class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
            iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

            # tconf = obj_mask.float()
            
        # return iou_scores, giou_loss, class_mask, obj_mask.type(torch.bool), noobj_mask.type(torch.bool), \
        tconf      = obj_mask.float()
        obj_mask   = obj_mask.type(torch.bool)
        noobj_mask = noobj_mask.type(torch.bool)

        return iou_scores, class_mask, obj_mask, noobj_mask, \
                tx, ty, tw, th, tcls, tconf

    def forward(self, x, targets=None, img_size=608, use_giou_loss=False):
        """
        :param x: [num_samples or batch, num_anchors * (self.num_b_b_attr + num_classes), grid_size, grid_size]
        :param targets: [num boxes, 8] (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
        :param img_size: default 608
        :return:
        """
        self.img_size = img_size
        self.use_giou_loss = use_giou_loss
        self.device = x.device
        
        num_samples   = x.size(0)
        grid_size     = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + self.num_b_b_attr, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        pred_x    = torch.sigmoid(prediction[..., 0])                    # Center x
        pred_y    = torch.sigmoid(prediction[..., 1])                    # Center y
        pred_w    = prediction[..., 2]                                   # Width
        pred_h    = prediction[..., 3]                                   # Height
        pred_conf = torch.sigmoid(prediction[..., self.num_b_b_attr-1])  # Conf
        pred_cls  = torch.sigmoid(prediction[..., self.num_b_b_attr:])   # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        # pred_boxes size: [num_samples, num_anchors, grid_size, grid_size, self.num_b_b_attr-3]
        pred_boxes = torch.empty(prediction[..., :self.num_b_b_attr-1].shape, device=self.device, dtype=torch.float)
        pred_boxes[..., 0] = pred_x + self.grid_x
        pred_boxes[..., 1] = pred_y + self.grid_y
        pred_boxes[..., 2] = torch.exp(pred_w).clamp(max=1E3) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(pred_h).clamp(max=1E3) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),  # conf
                pred_cls.view(num_samples, -1, self.num_classes),  # classes
            ),
            dim=-1
        )

        if targets is None:
            return output, 0
        else:
            self.reduction = 'mean'
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            loss_x = F.mse_loss(pred_x[obj_mask], tx[obj_mask], reduction=self.reduction)
            loss_y = F.mse_loss(pred_y[obj_mask], ty[obj_mask], reduction=self.reduction)
            loss_w = F.mse_loss(pred_w[obj_mask], tw[obj_mask], reduction=self.reduction)
            loss_h = F.mse_loss(pred_h[obj_mask], th[obj_mask], reduction=self.reduction)
            
            loss_conf_obj = F.binary_cross_entropy(pred_conf[obj_mask], tconf[obj_mask], reduction=self.reduction)
            loss_conf_noobj = F.binary_cross_entropy(pred_conf[noobj_mask], tconf[noobj_mask], reduction=self.reduction)
            loss_cls = F.binary_cross_entropy(pred_cls[obj_mask], tcls[obj_mask], reduction=self.reduction)

            if self.use_giou_loss:
                loss_obj = loss_conf_obj + loss_conf_noobj
                total_loss = giou_loss * self.lgiou_scale  + loss_obj * self.lobj_scale + loss_cls * self.lcls_scale
            else:
                loss_obj = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                total_loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_cls

            # Metrics (store loss values using tensorboard)
            cls_acc    = 100 * class_mask[obj_mask].mean()
            conf_obj   = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50  = (iou_scores > 0.5).float()
            iou75  = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50  = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75  = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "loss_x": to_cpu(loss_x).item(),
                "loss_y": to_cpu(loss_y).item(),
                "loss_w": to_cpu(loss_w).item(),
                "loss_h": to_cpu(loss_h).item(),
                "loss_obj": to_cpu(loss_obj).item(),
                "loss_cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss

