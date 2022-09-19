from msilib.schema import Error
import torch
import numpy as np
import os
from tqdm import tqdm
from .faster_rcnn_renset_fpn import FasterRCNNSWinFPN
from .visualize import visualize
from .utils import ConsoleLog
from .dataset import AnnotationDataset
from torch.utils.data import DataLoader
from .device import device
from typing import TypedDict, Literal

console_log = ConsoleLog(lines_up_on_end=1)

class TrainingErrorMessage(TypedDict):
    curr_epoch: int
    message: Literal["nan_loss"]

def train(faster_rcnn: FasterRCNNSWinFPN, lr, start_epoch, epoches, save_weight_interval=5): 
    opt = torch.optim.Adam(faster_rcnn.parameters(), lr=lr)
    dataset = AnnotationDataset()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=1)
    
    for epoch in range(epoches):
        epoch = epoch + start_epoch
                
        for batch_id, (img, boxes, cls_indexes) in enumerate(tqdm(data_loader)):
            batch_id = batch_id+1
            
            rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = faster_rcnn(img, boxes[0], cls_indexes[0])
            total_loss = rpn_cls_loss + 10*rpn_reg_loss + roi_cls_loss + 40*roi_reg_loss
            
            if torch.isnan(total_loss):
                return TrainingErrorMessage(message="nan_loss", curr_epoch=epoch)
                
            opt.zero_grad()
            total_loss.backward()
            opt.step()
            
            with torch.no_grad():
                console_log.print([
                    ("total_loss", total_loss.item()),
                    ("-rpn_cls_loss", rpn_cls_loss.item()),
                    ("-rpn_reg_loss", rpn_reg_loss.item()),
                    ("-roi_cls_loss", roi_cls_loss.item()),
                    ("-roi_reg_loss", roi_reg_loss.item())
                ])
            
                if batch_id % 20 == 0:
                    visualize(faster_rcnn, f"{epoch}_batch_{batch_id}.jpg")

        if epoch % save_weight_interval == 0:
            state_dict = faster_rcnn.state_dict()
            torch.save(state_dict, os.path.join("pths", f"model_epoch_{epoch}.pth"))


def train_with_nan(
    faster_rcnn,
    lr=1e-5,
    start_epoch=6,
    epoches=11,
    save_weight_interval=1
    ):
    
    continue_training = True
    restart_ep = start_epoch
    restart_for_eps = epoches
    curr_model = faster_rcnn    
    
    while continue_training:
        result = train(
            curr_model,
            lr,
            restart_ep,
            restart_for_eps,
            save_weight_interval
        )
        if result is not None:
            message = result["message"]
            if message == "nan_loss":
                curr_epoch = result["curr_epoch"]
                if curr_epoch > (start_epoch + epoches):
                    print("stop training")
                    continue_training = False
                else:
                    continue_training = True
                    model_latest_epoch = (curr_epoch-1) - ((curr_epoch-1) % save_weight_interval)
                    restart_ep = model_latest_epoch + 1
                    restart_for_eps = epoches - (model_latest_epoch - start_epoch)
                    model_path = f"pths/model_epoch_{model_latest_epoch}.pth"
                    curr_model = FasterRCNNSWinFPN().to(device)
                    curr_model.load_state_dict(torch.load(model_path))  
                    curr_model.train()  
                    
                    print(f"Get nan loss, restart training at epoch {restart_ep} for additional {restart_for_eps} epochs")
            else:
                continue_training = False
        else:
            continue_training = False



if __name__ == "__main__":
    train()
    