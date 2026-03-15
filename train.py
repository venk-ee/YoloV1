import torch
import numpy as np 
import torchvision.transforms as transforms
import torch.optim as optim 
import torchvision.transforms.functional as FT 
from tqdm import tqdm 
from torch.utils.data import DataLoader
from model import  YOLOV1
from utils import(intersection_over_union,non_max_suppression,mean_average_precision,cellboxes_to_boxes,get_bboxes,save_checkpoint,load_checkpoint,plot_image)
from loss import YOLOLOSS
from dataset import VOCDataset


seed=123
torch.manual_seed(seed)

#hyperparam
LEARNING_RATE=2e-5
DEVICE="cuda"if torch.cuda.is_available() else "cpu"
BATCH_SIZE=8
WEIGHT_DECAY=0
EPOCH=100
NUM_WORKERS=2
LOAD_MODEL=False
PIN_MEMORY=True
IMG_DIR="c:/Users/loll/Downloads/archive/images"
LABEL_DIR="c:/Users/loll/Downloads/archive/labels"
LOAD_MODEL_FILE=""

class Compose(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self,img,bboxes):
        for t in self.transforms:
            img,bboxes=t(img),bboxes

        return img,bboxes 

transforms=Compose([transforms.Resize((448,448)),transforms.ToTensor()])

def train_fn(train_loadder,model,optimizer,loss_fn):
    loop=tqdm(train_loadder,leave=True)
    mean_loss=[]

    for batch_idx,(x,y) in enumerate(loop):
        x,y=x.to(DEVICE),y.to(DEVICE)
        out=model(x)
        loss=loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())

    print(f"mean loss was {sum(mean_loss)/len(mean_loss)} ")


def main():
    model=YOLOV1(split_size=7,num_boxes=2,num_classes=20).to(DEVICE)

    optimizer=optim.Adam(
        model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY
    )

    loss_fn=YOLOLOSS()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE),model,optimizer)


    train_dataset=VOCDataset("c:/Users/loll/Downloads/archive/8examples.csv",transforms=transforms,image_dir=IMG_DIR,label_dir=LABEL_DIR)

    test_dataset=VOCDataset("c:/Users/loll/Downloads/archive/test.csv",transforms=transforms,image_dir=IMG_DIR,label_dir=LABEL_DIR)

    train_dataloader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=True,drop_last=False)


    test_dataloader=DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=True,drop_last=False)

    for epoch in range(EPOCH):
        pred_boxes,target_boxes=get_bboxes(train_dataloader,model,iou_threshold=0.5,threshold=0.4)

        mean_avg_prec=mean_average_precision(pred_boxes,target_boxes,iou_threshold=0.5,box_format="midpoint")

        print(f"Train map {mean_avg_prec}")

        train_fn(train_dataloader,model,optimizer,loss_fn)


if __name__ =="__main__":
    main()




