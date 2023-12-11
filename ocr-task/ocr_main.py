import numpy as np
import pandas as pd
pd.set_option("display.max_columns",None)
import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from ocr_model import OCRModel
import os
from torch.utils.data import Dataset,DataLoader
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A

transformer=A.Compose([A.Rotate(limit=5,p=0.5),A.Affine(shear=15,p=0.5)])   #图片增强

PAD_TOKENS="<PAD>"


class MyProcessor():
    def __init__(self,path):
        df=pd.read_csv(path)
        df["image_name"]=df["image_name"].apply(lambda f:'./resource/images/'+f)
        self.df=df

    def get_max_len(self):
        return max(list(self.df.len.values))

    def get_vocab(self,label_vocab):
        if os.path.exists(label_vocab):
            letter=[]
            with open(label_vocab,"r",encoding="utf8") as f:
                for w in f:
                    letter.append(w.strip())
            return letter
        char_vector=[char for tokens in list(self.df.loc[:,"class_name"].values) for char in tokens.strip().replace(" ","")]
        char=sorted(list(Counter(char_vector).keys()))
        letter=[PAD_TOKENS]+[w for w in char]
        with open(label_vocab,"w",encoding="utf8") as f:
            for w in letter:
                f.write(w+"\n")
        return letter




class MyDataset(Dataset):
    def __init__(self,dfs):
        super(MyDataset,self).__init__()
        self.dfs=dfs
        self.images=list(dfs.loc[:,"image_name"].values)
        self.target=list(dfs.loc[:,"class_name"].values)
        self.target_len=list(dfs.loc[:,"len"].values)
    def __len__(self):
        return len(self.dfs)

    def __getitem__(self, item):

        return self.images[item],self.target[item],self.target_len[item]


class MyCollator():
    def __init__(self,label_vocab,max_len,img_w,img_h,transformer):
        self.label_vocab={v:k for k,v in enumerate(label_vocab)}
        self.max_len=max_len
        self.img_w=img_w
        self.img_h=img_h
        self.transformer=transformer


    def __call__(self,data):
        image_name,class_name,len=zip(*data)
        images=[]
        targets=[]
        targets_len=[]
        num=0
        for w in image_name:
            num+=1
        for i in range(num):
            image_path=image_name[i]
            target=class_name[i]
            target_len=len[i]
            image=self.image_prepare(image_path) #读取图片
            image=self.prepare_input(image)
            image=np.expand_dims(image,-1)
            image=np.transpose(image,(2,0,1)).astype(np.float32)
            images.append(image)
            encode,length=self.text_to_labels(target)
            assert length==target_len
            targets.append(encode)
            targets_len.append(length)
        images=torch.tensor(images,dtype=torch.float32)
        targets=torch.tensor(targets,dtype=torch.long)
        targets_len=torch.tensor(targets_len,dtype=torch.int32)
        return images,targets,targets_len

    def text_to_labels(self,text):
        data=list(text)
        len_=len(text)
        data=[self.label_vocab[w] for w in data]
        while len(data)<self.max_len:
            data.append(self.label_vocab[PAD_TOKENS])
        return data,len_

    def prepare_input(self,image):
        image=image.astype(np.float32)
        image-=np.amin(image)
        image/=np.amax(image)
        return image


    def image_prepare(self,image_path):
        image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image,(self.img_w,self.img_h))
        if self.transformer:
            image=transformer(image=image)["image"]
        return image





def make_loader(collator,train_dfs,test_dfs,batch_size):
    train_loader=DataLoader(MyDataset(train_dfs),batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=collator)
    test_loader=DataLoader(MyDataset(test_dfs),batch_size=batch_size,shuffle=True,num_workers=4,collate_fn=collator)
    return train_loader,test_loader





def get_args():
    args=argparse.ArgumentParser(description="ocr")
    args.add_argument("--batch_size",default=32)
    args.add_argument("--path",default="./resource/images/annotations.csv")
    args.add_argument("--label_vocab",default="./resource/models/label_vocab.txt")
    args.add_argument("--max_len",default=8)
    args.add_argument("--num_classes",default=27)
    args.add_argument("--device",default=torch.device("cpu"))
    args.add_argument("--lr",default=0.001)
    args.add_argument("--writer",default="./resource/logs")
    args.add_argument("--log_step",default=50)
    args.add_argument("--epoch",default=200)
    args.add_argument("--save_path",default="./resource/models")
    return args.parse_args()





def main():
    args=get_args()
    myprocessor=MyProcessor(args.path)
    letters=myprocessor.get_vocab(args.label_vocab)
    args.num_classes=len(letters)
    args.max_len=myprocessor.get_max_len()

    train_dfs,test_dfs=train_test_split(myprocessor.df,test_size=0.2,random_state=17)
    train_mycollator=MyCollator(letters,args.max_len,img_w=128,img_h=64,transformer=True)
    test_mycollator=MyCollator(letters,args.max_len,img_w=128,img_h=64,transformer=False)
    train_loader=DataLoader(MyDataset(train_dfs),batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=train_mycollator)
    test_loader=DataLoader(MyDataset(test_dfs),batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=test_mycollator)
    features = [64, "M", 128, "M", 256, "M", 512, "M", 512]
    model=OCRModel(features=features,in_channels=1,out_features=32,num_classes=args.num_classes,num_layers=2)
    path = os.path.join(args.save_path, "model.pt")
    checkpoint_path = torch.load(path)
    model.load_state_dict(checkpoint_path)
    model.to(args.device)
    train(args,model,train_loader,test_loader)


def train(args,model,trainLoader,testLoader):
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.94,patience=2,verbose=True)
    writer=SummaryWriter(args.writer)
    step=0
    cur_loss=10000
    for epoch in range(args.epoch):
        model.train()
        train_loss=[]
        for id,data in enumerate(trainLoader):
            images,target,target_length=data
            images.to(args.device)
            target.to(args.device)
            target_length.to(args.device)
            x,loss=model(images,[target,target_length])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step+=1
            train_loss.append(loss.item())
            if id%args.log_step==0:
                print("epoch:",epoch,"/",args.epoch,"id:",id,"train loss:",loss.item())
                writer.add_scalar("train_loss",loss.item(),step)
                if loss.item()<cur_loss:
                    cur_loss=loss.item()
                    path=os.path.join(args.save_path,"model.pt")
                    torch.save(model.state_dict(),path)
        train_loss=np.average(train_loss)
        val_loss=valid(model,args,testLoader)
        print("epoch:",epoch,"loss:",train_loss,"val_loss:",val_loss)
        scheduler.step(val_loss)   # 损失函数递减
        if val_loss<=0 or train_loss<=0:break



def valid(model,args,testLoader):
    model.eval()
    with torch.no_grad():
        fin_loss=[]
        for id,data in enumerate(testLoader):
            images,targets,targets_len=data
            images.to(args.device)
            targets.to(args.device)
            targets_len.to(args.device)
            x,loss=model(images,[targets,targets_len])
            fin_loss.append(loss.item())
    loss=np.average(fin_loss)
    return loss


if __name__ == '__main__':
    main()










