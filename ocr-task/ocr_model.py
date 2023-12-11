import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
模型采样gru+cnn+ctc损失函数
ctc损失函数：不用保证输入输出对齐，适用于实体识别，语音识别
ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')
black: 空白标签所在的label,默认为0
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
log_probs=[seq_len,batch_size,vocab]
targets=[batch_size,seq_len]
input_lengths=[batch_size]
target_lengths=[batch_size]
预测时 输出最大长度，需要解码。
训练输入输出： 输入：图片特征  输出：预测文本
'''


class OCRModel(nn.Module):
    def __init__(self,features,in_channels,out_features,num_classes,num_layers):
        super(OCRModel,self).__init__()
        self.features=features
        self.in_channels=in_channels
        self.out_features=out_features
        self.num_classes=num_classes
        self.num_layers=num_layers

        '''卷积层'''
        self.conv=self.conv_layers(features)
        '''全连接层'''
        self.linear=self.linear_layer()
        '''lstm层'''
        self.gru=nn.GRU(input_size=self.out_features*2,hidden_size=self.out_features,num_layers=num_layers,
                        batch_first=True,bidirectional=True,dropout=0.2)

        '''全连接层'''
        self.cls=nn.Linear(in_features=2*out_features,out_features=self.num_classes)

        '''损失函数层'''
        self.criterion=nn.CTCLoss(blank=0,zero_infinity=True,reduction="mean")


    def forward(self,x,target=None):
        batch_size,in_c,h,w=x.size()
        x=self.conv(x)       #[batch,1,64,128]->[batch,256,4,8]
        x=x.permute(0,3,1,2)
        x=x.view(batch_size,x.size(1),-1)
        x=self.linear(x)   #[batch,8,64]
        x,_=self.gru(x)
        x=self.cls(x)     #[batch,8,64]
        x=x.permute(1,0,2)

        if target!=None:
            loss=self.cal_loss(x,target)
            return x,loss
        elif not self.training:
            return F.log_softmax(x,dim=2).argmax(2).squeeze(1)

        return x,None

    def cal_loss(self,logits,target):
        input_len,batch,vocab_size=logits.size()
        logits_length=torch.full(size=(batch,),fill_value=input_len,dtype=torch.int32)
        logits=torch.log_softmax(logits,dim=2)
        targets,targets_len=target[0],target[1]
        loss=self.criterion(logits,targets,logits_length,targets_len)
        return loss

    def linear_layer(self):
        layers=nn.Sequential(nn.Linear(in_features=2048,out_features=2*self.out_features),
                             nn.BatchNorm1d(8),
                             nn.ReLU()

        )
        return layers

    def conv_layers(self,architetures):
        '''
        :param architetures:
        串联连接层， 输入维度[batch,1,64,128]
        args = [64, "M", 128, "M", 256, "M", 512, "M", 512]
        输出：[batch,512,4,8]
        '''
        layers=[]
        in_channels=1
        for args in architetures:
            if type(args)==int:
                layer=[nn.Conv2d(in_channels=in_channels,out_channels=args,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                       nn.BatchNorm2d(args),
                       nn.ReLU(),
                       nn.Dropout2d(0.1)
                       ]
                layers.extend(layer)
                in_channels=args
            elif args=="M":
                layer=[nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
                layers.extend(layer)
        return nn.Sequential(*layers)


    def decode(self,logits,id2char,pad_ids):
        '''对输出进行解码'''
        outstr=""
        logits=logits.numpy()
        out_best=[k for k,g in itertools.groupby(logits)]
        for w in out_best:
            if w!=pad_ids:
                outstr+=id2char[w]
        return outstr








if __name__ == '__main__':
    args = [64, "M", 128, "M", 256, "M", 512, "M", 512]
    num_classes = 35
    net = OCRModel(features=args, in_channels=1, num_classes=num_classes, out_features=32, num_layers=2)
    print(net)
    print(net(torch.rand(1, 1, 64, 128))[0].shape)  # [8, 1, num_classes]
