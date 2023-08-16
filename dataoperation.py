import cv2
import numpy as np
import torch.utils.data as data
import matplotlib
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os

def preprocess_input(x):
    x/=127.5
    x-=1.
    return x
def cvtColor(image):
    if len(np.shape(image))==3 and np.shape(image)[-2]==3:
        return image
    else:
        image=image.convert('RGB')
        return image


class DataGenerator(data.Dataset):
    def __init__(self,annotation_lines,inpt_shape,random=True):
        self.annotation_lines=annotation_lines
        self.input_shape=inpt_shape
        self.random=random

    def __len__(self):
        return len(self.annotation_lines)
    def __getitem__(self, index):
        annotation_path=self.annotation_lines[index].split(';')[1].split()[0]
        image=Image.open(annotation_path)
        image=self.get_random_data(image,self.input_shape,random=self.random)
        image=np.transpose(preprocess_input(np.array(image).astype(np.float32)),[2,0,1])
        y=int(self.annotation_lines[index].split(';')[0])
        return image,y
    def rand(self,a=0.,b=1.):
        return np.random.rand()*(b-a)+a

    def get_random_data(self,image,inpt_shape,jitter=.3,hue=.1,sat=1.5,val=1.5,random=True):

        image=cvtColor(image)
        iw,ih=image.size
        h,w=inpt_shape
        if not random:
            scale=min(w/iw,h/ih)
            nw=int(iw*scale)
            nh=int(ih*scale)
            dx=(w-nw)//2
            dy=(h-nh)//2

            image=image.resize((nw,nh),Image.BICUBIC)
            new_image=Image.new('RGB',(w,h),(128,128,128))

            new_image.paste(image,(dx,dy))
            image_data=np.array(new_image,np.float32)
            return image_data
        new_ar=w/h*self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale=self.rand(.75,1.25)
        if new_ar<1:
            nh=int(scale*h)
            nw=int(nh*new_ar)
        else:
            nw=int(scale*w)
            nh=int(nw/new_ar)
        image=image.resize((nw,nh),Image.BICUBIC)
        dx=int(self.rand(0,w-nw))
        dy=int(self.rand(0,h-nh))
        new_image=Image.new('RGB',(w,h),(128,128,128))
        new_image.paste(image,(dx,dy))
        image=new_image

        flip=self.rand()<.5
        if flip: image=image.transpose(Image.FLIP_LEFT_RIGHT)
        rotate=self.rand()<.5
        if rotate:
            angle=np.random.randint(-15,15)
            a,b=w/2,h/2
            M=cv2.getRotationMatrix2D((a,b),angle,1)
            image=cv2.warpAffine(np.array(image),M,(w,h),borderValue=[128,128,128])

        hue=self.rand(-hue,hue)
        sat=self.rand(1,sat) if self.rand()<.5 else 1/self.rand(1,sat)
        val=self.rand(1,val) if self.rand()<.5 else 1/self.rand(1,val)
        x=cv2.cvtColor(np.array(image,np.float32)/255,cv2.COLOR_RGB2HSV)#颜色空间转换
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data=cv2.cvtColor(x,cv2.COLOR_HSV2RGB)*255
        return image_data


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir,True)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        # plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        # plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        plt.plot(iters, [loss.item() for loss in self.losses], 'red', linewidth=2, label='train loss')
        plt.plot(iters, [loss.item() for loss in self.val_loss], 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")