import torch.nn as nn
from net import vgg
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
from dataoperation import *

if __name__=="__main__":
    #---------------------------------#
    # Cuda       是否使用Cuda
    #            没有GPU可以设置成False
    #---------------------------------#
    Cuda = False
    # ---------------------------------#
    # 'vgg16' and  'vgg19'
    # ---------------------------------#
    Net = 'vgg16'
    # ---------------------------------#
    # 先运行annotation_txt脚本
    # ---------------------------------#
    annotation_path='class_data.txt'
    # ---------------------------------#
    # 输入图片尺寸
    # ---------------------------------#
    input_shape = [224, 224]
    # ---------------------------------#
    #  分类个数,比如这里只要猫和狗两类
    # ---------------------------------#
    num_classes = 2
    # -------------------------------------------------------#
    #   lr         模型的最大学习率
    #              当使用Adam优化器时建议设置  lr=5e-4
    #              当使用SGD优化器时建议设置   lr=7e-3
    # -------------------------------------------------------#
    lr = 0.0001
    # ---------------------------------#
    # 优化器选择 SGD 与 Adam
    # ---------------------------------#
    optimizer_type = "Adam"
    # ---------------------------------#
    # 验证集所占百分比
    # ---------------------------------#
    percentage = 0.2
    # ---------------------------------#
    # 训练轮次
    # ---------------------------------#
    epochs = 80
    # ---------------------------------#
    #   save_period 多少个epoch保存一次权值
    # ---------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'log'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))

    loss_history = LossHistory(log_dir=log_dir, model=Net, input_shape=input_shape)

    with open(annotation_path,'r') as f:
        lines=f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val=int(len(lines) * percentage)
    num_train=len(lines) - num_val


    train_data=DataGenerator(lines[:num_train],input_shape,True)
    val_data=DataGenerator(lines[num_train:],input_shape,False)
    val_len=len(val_data)
    print(val_len)

    gen_train=DataLoader(train_data,batch_size=4)
    gen_test=DataLoader(val_data,batch_size=4)

    device=torch.device('cuda'if torch.cuda.is_available() and Cuda else "cpu")
    net=vgg(mode=Net, pretrained=True, progress=True, num_classes=num_classes)
    net.to(device)
    if optimizer_type == 'Adam':
        optim = torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer_type == 'SGD':
        optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(optimizer_type))

    sculer=torch.optim.lr_scheduler.StepLR(optim,step_size=1)


    for epoch in range(epochs):
        total_train=0
        for data in tqdm(gen_train, desc=f"Epoch{epoch + 1}/Train"):
            img,label=data
            with torch.no_grad():
                img =img.to(device)
                label=label.to(device)
            optim.zero_grad()
            output=net(img)
            train_loss=nn.CrossEntropyLoss()(output,label).to(device)
            train_loss.backward()
            optim.step()
            total_train+=train_loss

        sculer.step()
        total_test=0
        total_accuracy=0

        for data in tqdm(gen_test, desc=f"Epoch{epoch + 1}/Test"):
            img,label =data
            with torch.no_grad():
                img=img.to(device)
                label=label.to(device)
                optim.zero_grad()
                out=net(img)
                test_loss=nn.CrossEntropyLoss()(out,label).to(device)
                total_test+=test_loss
                accuracy=((out.argmax(1)==label).sum()).clone().detach().cpu().numpy()
                total_accuracy += accuracy
        print("训练集上的损失：{}".format(total_train))
        print("测试集上的损失：{}".format(total_test))
        print("测试集上的精度：{:.1%}".format(total_accuracy/val_len))
        loss_history.append_loss(epoch + 1, total_train, total_test)
        if (epoch+1) % save_period == 0:
            modepath = os.path.join(log_dir,"DogandCat{}.pth".format(epoch+1))
            torch.save(net.state_dict(),modepath)
        print("模型已保存")

