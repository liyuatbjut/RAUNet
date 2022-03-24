import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import FundusDataset
import cv2
import time
import copy
from torch.optim import lr_scheduler
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from attunet import AttUNet
from resunet import Resnet34_Unet
from resunets import ResNet, resnet34
from RAUNet import RAUNet, RAUNet34
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize



# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

#定义空列表
Loss_list_train = []
Accuracy_list_train = []
IOU_list_train = []
Loss_list_val = []
Accuracy_list_val = []
IOU_list_val = []

def performance_index_calculation_ACC(img1,img2):
    ret1, thresh1 = cv2.threshold(img1 * 255, 127, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img2 * 255, 50, 255, cv2.THRESH_BINARY)
    erosion1 = cv2.dilate(thresh1, kernel=np.ones((3, 3), np.uint8))
    dst1 = cv2.erode(erosion1, kernel=np.ones((3, 3), np.uint8)).astype(np.uint8)
    erosion2 = cv2.dilate(thresh2, kernel=np.ones((3, 3), np.uint8))
    dst2 = cv2.erode(erosion2, kernel=np.ones((3, 3), np.uint8)).astype(np.uint8)
    dstj = cv2.bitwise_and(dst1, dst2)
    area1 = areal = 0
    contours, hierarchy = cv2.findContours(dst1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for i in range(len(contours)):
            areal += cv2.contourArea(contours[i])
    contours1, hierarchy1 = cv2.findContours(dstj, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours1) != 0:
        for i in range(len(contours1)):
            area1 += cv2.contourArea(contours1[i])
    if (areal!= 0):
        ACC = area1/areal
    if (areal == 0):
        ACC = 1.0
    return ACC

def performance_index_calculation_IOU(img1,img2):
    ret1, thresh1 = cv2.threshold(img1 * 255, 127, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img2 * 255, 50, 255, cv2.THRESH_BINARY)
    erosion1 = cv2.dilate(thresh1, kernel=np.ones((3, 3), np.uint8))
    dst1 = cv2.erode(erosion1, kernel=np.ones((3, 3), np.uint8)).astype(np.uint8)
    erosion2 = cv2.dilate(thresh2, kernel=np.ones((3, 3), np.uint8))
    dst2 = cv2.erode(erosion2, kernel=np.ones((3, 3), np.uint8)).astype(np.uint8)
    dstb = cv2.bitwise_or(dst1, dst2)
    dstj = cv2.bitwise_and(dst1, dst2)
    area = area1 = 0
    contours, hierarchy = cv2.findContours(dstb, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        for i in range(len(contours)):
            area += cv2.contourArea(contours[i])
    contours1, hierarchy1 = cv2.findContours(dstj, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours1) != 0:
        for i in range(len(contours1)):
            area1 += cv2.contourArea(contours1[i])
    if (area!= 0):
        IOU = area1 / area
    if (area == 0):
        IOU = 1.0
    return IOU

def train_model(model, criterion, optimizer, dataloaders_t, dataloaders_v, scheduler, num_epochs=200):
    since = time.time()
    best_acc = 0.0
    best_iou = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            dt_size_t = len(dataloaders_t.dataset)
            dt_size_v = len(dataloaders_v.dataset)
            running_loss_t = 0.0
            running_corrects_t = 0.0
            running_IOU_t = 0.0
            running_loss_v = 0.0
            running_corrects_v = 0.0
            running_IOU_v = 0.0
            step = 0.0
            if (phase == 'train'):
               for x, y in dataloaders_t:
                   step += 1
                   inputs = x.to(device)
                   labels = y.to(device)
                   optimizer.zero_grad()
                   with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                   running_loss_t += loss.item()
                   label = torch.squeeze(y).numpy().transpose((1,2,0))
                   r,g,b = cv2.split(label)
                   prelabel = torch.squeeze(outputs.sigmoid()).cpu().detach().numpy().transpose((1,2,0))
                   r1, g1, b1 = cv2.split(prelabel)
                   running_corrects_t += performance_index_calculation_ACC(g, g1)
                   running_IOU_t += performance_index_calculation_IOU(g, g1)
               epoch_loss = 3 * running_loss_t / dt_size_t
               epoch_acc = 3 * running_corrects_t / dt_size_t
               epoch_IOU = 3 * running_IOU_t / dt_size_t
               Loss_list_train.append(epoch_loss)
               Accuracy_list_train.append(100 * epoch_acc)
               IOU_list_train.append(100 * epoch_IOU)
               print('{} Loss: {:.4f} Acc: {:.4f} IOU: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_IOU))
            if (phase == 'val'):
                for x, y in dataloaders_v:
                    #step += 1
                    inputs = x.to(device)
                    labels = y.to(device)
                    optimizer.zero_grad()
                    #with torch.no_grad():
                    with torch.set_grad_enabled(phase == 'train'):
                         outputs = model(inputs)
                         loss = criterion(outputs, labels)
                    running_loss_v += loss.item()
                    label = torch.squeeze(y).numpy().transpose((1, 2, 0))
                    r, g, b = cv2.split(label)
                    prelabel = torch.squeeze(outputs.sigmoid()).cpu().detach().numpy().transpose((1, 2, 0))
                    r1, g1, b1 = cv2.split(prelabel)
                    running_corrects_v += performance_index_calculation_ACC(g, g1)
                    running_IOU_v += performance_index_calculation_IOU(g, g1)
                epoch_loss = 3 * running_loss_v / dt_size_v
                epoch_acc = 3 * running_corrects_v / dt_size_v
                epoch_IOU = 3 * running_IOU_v / dt_size_v
                Loss_list_val.append(epoch_loss)
                Accuracy_list_val.append(100 * epoch_acc)
                IOU_list_val.append(100 * epoch_IOU)
                print('{} Loss: {:.4f} Acc: {:.4f} IOU: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_IOU))

            #if phase == 'val' and epoch > 5 and (epoch_acc + epoch_IOU) > (best_acc + best_iou):

            if phase == 'val' and epoch > 15 and epoch_IOU > best_iou:
                best_acc = epoch_acc
                best_iou = epoch_IOU
                best_model_wts = copy.deepcopy(model.state_dict())
                print(1)
        #torch.cuda.empty_cache()
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    torch.save(best_model_wts, 'weights_%d.pth' % epoch)
    x1 = range(0, 200)
    x2 = range(0, 200)
    x3 = range(0, 200)
    y1 = Accuracy_list_train
    y2 = Loss_list_train
    y3 = Accuracy_list_val
    y4 = Loss_list_val
    y5 = IOU_list_train
    y6 = IOU_list_val
    plt.subplot(3, 1, 2)
    plt.plot(x1, y1, 'b', label="train accuracy")
    plt.plot(x1, y3, 'r', label="val accuracy")
    plt.title('accuracy vs. epoches')
    plt.ylabel('accuracy')
    plt.subplot(3, 1, 1)
    plt.plot(x2, y2, 'b', label="train loss")
    plt.plot(x2, y4, 'r', label="val loss")
    plt.xlabel(' loss vs. epoches')
    plt.ylabel('loss')
    plt.subplot(3, 1, 3)
    plt.plot(x3, y5, 'b', label="train IOU")
    plt.plot(x3, y6, 'r', label="val IOU")
    plt.xlabel('IOU vs. epoches')
    plt.ylabel('IOU')
    plt.show()
    return model

#训练模型
def train(args):
    model = RAUNet34(3, 1, False).to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    fundus_dataset_t = FundusDataset("data/train",transform=x_transforms,target_transform=y_transforms)
    fundus_dataset_v = FundusDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders_t = DataLoader(fundus_dataset_t, batch_size=3, shuffle=True, num_workers=4)
    dataloaders_v = DataLoader(fundus_dataset_v, batch_size=3, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders_t, dataloaders_v, exp_lr_scheduler)

#显示模型的输出结果
def test(args):
    since = time.time()
    model = RAUNet34(3, 1, False)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    fundus_dataset = FundusDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(fundus_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    epoch = 0
    with torch.no_grad():
        for x, z in dataloaders:
            epoch+= 1
            y=model(x).sigmoid()
            img_y=torch.squeeze(y).numpy()
            img_y = cv2.cvtColor(np.asarray(img_y), cv2.COLOR_RGB2BGR)
            labels = torch.squeeze(z).numpy()
            r1, g1, b1 = cv2.split(img_y)
            ACC = performance_index_calculation_ACC(labels, g1)
            IOU = performance_index_calculation_IOU(labels, g1)
            ret1, thresh1 = cv2.threshold(g1*255, 50, 255, cv2.THRESH_BINARY)
            print(ACC,IOU)
        time_elapse = time.time() - since
        print(time_elapse)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
