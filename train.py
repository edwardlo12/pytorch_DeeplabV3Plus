import torch
from torchsummary import summary
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch import optim
from dataset import CityscapesTrainInform, cityscapesFineTrain, cityscapesFineVal
from utlis import setup_seed, init_weight, netParams, pad_image
from earlyStopping import EarlyStopping
from DeeplabV3Plus.DeeplabV3Plus import Deeplabv3plus
from DeeplabV3Plus.config import cfg
from loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d
from custom_optim import RAdam, Ranger, AdamW
from scheduler.lr_scheduler import WarmupPolyLR
from metric.SegmentationMetric import SegmentationMetric
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os, json, pickle, time, math


def train(train_loader, model, criterion, optimizer, epoch):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """

    model.train()
    epoch_loss = []

    total_batches = len(train_loader)
    st = time.time()
    pbar = tqdm(iterable=enumerate(train_loader), total=total_batches,
                desc='Epoch {}/{}'.format(epoch, maxEpoch))
    for iteration, batch in pbar:

        per_iter = total_batches
        max_iter = maxEpoch * per_iter
        cur_iter = epoch * per_iter + iteration
        # learming scheduling
        if lr_schedule == 'poly':
            lambda1 = lambda epoch: math.pow((1 - (cur_iter / max_iter)), poly_exp)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif lr_schedule == 'warmpoly':
            scheduler = WarmupPolyLR(optimizer, T_max=max_iter, cur_iter=cur_iter, warmup_factor=1.0 / 3,
                                     warmup_iters=warmup_iters, power=0.9)

        lr = optimizer.param_groups[0]['lr']

        images, labels, _ = batch

        images = images.cuda()
        labels = labels.long().cuda()
        # if model == 'PSPNet50':
        #     x, aux = model(images)
        #     main_loss = criterion(x, labels)
        #     aux_loss = criterion(aux, labels)
        #     loss = 0.6 * main_loss + 0.4 * aux_loss
        # else:
        output = model(images)
        if type(output) is tuple:
            output = output[0]
#         print(output.dtype, labels.dtype)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item())

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (maxEpoch - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def val(val_loader, criteria, model, tile_size):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    val_loss = []
    metric = SegmentationMetric(num_classes)
    pbar = tqdm(iterable=enumerate(val_loader), total=total_batches, desc='Val')
    for i, (input, gt, name) in pbar:
        image_size = input.shape  # (1,3,3328,3072)
        overlap = 1 / 3 # 每次滑動的覆蓋率為1/3
        # print(image_size, tile_size)
        stride = math.ceil(tile_size[0] * (1 - overlap)) # 滑動步長:512*(1-1/3) = 513 512*(1-1/3)= 342
        tile_rows = int(math.ceil((image_size[2] - tile_size[0]) / stride) + 1) # 行滑動步數:(3072-512)/342+1=9
        tile_cols = int(math.ceil((image_size[3] - tile_size[1]) / stride) + 1) # 列滑動步數:(3328-512)/342+1=10
        full_probs = np.zeros((image_size[2], image_size[3], num_classes)) # 初始化全概率矩陣shape(3072,3328,3)
        count_predictions = np.zeros((image_size[2], image_size[3], num_classes)) # 初始化計數矩陣shape(3072,3328,3)

        for row in range(tile_rows):  # row = 0,1     0,1,2,3,4,5,6,7,8
            for col in range(tile_cols):  # col = 0,1,2,3     0,1,2,3,4,5,6,7,8,9
                x1 = int(col * stride) # 起始位置x1 = 0 * 513 = 0 0*342
                y1 = int(row * stride) # y1 = 0 * 513 = 0 0*342
                x2 = min(x1 + tile_size[1], image_size[3]) # 末位置x2 = min(0+512, 3328)
                y2 = min(y1 + tile_size[0], image_size[2]) # y2 = min(0+512, 3072)
                x1 = max(int(x2 - tile_size[1]), 0) # 重新校準起始位置x1 = max(512-512, 0)
                y1 = max(int(y2 - tile_size[0]), 0) # y1 = max(512-512, 0)

                img = input[:, :, y1:y2, x1:x2] # 滑動窗口對應的圖像 imge[:, :, 0:512, 0:512]
                padded_img = pad_image(img, tile_size) # padding 確保扣下來的圖像為512*512
                # plt.imshow(padded_img)
                # plt.show()

                # 將扣下來的部分傳入網絡，網絡輸出概率圖。
                with torch.no_grad():
                    input_var = torch.from_numpy(padded_img).cuda().float()
                    padded_prediction = model(input_var)

                    if type(padded_prediction) is tuple:
                        padded_prediction = padded_prediction[0]

                    torch.cuda.synchronize()

                if isinstance(padded_prediction, list):
                    padded_prediction = padded_prediction[0]  # shape(1,3,512,512)

                padded_prediction = padded_prediction.cpu().data[0].numpy().transpose(1, 2, 0) # 通道位置變換(512,512,3)
                prediction = padded_prediction[0:img.shape[2], 0:img.shape[3], :] # 扣下相應面積 shape(512,512,3)
                count_predictions[y1:y2, x1:x2] += 1 # 窗口區域內的計數矩陣加1
                full_probs[y1:y2, x1:x2] += prediction # 窗口區域內的全概率矩陣疊加預測結果

        # average the predictions in the overlapping regions
        full_probs /= count_predictions  # 全概率矩陣 除以 計數矩陣 即得 平均概率
        full_probsTensor = full_probs.transpose(2,0,1)
        full_probsTensor = torch.tensor(full_probsTensor).reshape((1,) + full_probsTensor.shape).float().cuda()
        loss = criteria(full_probsTensor, gt.long().cuda())
        val_loss.append(loss)
        full_probs = np.asarray(np.argmax(full_probs, axis=2), dtype=np.uint8)
        # 設置輸出原圖和預測圖片的顏色灰度還是彩色

        gt = gt[0].numpy()
        # 計算miou
        metric.addBatch(full_probs, gt)

    val_loss = sum(val_loss) / len(val_loss)

    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    Miou, PerMiou_set = metric.meanIntersectionOverUnion()
    FWIoU = metric.Frequency_Weighted_Intersection_over_Union()

    return val_loss, FWIoU, Miou, PerMiou_set


# ==========User Setup============
use_cuda = torch.cuda.is_available() # Can be set to False manual
gpus = 0
randomSeed = 1
input_size = (512, 1024)
tile_size = 1024
ignore_label = 255
num_classes = 19
batch_size = 4
num_workers = 4
maxEpoch = 500
val_epochs = 10
baseLr = 5e-4
lr_schedule = 'warmpoly'
poly_exp = 0.9
warmup_iters = 500
lossFunc = 'LovaszSoftmax' # ohem, label_smoothing, LovaszSoftmax, focal
optimSelect = 'adam' # sgd, adam, radam, ranger, adamw
inform_data_file = './inform/cityscapes_imform.pkl'
trainDictPath = '/home/edward/test/pytorch_test/Cityscapes/pytorch_DeeplabV3Plus/trainDict.json'
valDictPath = '/home/edward/test/pytorch_test/Cityscapes/pytorch_DeeplabV3Plus/valDict.json'
savedir = '/home/edward/test/pytorch_test/Cityscapes/pytorch_DeeplabV3Plus/save/'
logFile = 'log.txt'
resume = '/home/edward/test/pytorch_test/Cityscapes/pytorch_DeeplabV3Plus/save/cityscapes/deeplabV3Plusbs4gpu1_train/model_299.pth'
#===========User Setup============

h, w = input_size
# set the seed
setup_seed(randomSeed)
cudnn.enabled = True

# set the device
device = torch.device("cuda" if use_cuda else "cpu")

# build the model and initialization
model = Deeplabv3plus(cfg, num_classes=num_classes).to(device)
init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')
print("computing network parameters and FLOPs")
total_paramters = netParams(model)
print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))
# summary(model,(3,512,1024),device='cpu')

# load data and data augmentation
trainDict = json.load(open(trainDictPath))
valDict = json.load(open(valDictPath))

if not os.path.isfile(inform_data_file):
    dataCollect = CityscapesTrainInform(trainDict, 19, inform_data_file=inform_data_file)
    datas = dataCollect.collectDataAndSave()
    if datas is None:
        print("error while pickling data. Please check.")
        exit(-1)
else:
    print("find file: ", str(inform_data_file))
    datas = pickle.load(open(inform_data_file, "rb"))

trainLoader = data.DataLoader(
    cityscapesFineTrain(trainDict, crop_size=input_size, mean=datas['mean']),
    batch_size=batch_size, shuffle=True, num_workers=num_workers,
    pin_memory=True, drop_last=True)

valLoader = data.DataLoader(
    cityscapesFineVal(valDict, mean=datas['mean']),
    batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
    drop_last=True)

perEpooch = len(trainLoader)
maxIter = maxEpoch * perEpooch
print('Dataset statistics')
print("data['classWeights']: ", datas['classWeights'])
print('mean and std: ', datas['mean'], datas['std'])

# define loss function, respectively
weight = torch.from_numpy(datas['classWeights'])

if(lossFunc == 'ohem'):
    min_kept = int(batch_size // len(gpus) * h * w // 16)
    criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
elif(lossFunc == 'label_smoothing'):
    criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
elif(lossFunc == 'LovaszSoftmax'):
    criteria = LovaszSoftmax(ignore_index=ignore_label)
elif(lossFunc == 'focal'):
    criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)
else:
    raise NotImplementedError('We only support ohem, label_smoothing, LovaszSoftmax and focal as loss function.')

if use_cuda:
    criteria = criteria.cuda()
    if torch.cuda.device_count() > 1:
        print("torch.cuda.device_count()=", torch.cuda.device_count())
        gpu_nums = torch.cuda.device_count()
        model = nn.DataParallel(model).cuda()  # multi-card data parallel
    else:
        gpu_nums = 1
        print("single GPU for training")
        model = model.cuda()  # 1-card data parallel

    savedir = (savedir + 'cityscapes/deeplabV3Plusbs'
                    + str(batch_size) + 'gpu' + str(gpu_nums) + '_train/')
else:
    savedir = (savedir + 'cityscapes/deeplabV3Plusbs'
                    + str(batch_size) + 'cpu' + '_train/')

if not os.path.exists(savedir):
    os.makedirs(savedir)

with open(savedir + 'args.txt', 'w') as f:
        f.write('mean:{}\nstd:{}\n'.format(datas['mean'], datas['std']))
        f.write("Parameters: {} Seed: {}\n".format(str(total_paramters), randomSeed))

start_epoch = 0
# continue training
if resume:
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict(convert_state_dict(checkpoint['model']))
        print("loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    else:
        print("no checkpoint found at '{}'".format(resume))

model.train()
cudnn.benchmark = True # 尋找最優配置
cudnn.deterministic = True # 減少波動

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=50)

logFileLoc = savedir + logFile
if os.path.isfile(logFileLoc):
    logger = open(logFileLoc, 'a')
else:
    logger = open(logFileLoc, 'w')
    logger.write("%s\t%s\t\t%s\t%s\t%s\t%s\t%s\n" % ('Epoch', '   lr', 'Loss(Tr)', 'Loss(Val)', 'FWIOU(Val)', 'mIOU(Val)', 'Per Class IOU'))
logger.flush()

# define optimization strategy
if optimSelect == 'sgd':
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=baseLr, momentum=0.9, weight_decay=1e-4)
elif optimSelect == 'adam':
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=baseLr, betas=(0.9, 0.999), eps=1e-08,
        weight_decay=1e-4)
elif optimSelect == 'radam':
    optimizer = RAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=baseLr, betas=(0.90, 0.999), eps=1e-08,
        weight_decay=1e-4)
elif optimSelect == 'ranger':
    optimizer = Ranger(
        filter(lambda p: p.requires_grad, model.parameters()), lr=baseLr, betas=(0.95, 0.999), eps=1e-08,
        weight_decay=1e-4)
elif optimSelect == 'adamw':
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=baseLr, betas=(0.9, 0.999), eps=1e-08,
        weight_decay=1e-4)
else:
    raise NotImplementedError('We only support sgd, adam, radam, ranger and adamw as optimizer.')

lossTr_list = []
epoches = []
mIOU_val_list = []
lossVal_list = []
print('>>>>>>>>>>>beginning training>>>>>>>>>>>')
for epoch in range(start_epoch, maxEpoch):
    # training
    lossTr, lr = train(trainLoader, model, criteria, optimizer, epoch)
    lossTr_list.append(lossTr)
#     lossTr = 0
#     lr =0
#     lossTr_list.append(lossTr)

    # validation
    if epoch % val_epochs == 0 or epoch == maxEpoch-1:
        epoches.append(epoch)
        val_loss, FWIoU, mIOU_val, per_class_iu = val(valLoader, criteria, model,tile_size=(tile_size, tile_size))
        mIOU_val_list.append(mIOU_val)
        lossVal_list.append(val_loss.item())
        # record train information
        logger.write(
            "%d\t%.6f\t%.4f\t\t%.4f\t\t%0.4f\t\t%0.4f\t\t%s\n" % (epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, per_class_iu))
        logger.flush()
        print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\tVal Loss = %.4f\tFWIOU(val) = %.4f\tmIOU(val) = %.4f\tper_class_iu= %s\n" % (
            epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, str(per_class_iu)))
    else:
        # record train information
        logger.write("%d\t%.6f\t%.4f\n" % (epoch, lr, lossTr))
        logger.flush()
        print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\n" % (epoch, lr, lossTr))

    # save the model
    model_file_name = savedir + '/model_' + str(epoch) + '.pth'
    state = {"epoch": epoch, "model": model.state_dict()}

    # Individual Setting for save model
    if epoch >= maxEpoch - 10:
        torch.save(state, model_file_name)
    elif epoch % 10 == 0:
        torch.save(state, model_file_name)

    # draw plots for visualization
    if os.path.isfile(savedir + "loss.png"):
        f = open(savedir + 'log.txt', 'r')
        next(f)
        epoch_list = []
        epoch_listVal = []
        lossTr_list = []
        lossVal_list = []
        mIOU_val_list = []
        FWIOU_val_list = []
        for line in f.readlines():
            if(len(line.split())>3):
                # With IOU
                epoch_list.append(float(line.strip().split()[0]))
                lossTr_list.append(float(line.strip().split()[2]))
                epoch_listVal.append(float(line.strip().split()[0]))
                lossVal_list.append(float(line.strip().split()[3]))
                FWIOU_val_list.append(float(line.strip().split()[4]))
                mIOU_val_list.append(float(line.strip().split()[5]))
            else:
                # Without IOU
                epoch_list.append(float(line.strip().split()[0]))
                lossTr_list.append(float(line.strip().split()[2]))
        # assert len(epoch_list) == len(lossTr_list) == len(lossVal_list)

        fig1, ax1 = plt.subplots(figsize=(11, 8))

        ax1.plot(epoch_list, lossTr_list, label='Train_loss')
        ax1.plot(epoch_listVal, lossVal_list, label='Val_loss')
        ax1.set_title("Average training loss vs epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current loss")
        ax1.legend()

        plt.savefig(savedir + "loss.png")
        plt.close('all')
        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(11, 8))

        ax2.plot(epoch_listVal, mIOU_val_list, label="Val mean IoU")
        ax2.plot(epoch_listVal, FWIOU_val_list, label="Val frequency weighted IoU")
        ax2.set_title("Average IoU vs epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Current IoU")
        ax2.legend()

        plt.savefig(savedir + "mIou.png")
        plt.close('all')
    else:
        fig1, ax1 = plt.subplots(figsize=(11, 8))

        ax1.plot(range(0, epoch + 1), lossTr_list, label='Train_loss')
        ax1.plot(range(0, epoch + 1), lossVal_list, label='Val_loss')
        ax1.set_title("Average loss vs epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current loss")
        ax1.legend()

        plt.savefig(savedir + "loss.png")
        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(11, 8))

        ax2.plot(epoches, mIOU_val_list, label="Val IoU")
        ax2.set_title("Average IoU vs epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Current IoU")
        ax2.legend()

        plt.savefig(savedir + "mIou.png")
        plt.close('all')
    if epoch % val_epochs == 0 or epoch == maxEpoch-1:
        early_stopping.monitor(monitor=mIOU_val)
        if early_stopping.early_stop:
            print("Early stopping and Save checkpoint")
            if not os.path.exists(model_file_name):
                torch.save(state, model_file_name)
                val_loss, FWIoU, mIOU_val, per_class_iu = val(valLoader, criteria, model, tile_size=(tile_size, tile_size))
                print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\tVal Loss = %.4f\tmIOU(val) = %.4f\tper_class_iu= %s\n" % (
                        epoch, lr, lossTr, val_loss, mIOU_val, str(per_class_iu)))
            break

logger.close()