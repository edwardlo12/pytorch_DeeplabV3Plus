import torch
from torchsummary import summary
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data
from dataset import CityscapesTrainInform, cityscapesFineTrain, cityscapesFineVal
from utlis import setup_seed, init_weight
from DeeplabV3Plus.DeeplabV3Plus import Deeplabv3plus
from DeeplabV3Plus.config import cfg
from loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d
import os, json, pickle

# ==========User Setup============
use_cuda = False # torch.cuda.is_available() # Can be set to False manual
gpus = 0
randomSeed = 1
input_size = (512, 1024)
ignore_label = 255
num_classes = 19
batch_size = 4
num_workers = 4
maxEpoch = 30
lossFunc = 'label_smoothing' # ohem, label_smoothing, LovaszSoftmax, focal
inform_data_file = './inform/cityscapes_imform.pkl'
trainDictPath = 'D:/jupyter_code/pytorchCityscape/pytorch_ResneSt/trainDict.json'
valDictPath = 'D:/jupyter_code/pytorchCityscape/pytorch_ResneSt/valDict.json'
savedir = 'D:/jupyter_code/pytorchCityscape/pytorch_ResneSt/save/'
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
    raise NotImplementedError('We only support ohem, label_smoothing, LovaszSoftmax, focal.')

if args.cuda:
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