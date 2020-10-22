import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from dataset import CityscapesTrainInform, cityscapesFineTest
from DeeplabV3Plus.DeeplabV3Plus import Deeplabv3plus
from DeeplabV3Plus.config import cfg
from tqdm import tqdm
from metric.SegmentationMetric import SegmentationMetric
from utlis import pad_image, save_predict
import math, numpy as np, os, json, pickle


def test(val_loader, model, tile_size):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)
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
        full_probs = np.asarray(np.argmax(full_probs, axis=2), dtype=np.uint8)
        # 設置輸出原圖和預測圖片的顏色灰度還是彩色
        if none_gt:
            gt = np.zeros((image_size[2], image_size[3]), dtype=np.float32)
        else:
            gt = gt[0].numpy()
            # 計算miou
            metric.addBatch(full_probs, gt)

        saveDir = os.path.join(save_seg_dir, name[0].split('_')[0])
        if not(os.path.isdir(saveDir)):
            os.makedirs(saveDir)
        save_predict(full_probs, gt, name[0], saveDir, output_grey=False, output_color=True, gt_color=True)

        if not(none_gt):
            pa = metric.pixelAccuracy()
            cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            Miou, PerMiou_set = metric.meanIntersectionOverUnion()
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()

            print('miou {}\nclass iou {}'.format(Miou, PerMiou_set))
            result = save_seg_dir + '/results.txt'
            with open(result, 'w') as f:
                f.write(str(Miou))
                f.write('\n{}'.format(PerMiou_set))


# ==========User Setup============
use_cuda = False # torch.cuda.is_available() # Can be set to False manual
num_workers = 4
num_classes = 19
tile_size = 1024
none_gt = True
savedir = 'D:/jupyter_code/pytorchCityscape/pytorch_ResneSt/save/'
checkpoint = ''
trainDictPath = 'D:/jupyter_code/pytorchCityscape/pytorch_ResneSt/trainDict.json'
testDictPath = 'D:/jupyter_code/pytorchCityscape/pytorch_ResneSt/testDict.json'
inform_data_file = './inform/cityscapes_imform.pkl'
# ==========User Setup============

save_dirname = checkpoint.split('/')[-2] + '_' + checkpoint.split('/')[-1].split('.')[0]
save_seg_dir = os.path.join(savedir, 'cityscapes', 'predict_sliding', save_dirname)

# set the device
device = torch.device("cuda" if use_cuda else "cpu")

# build the model
model = Deeplabv3plus(cfg, num_classes=num_classes).to(device)
if use_cuda:
    cudnn.benchmark = True

if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)

testDict = json.load(open(testDictPath))

if not os.path.isfile(inform_data_file):
    trainDict = json.load(open(trainDictPath))
    dataCollect = CityscapesTrainInform(trainDict, 19, inform_data_file=inform_data_file)
    datas = dataCollect.collectDataAndSave()
    if datas is None:
        print("error while pickling data. Please check.")
        exit(-1)
else:
    print("find file: ", str(inform_data_file))
    datas = pickle.load(open(inform_data_file, "rb"))

if none_gt:
    testLoader = data.DataLoader(
        cityscapesFineTest(testDict, mean=datas['mean']),
        batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
else:
    testLoader = data.DataLoader(
        cityscapesFineVal(testDict, mean=datas['mean']),
        batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

if os.path.isfile(checkpoint):
    print("loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model'])
else:
    print("no checkpoint found at '{}'".format(checkpoint))
    raise FileNotFoundError("no checkpoint found at '{}'".format(checkpoint))

print(">>>>>>>>>>>beginning testing>>>>>>>>>>>")
test(testLoader, model, tile_size=(tile_size, tile_size))