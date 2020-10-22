from torch.utils.data.dataset import Dataset
import cv2,numpy as np, random, torch, pickle, os

class cityscapesFineTrain(Dataset):
    def __init__(self, imagePathDict, crop_size=(512, 1024), mean=(128, 128, 128), ignore_label=255):
        self.imagePathDict = imagePathDict
        self.imgList = list(self.imagePathDict.keys())
        self.mean = mean
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label

    def __getitem__(self,index):
        image = cv2.imread(self.imgList[index], cv2.IMREAD_COLOR)
        label = cv2.imread(self.imagePathDict[self.imgList[index]], cv2.IMREAD_GRAYSCALE)
        name = self.imgList[index].split('/')[-1].split('.')[0]
        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image[:,:,::-1]
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label
        
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))  # NHWC -> NCHW

        return torch.tensor(image.copy()), torch.tensor(label.copy()), name


    def __len__(self):
        return len(self.imgList)


class cityscapesFineVal(Dataset):
    def __init__(self, imagePathDict, mean=(128, 128, 128), ignore_label=255):
        self.imagePathDict = imagePathDict
        self.imgList = list(self.imagePathDict.keys())
        self.mean = mean
        self.ignore_label = ignore_label
    
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        image = cv2.imread(self.imgList[index], cv2.IMREAD_COLOR)
        label = cv2.imread(self.imagePathDict[self.imgList[index]], cv2.IMREAD_GRAYSCALE)
        name = self.imgList[index].split('/')[-1].split('.')[0]

        image = np.asarray(image, np.float32)

        image -= self.mean
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        return torch.tensor(image.copy()), torch.tensor(label.copy()), name


class cityscapesFineTest(Dataset):
    def __init__(self, imagePathDict, mean=(128, 128, 128), ignore_label=255):
        self.imagePathDict = imagePathDict
        self.imgList = list(self.imagePathDict.keys())
        self.mean = mean
        self.ignore_label = ignore_label
    
    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        image = cv2.imread(self.imgList[index], cv2.IMREAD_COLOR)
        name = self.imgList[index].split('/')[-1].split('.')[0]

        image = np.asarray(image, np.float32)

        image -= self.mean
        image = image[:, :, ::-1]  # change to RGB
        image = image.transpose((2, 0, 1))  # HWC -> CHW

        label = None

        return torch.tensor(image.copy()), label, name


class CityscapesTrainInform:
    """ To get statistical information about the train set, such as mean, std, class distribution.
        The class is employed for tackle class imbalance.
    """

    def __init__(self, imagePathDict, classes=19, inform_data_file="", normVal=1.10, ignore_label=255):
        """
        Args:
           data_dir: directory where the dataset is kept
           classes: number of classes in the dataset
           inform_data_file: location where cached file has to be stored
           normVal: normalization value, as defined in ERFNet paper
        """
        self.imagePathDict = imagePathDict
        self.imgList = list(self.imagePathDict.keys())
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.inform_data_file = inform_data_file
        self.ignore_label = ignore_label

    def compute_class_weights(self, histogram):
        """to compute the class weights
        Args:
            histogram: distribution of class samples
        """
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readWholeTrainSet(self, train_flag=True):
        """to read the whole train set of current dataset.
        Args:
        fileName: train set file that stores the image locations
        trainStg: if processing training or validation data
        
        return: 0 if successful
        """
        global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        min_val_al = 0
        max_val_al = 0
        for index in range(len(self.imgList)):
            # we expect the text file to contain the data in following format
            # <RGB Image> <Label Image>
            img_file = self.imgList[index]
            label_file = self.imagePathDict[self.imgList[index]]

            label_img = cv2.imread(label_file, 0)
            unique_values = np.unique(label_img)
            unique_values = np.delete(unique_values, np.where(unique_values == self.ignore_label))
            max_val = max(unique_values)
            min_val = min(unique_values)

            max_val_al = max(max_val, max_val_al)
            min_val_al = min(min_val, min_val_al)

            if train_flag == True:
                hist = np.histogram(label_img, self.classes, range=(0, 18))
                global_hist += hist[0]

                rgb_img = cv2.imread(img_file)
                self.mean[0] += np.mean(rgb_img[:, :, 0])
                self.mean[1] += np.mean(rgb_img[:, :, 1])
                self.mean[2] += np.mean(rgb_img[:, :, 2])

                self.std[0] += np.std(rgb_img[:, :, 0])
                self.std[1] += np.std(rgb_img[:, :, 1])
                self.std[2] += np.std(rgb_img[:, :, 2])

            else:
                print("we can only collect statistical information of train set, please check")

            if max_val > (self.classes - 1) or min_val < 0:
                print('Labels can take value between 0 and number of classes.')
                print('Some problem with labels. Please check. label_set:', unique_values)
                print('Label Image ID: ' + label_file)
            no_files += 1

        # divide the mean and std values by the sample space size
        self.mean /= no_files
        self.std /= no_files

        # compute the class imbalance information
        self.compute_class_weights(global_hist)
        return 0

    def collectDataAndSave(self):
        """ To collect statistical information of train set and then save it.
        The file train.txt should be inside the data directory.
        """
        print('Processing training data')
        return_val = self.readWholeTrainSet()

        print('Pickling data')
        if return_val == 0:
            data_dict = dict()
            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            if not(os.path.isdir(os.path.dirname(self.inform_data_file))):
                os.makedirs(os.path.dirname(self.inform_data_file))
            pickle.dump(data_dict, open(self.inform_data_file, "wb"))
            return data_dict
        return None