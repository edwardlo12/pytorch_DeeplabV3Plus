import os,json

imgRoot = '/home/edward/test/pytorch_test/Cityscapes/vision_datasets/cityscapes/leftImg8bit/'
gtRoot = '/home/edward/test/pytorch_test/Cityscapes/vision_datasets/cityscapes/gtFine/'


trainDict = {}
for fp,dl,fl in os.walk(os.path.join(imgRoot, 'train')):
    # print(fp,dl)
    if(len(dl) == 0):
        for imgPath in fl:
            # print(imgPath)
            search = os.path.join(fp.split(imgRoot)[-1],imgPath.split('_leftImg8bit.png')[0])
            # print(search)
            maskPath = os.path.join(gtRoot,search + '_gtFine_labelTrainIds.png')
            # print(maskPath)
            if(os.path.isfile(maskPath)):
                imgPath = os.path.join(fp,imgPath).replace('\\','/')
                maskPath = maskPath.replace('\\','/')
                # print(imgPath,maskPath)
                trainDict[imgPath] = maskPath
            else:
                print(search, 'Mask not found!')
with open('./trainDict.json','w') as f:
    json.dump(trainDict,f,indent=4)

valDict = {}
for fp,dl,fl in os.walk(os.path.join(imgRoot, 'val')):
    if(len(dl) == 0):
        for imgPath in fl:
            # print(imgPath)
            search = os.path.join(fp.split(imgRoot)[-1],imgPath.split('_leftImg8bit.png')[0])
            # print(search)
            maskPath = os.path.join(gtRoot,search + '_gtFine_labelTrainIds.png')
            # print(maskPath)
            if(os.path.isfile(maskPath)):
                imgPath = os.path.join(fp,imgPath).replace('\\','/')
                maskPath = maskPath.replace('\\','/')
                # print(imgPath,maskPath)
                valDict[imgPath] = maskPath
            else:
                print(search, 'Mask not found!')
            
with open('./valDict.json','w') as f:
    json.dump(valDict,f,indent=4)

testDict = {}
for fp,dl,fl in os.walk(os.path.join(imgRoot, 'test')):
    if(len(dl) == 0):
        for imgPath in fl:
            # print(imgPath)
            search = os.path.join(fp.split(imgRoot)[-1],imgPath.split('_leftImg8bit.png')[0])
            # print(search)
            maskPath = os.path.join(gtRoot,search + '_gtFine_labelTrainIds.png')
            # print(maskPath)
            if(os.path.isfile(maskPath)):
                imgPath = os.path.join(fp,imgPath).replace('\\','/')
                maskPath = maskPath.replace('\\','/')
                # print(imgPath,maskPath)
                testDict[imgPath] = maskPath
            else:
                print(search, 'Mask not found!')
with open('./testDict.json','w') as f:
    json.dump(testDict,f,indent=4)