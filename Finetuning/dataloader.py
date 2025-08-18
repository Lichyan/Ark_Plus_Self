import os
import torch
import random
import copy
import csv
from PIL import Image, ImageFile
import json
import SimpleITK as sitk

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence


class ChestXray14(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpert(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        if "test" in line[0]:
          label = line[1:]
        else:
          label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, few_shot = -1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

        indexes = np.arange(len(self.img_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])
            print(f"{few_shot} of total: {len(self.img_list)}")

    def __getitem__(self, index):

        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):
        return len(self.img_list)
    
class VinDrCXR_all(Dataset):
    def __init__(self, images_path, file_path, diseases, augment = None, few_shot = -1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            if "train" in file_path:
                all_diseases = next(csvReader, None)[2:]
                disease_idxs = [all_diseases.index(d) for d in diseases]
                # print(diseases)
                # print(disease_idxs)
                lines = [line for line in csvReader]
                assert len(lines)/3 == 15000
                for i in range(15000):
                    imagePath = os.path.join(images_path, "train_jpeg", lines[i*3][0]+".jpeg")
                    label = [0 for _ in range(len(diseases))]
                    r1,r2,r3 = lines[i*3][2:],lines[i*3+1][2:],lines[i*3+2][2:] 
                    for c in disease_idxs:
                        label[c] = 1  if int(r1[c])+int(r2[c])+int(r3[c]) > 0 else 0
                    self.img_list.append(imagePath)
                    self.img_label.append(label)
            else:
                all_diseases = next(csvReader, None)[1:]
                disease_idxs = [all_diseases.index(d) for d in diseases]
                # print(diseases)
                # print(disease_idxs)
                for line in csvReader:
                    imagePath = os.path.join(images_path, "test_jpeg", line[0]+".jpeg")
                    label = [int(l) for l in line[1:]]
                    # label = label[disease_idxs]
                    self.img_list.append(imagePath)
                    self.img_label.append(label)
        
        print("label shape: ", np.array(self.img_label).shape, np.sum(np.array(self.img_label), axis=0))

        indexes = np.arange(len(self.img_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])
            print(f"{few_shot} of total: {len(self.img_list)}")

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):
        return len(self.img_list)


# ---------------------------------------------Downstream RSNA Pneumonia------------------------------------------
class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=3, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          self.img_label.append(int(lineItems[-1]))

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = np.zeros(3)
    imageLabel[self.img_label[index]] = 1
    imageLabel = torch.FloatTensor(imageLabel)
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream COVIDx------------------------------------------
class COVIDx(Dataset):

  def __init__(self, images_path, file_path, augment, classes, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          patient_id, fname, label, source  = line.strip().split(' ')
          imagePath = os.path.join(images_path, fname)

          self.img_list.append(imagePath)
          self.img_label.append(classes.index(label))

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = np.zeros(3)
    imageLabel[self.img_label[index]] = 1
    imageLabel = torch.FloatTensor(imageLabel)
    if self.augment != None: imageData = self.augment(imageData)
 
    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream MIMIC------------------------------------------
class MIMIC(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

class ChestDR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=19, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0]+'.png')
          imageLabel = lineItems[1].split(',')
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")


  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)
  
# ---------------------------------------------Downstream advCheX------------------------------------------
class advCheX_old(Dataset):
    """
    适配advCheX数据集的数据加载类，用于Ark_Plus微调
    支持多标签分类（19类疾病），兼容数据增强和少样本学习
    """
    def __init__(self, images_path, file_path, augment, num_class=19,
                 uncertain_label="Ones", unknown_label=0, few_shot=-1, target_size=768):
        # 初始化变量
        self.img_list = []  # 存储图像绝对路径
        self.img_label = []  # 存储图像标签（19维列表）
        self.augment = augment  # 数据增强方法（训练时使用）
        self.num_class = num_class  # 类别数：19类
        self.target_size = int(target_size)
        
        # 校验不确定标签处理策略（你的数据可能用不到，但保留兼容性）
        assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
        self.uncertain_label = uncertain_label
        self.unknown_label = unknown_label  # 未知标签填充值（默认0）

        # 读取CSV文件并解析图像路径和标签
        self._parse_csv(images_path, file_path)

        # 处理少样本学习（如需仅使用部分数据）
        if few_shot > 0:
            self._subsample_data(few_shot)

    def _parse_csv(self, images_path, file_path):
        """解析CSV文件，提取图像路径和19类标签"""
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)  # 跳过表头：Path, Normal, ASD, ..., Other
            
            # 遍历CSV中的每一行数据
            for line in csv_reader:
                # 第0列是图像相对路径（如"advCheX/train/patient1202521303/study1/view1_frontal.jpg"）
                img_rel_path = line[0]
                # 拼接绝对路径：images_path + 相对路径（确保路径正确）
                img_abs_path = os.path.join(images_path, img_rel_path)
                self.img_list.append(img_abs_path)

                # 第1-19列是标签（Normal到Other共19类）
                labels = line[1:1+self.num_class]  # 取19个标签值
                # 转换标签为整数（0或1，多标签分类）
                parsed_labels = []
                for label in labels:
                    # 处理空值（如果有），填充为unknown_label（默认0）
                    if not label.strip():
                        parsed_labels.append(self.unknown_label)
                    else:
                        parsed_labels.append(int(label))  # 正常标签转换为0/1
                self.img_label.append(parsed_labels)

    def _subsample_data(self, few_shot):
        """少样本学习：随机选取部分数据（如few_shot=0.1表示10%数据）"""
        # 生成随机索引（固定种子保证可复现）
        indexes = np.arange(len(self.img_list))
        random.Random(99).shuffle(indexes)  # 固定随机种子
        
        # 计算需要选取的样本数
        if few_shot < 1:
            num_data = int(len(self.img_list) * few_shot)  # 比例
        else:
            num_data = int(few_shot)  # 绝对数量
        num_data = max(1, num_data)  # 至少保留1个样本
        selected_indexes = indexes[:num_data]

        # 保留选中的样本
        _img_list = copy.deepcopy(self.img_list)
        _img_label = copy.deepcopy(self.img_label)
        self.img_list = [_img_list[i] for i in selected_indexes]
        self.img_label = [_img_label[i] for i in selected_indexes]
        
        print(f"少样本模式：选取 {len(self.img_list)} 条数据（总{len(_img_list)}）")

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.img_label[index]

        # 1) 读图（出错则返回 None 让 collate_fn 过滤）
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # 统一3通道
            _ = img.size  # 强制触发lazy-load
        except Exception as e:
            print(f"[IO ERROR] idx={index} path={img_path} err={repr(e)}", flush=True)
            return None, None

        # 2) 变换/增强（出错也跳过该样本）
        try:
            if self.augment is not None:
                img = self.augment(img)
        except Exception as e:
            print(f"[AUG ERROR] idx={index} path={img_path} err={repr(e)}", flush=True)
            return None, None

        image_label = torch.FloatTensor(label)
        return img, image_label

    
    # def __getitem__(self, index):
    #     """获取单个样本：图像+标签（适配模型输入）"""
    #     # 1. 读取图像并转换为RGB（模型输入为3通道）
    #     img_path = self.img_list[index]
    #     image = Image.open(img_path).convert('RGB')  # 转换为RGB格式

    #     # 2. 处理标签（你的数据无-1，简化处理）
    #     label = self.img_label[index]
    #     # 如需支持不确定标签（-1），可在此处添加处理逻辑（参考CheXpert）
    #     # 此处直接转换为FloatTensor（多标签分类用float类型）
    #     image_label = torch.FloatTensor(label)

    #     # 3. 应用数据增强（训练时）
    #     if self.augment is not None:
    #         image = self.augment(image)

    #     return image, image_label  # 返回（图像张量，标签张量）
    

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.img_list)


#---------------------------------------------Downstream advCheX_binary_2types------------------------------------------
class advCheX_binary(Dataset):
    """二分类版本的 advCheX 数据集，标签为 [CHD, nonCHD]"""

    def __init__(self, images_path, file_path, augment, num_class=2, few_shot=-1):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.num_class = num_class

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for line in csv_reader:
                img_rel_path = line[0]
                img_abs_path = os.path.join(images_path, img_rel_path)
                self.img_list.append(img_abs_path)
                labels = [int(i) for i in line[1:1 + self.num_class]]
                self.img_label.append(labels)

        if few_shot > 0:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(len(indexes) * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            _img_label = copy.deepcopy(self.img_label)
            self.img_list = [_img_list[i] for i in indexes]
            self.img_label = [_img_label[i] for i in indexes]
            print(f"少样本模式：选取 {len(self.img_list)} 条数据（总{len(_img_list)}）")

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.img_label[index]
        try:
            img = Image.open(img_path).convert('RGB')
            _ = img.size
        except Exception as e:
            print(f"[IO ERROR] idx={index} path={img_path} err={repr(e)}", flush=True)
            return None, None
        if self.augment is not None:
            try:
                img = self.augment(img)
            except Exception as e:
                print(f"[AUG ERROR] idx={index} path={img_path} err={repr(e)}", flush=True)
                return None, None

        image_label = torch.FloatTensor(label)
        return img, image_label

    def __len__(self):
        return len(self.img_list)
    
#---------------------------------------------Downstream advCheX_new_3types------------------------------------------
class advCheX(Dataset):
    """
    适配advCheX数据集的数据加载类，用于Ark_Plus微调
    支持多标签分类（3类疾病），兼容数据增强和少样本学习
    """
    def __init__(self, images_path, file_path, augment, num_class=3,
                 uncertain_label="Ones", unknown_label=0, few_shot=-1, target_size=768):
        # 初始化变量
        self.img_list = []  # 存储图像绝对路径
        self.img_label = []  # 存储图像标签（3维列表）
        self.augment = augment  # 数据增强方法（训练时使用）
        self.num_class = num_class  # 类别数：3类
        self.target_size = int(target_size)
        
        # 校验不确定标签处理策略（你的数据可能用不到，但保留兼容性）
        assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
        self.uncertain_label = uncertain_label
        self.unknown_label = unknown_label  # 未知标签填充值（默认0）

        # 读取CSV文件并解析图像路径和标签
        self._parse_csv(images_path, file_path)

        # 处理少样本学习（如需仅使用部分数据）
        if few_shot > 0:
            self._subsample_data(few_shot)

    def _parse_csv(self, images_path, file_path):
        """解析CSV文件，提取图像路径和3类标签"""
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)  # 跳过表头：Path, CHD, nonCHD, Other
            
            # 遍历CSV中的每一行数据
            for line in csv_reader:
                # 第0列是图像相对路径（如"advCheX/train/patient1202521303/study1/view1_frontal.jpg"）
                img_rel_path = line[0]
                # 拼接绝对路径：images_path + 相对路径（确保路径正确）
                img_abs_path = os.path.join(images_path, img_rel_path)
                self.img_list.append(img_abs_path)

                # 第1-3列是标签（Normal到Other共3类）
                labels = line[1:1+self.num_class]  # 取3个标签值
                # 转换标签为整数（0或1，多标签分类）
                parsed_labels = []
                for label in labels:
                    # 处理空值（如果有），填充为unknown_label（默认0）
                    if not label.strip():
                        parsed_labels.append(self.unknown_label)
                    else:
                        parsed_labels.append(int(label))  # 正常标签转换为0/1
                self.img_label.append(parsed_labels)

    def _subsample_data(self, few_shot):
        """少样本学习：随机选取部分数据（如few_shot=0.1表示10%数据）"""
        # 生成随机索引（固定种子保证可复现）
        indexes = np.arange(len(self.img_list))
        random.Random(99).shuffle(indexes)  # 固定随机种子
        
        # 计算需要选取的样本数
        if few_shot < 1:
            num_data = int(len(self.img_list) * few_shot)  # 比例
        else:
            num_data = int(few_shot)  # 绝对数量
        num_data = max(1, num_data)  # 至少保留1个样本
        selected_indexes = indexes[:num_data]

        # 保留选中的样本
        _img_list = copy.deepcopy(self.img_list)
        _img_label = copy.deepcopy(self.img_label)
        self.img_list = [_img_list[i] for i in selected_indexes]
        self.img_label = [_img_label[i] for i in selected_indexes]
        
        print(f"少样本模式：选取 {len(self.img_list)} 条数据（总{len(_img_list)}）")

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.img_label[index]

        # 1) 读图（出错则返回 None 让 collate_fn 过滤）
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # 统一3通道
            _ = img.size  # 强制触发lazy-load
        except Exception as e:
            print(f"[IO ERROR] idx={index} path={img_path} err={repr(e)}", flush=True)
            return None, None

        # 2) 变换/增强（出错也跳过该样本）
        try:
            if self.augment is not None:
                img = self.augment(img)
        except Exception as e:
            print(f"[AUG ERROR] idx={index} path={img_path} err={repr(e)}", flush=True)
            return None, None

        image_label = torch.FloatTensor(label)
        return img, image_label

    
    # def __getitem__(self, index):
    #     """获取单个样本：图像+标签（适配模型输入）"""
    #     # 1. 读取图像并转换为RGB（模型输入为3通道）
    #     img_path = self.img_list[index]
    #     image = Image.open(img_path).convert('RGB')  # 转换为RGB格式

    #     # 2. 处理标签（你的数据无-1，简化处理）
    #     label = self.img_label[index]
    #     # 如需支持不确定标签（-1），可在此处添加处理逻辑（参考CheXpert）
    #     # 此处直接转换为FloatTensor（多标签分类用float类型）
    #     image_label = torch.FloatTensor(label)

    #     # 3. 应用数据增强（训练时）
    #     if self.augment is not None:
    #         image = self.augment(image)

    #     return image, image_label  # 返回（图像张量，标签张量）
    

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.img_list)

