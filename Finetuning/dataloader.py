import os
import torch
import random
import copy
import csv
import hashlib
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
from utils import JOINT_LABELS, JOINT_LABEL_TO_INDEX
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
        if getattr(self, "return_path", False):
            return img, image_label, img_path
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


class advCheX_hyp(Dataset):
    """advCheX 数据集的高血压（Hypertension vs nonHypertension）二分类版本"""

    def __init__(self, images_path, file_path, augment, num_class=2, few_shot=-1):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.num_class = num_class
        self.label_names = ["Hypertension", "nonHypertension"][:num_class]

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            if header and len(header) >= self.num_class + 1:
                header_labels = header[1:1 + self.num_class]
                if all(h.strip() for h in header_labels):
                    self.label_names = header_labels
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
        if getattr(self, "return_path", False):
            return img, image_label, img_path
        return img, image_label

    def __len__(self):
        return len(self.img_list)


class advCheX_hyp_multi_level(Dataset):
    """高血压分级：0~3 -> 三个“是否 ≥k”阈值标签"""

    def __init__(self, images_path, file_path, augment, few_shot=-1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for line in csv_reader:
                img_rel_path = line[0]
                img_abs_path = os.path.join(images_path, img_rel_path)
                self.img_list.append(img_abs_path)
                grade = int(line[1])
                # ordinal encode: >=1, >=2, >=3
                lab = [1 if grade >= k else 0 for k in [1, 2, 3]]
                self.img_label.append(lab)

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
        if getattr(self, "return_path", False):
            return img, image_label, img_path
        return img, image_label

    def __len__(self):
        return len(self.img_list)


class advCheX_hyp_multi_grade_stage_v1(Dataset):
    """高血压分级+分层联合：grade 0~3 / stage 0~2 的双头序数标签"""

    def __init__(self, images_path, file_path, augment, few_shot=-1, inconsistent_policy="drop"):
        self.img_list = []
        self.grade_labels = []
        self.stage_labels = []
        self.grade_list = []
        self.stage_list = []
        self.joint_ids = []
        self.augment = augment

        total_rows = 0
        drop_rows = 0
        bad_grade0 = 0
        bad_grade3 = 0
        bad_stage0 = 0

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)
            for line in csv_reader:
                if len(line) < 3:
                    continue
                total_rows += 1
                img_rel_path = line[0]
                stage = int(line[1])
                grade = int(line[2])

                inconsistent = False
                if grade == 0 and stage != 0:
                    bad_grade0 += 1
                    inconsistent = True
                if grade == 3 and stage != 2:
                    bad_grade3 += 1
                    inconsistent = True
                if stage == 0 and grade > 0:
                    bad_stage0 += 1
                    inconsistent = True

                if inconsistent:
                    if inconsistent_policy == "fix":
                        if grade == 0:
                            stage = 0
                        if grade == 3:
                            stage = 2
                        if stage == 0:
                            grade = 0
                    else:
                        drop_rows += 1
                        continue

                img_abs_path = os.path.join(images_path, img_rel_path)
                y_grade = [1 if grade >= k else 0 for k in [1, 2, 3]]
                y_stage = [1 if stage >= k else 0 for k in [1, 2]]
                joint_id = JOINT_LABEL_TO_INDEX[(grade, stage)]

                self.img_list.append(img_abs_path)
                self.grade_labels.append(y_grade)
                self.stage_labels.append(y_stage)
                self.grade_list.append(grade)
                self.stage_list.append(stage)
                self.joint_ids.append(joint_id)

        print(
            "[advCheX_hyp_multi_grade_stage_v1] total_rows={}, drop_rows={}, "
            "bad_grade0={}, bad_grade3={}, bad_stage0={}, policy={}".format(
                total_rows, drop_rows, bad_grade0, bad_grade3, bad_stage0, inconsistent_policy
            ),
            flush=True,
        )

        if few_shot > 0:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(len(indexes) * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            _grade_labels = copy.deepcopy(self.grade_labels)
            _stage_labels = copy.deepcopy(self.stage_labels)
            _grade_list = copy.deepcopy(self.grade_list)
            _stage_list = copy.deepcopy(self.stage_list)
            _joint_ids = copy.deepcopy(self.joint_ids)
            self.img_list = [_img_list[i] for i in indexes]
            self.grade_labels = [_grade_labels[i] for i in indexes]
            self.stage_labels = [_stage_labels[i] for i in indexes]
            self.grade_list = [_grade_list[i] for i in indexes]
            self.stage_list = [_stage_list[i] for i in indexes]
            self.joint_ids = [_joint_ids[i] for i in indexes]
            print(f"少样本模式：选取 {len(self.img_list)} 条数据（总{len(_img_list)}）")

    def __getitem__(self, index):
        img_path = self.img_list[index]
        y_grade = self.grade_labels[index]
        y_stage = self.stage_labels[index]
        grade = self.grade_list[index]
        stage = self.stage_list[index]
        joint_id = self.joint_ids[index]
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

        target = {
            "y_grade": torch.FloatTensor(y_grade),
            "y_stage": torch.FloatTensor(y_stage),
            "raw_grade": torch.tensor(int(grade), dtype=torch.long),
            "raw_stage": torch.tensor(int(stage), dtype=torch.long),
            "joint_id": torch.tensor(int(joint_id), dtype=torch.long),
            "meta": {
                "grade": int(grade),
                "stage": int(stage),
                "path": img_path,
            },
        }
        return img, target

    def __len__(self):
        return len(self.img_list)


class advCheX_hyp_grade_stage_embtab_base(Dataset):
    """Embedding + tabular baseline dataset for grade/stage ordinal multi-task."""

    TAB_COLS = ["age_abs", "age_topcoded", "sex_bin", "bmi_stat", "bmi_missing"]

    def __init__(self, images_path, file_path, split="train", tab_norm_stats=None, few_shot=-1, load_img_emb=True):
        self.images_path = images_path
        self.file_path = file_path
        self.split = str(split).lower()
        self.img_list = []
        self.grade_labels = []
        self.stage_labels = []
        self.grade_list = []
        self.stage_list = []
        self.tab_list = []
        self.joint_ids = []
        self.tab_norm_stats = dict(tab_norm_stats) if tab_norm_stats is not None else None
        self.load_img_emb = bool(load_img_emb)

        with open(file_path, "r") as f:
            csv_reader = csv.DictReader(f)
            missing_cols = [c for c in (["Path", "grade", "stage"] + self.TAB_COLS) if c not in csv_reader.fieldnames]
            if missing_cols:
                raise ValueError(f"CSV缺少必要列: {missing_cols}, got={csv_reader.fieldnames}")
            for row in csv_reader:
                rel_path = row["Path"].strip()
                emb_path = rel_path if os.path.isabs(rel_path) else os.path.join(images_path, rel_path)
                grade = int(row["grade"])
                stage = int(row["stage"])
                if grade < 0 or grade > 3 or stage < 0 or stage > 2:
                    continue
                if (grade, stage) not in JOINT_LABEL_TO_INDEX:
                    continue

                age_abs = float(row["age_abs"])
                age_topcoded = float(row["age_topcoded"])
                sex_bin = float(row["sex_bin"])
                bmi_stat = float(row["bmi_stat"])
                bmi_missing = float(row["bmi_missing"])
                self.img_list.append(emb_path)
                self.grade_labels.append([1 if grade >= k else 0 for k in [1, 2, 3]])
                self.stage_labels.append([1 if stage >= k else 0 for k in [1, 2]])
                self.grade_list.append(grade)
                self.stage_list.append(stage)
                self.joint_ids.append(JOINT_LABEL_TO_INDEX[(grade, stage)])
                self.tab_list.append([age_abs, age_topcoded, sex_bin, bmi_stat, bmi_missing])

        if self.split == "train" and self.tab_norm_stats is None:
            self.tab_norm_stats = self._compute_tab_norm_stats()
            self._save_tab_norm_stats()
        if self.tab_norm_stats is None:
            self.tab_norm_stats = self._load_tab_norm_stats()
        if self.tab_norm_stats is None:
            raise ValueError("tab_norm_stats 不可用；请先构建 train split 或传入统计量。")

        self.tab_list = [self._normalize_tab(x) for x in self.tab_list]

        if few_shot > 0:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(len(indexes) * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            self.img_list = [self.img_list[i] for i in indexes]
            self.grade_labels = [self.grade_labels[i] for i in indexes]
            self.stage_labels = [self.stage_labels[i] for i in indexes]
            self.grade_list = [self.grade_list[i] for i in indexes]
            self.stage_list = [self.stage_list[i] for i in indexes]
            self.joint_ids = [self.joint_ids[i] for i in indexes]
            self.tab_list = [self.tab_list[i] for i in indexes]

        print(
            f"[advCheX_hyp_grade_stage_embtab_base][{self.split}] N={len(self.img_list)} tab_norm={self.tab_norm_stats}",
            flush=True,
        )

    def _norm_cache_path(self):
        train_tag = hashlib.md5(os.path.abspath(self.file_path).encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.images_path, f"embtab_norm_stats_{train_tag}.json")

    def _compute_tab_norm_stats(self):
        tab = np.asarray(self.tab_list, dtype=np.float32)
        age = tab[:, 0]
        bmi = tab[:, 3]
        age_mean = float(np.mean(age))
        age_std = float(np.std(age))
        bmi_mean = float(np.mean(bmi))
        bmi_std = float(np.std(bmi))
        return {
            "age_mean": age_mean,
            "age_std": age_std if age_std > 1e-8 else 1.0,
            "bmi_mean": bmi_mean,
            "bmi_std": bmi_std if bmi_std > 1e-8 else 1.0,
        }

    def _save_tab_norm_stats(self):
        save_path = self._norm_cache_path()
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.tab_norm_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[advCheX_hyp_grade_stage_embtab_base] save norm stats failed: {e}", flush=True)

    def _load_tab_norm_stats(self):
        load_path = self._norm_cache_path()
        if not os.path.exists(load_path):
            return None
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_tab(self, x):
        age_abs, age_topcoded, sex_bin, bmi_stat, bmi_missing = x
        age_z = (float(age_abs) - float(self.tab_norm_stats["age_mean"])) / float(self.tab_norm_stats["age_std"])
        bmi_z = (float(bmi_stat) - float(self.tab_norm_stats["bmi_mean"])) / float(self.tab_norm_stats["bmi_std"])
        return [age_z, float(age_topcoded), float(sex_bin), bmi_z, float(bmi_missing)]

    def __getitem__(self, index):
        emb_path = self.img_list[index]
        if self.load_img_emb:
            img_emb = np.load(emb_path)
            img_emb = np.asarray(img_emb, dtype=np.float32).reshape(-1)
        else:
            img_emb = np.zeros((1376,), dtype=np.float32)
        tab = np.asarray(self.tab_list[index], dtype=np.float32)
        grade = int(self.grade_list[index])
        stage = int(self.stage_list[index])
        target = {
            "y_grade": torch.FloatTensor(self.grade_labels[index]),
            "y_stage": torch.FloatTensor(self.stage_labels[index]),
            "raw_grade": torch.tensor(grade, dtype=torch.long),
            "raw_stage": torch.tensor(stage, dtype=torch.long),
            "joint_id": torch.tensor(int(self.joint_ids[index]), dtype=torch.long),
            "meta": {"grade": grade, "stage": stage, "path": emb_path},
        }
        sample = {
            "img_emb": torch.from_numpy(img_emb),
            "tab": torch.from_numpy(tab),
        }
        if getattr(self, "return_path", False):
            return sample, target, emb_path
        return sample, target

    def __len__(self):
        return len(self.img_list)


class advCheX_hyp_grade_stage_embtab_v2lite(advCheX_hyp_grade_stage_embtab_base):
    """Embedding + tabular v2-lite dataset: reuse embtab-base loading and tab normalization."""

    pass


class advCheX_hyp_grade_stage_v2(advCheX_hyp_multi_grade_stage_v1):
    """v2 数据集薄包装：复用 v1 的 CSV 读取与监督结构。"""

    def __init__(self, images_path, file_path, augment, few_shot=-1, inconsistent_policy="drop"):
        super().__init__(images_path=images_path, file_path=file_path, augment=augment, few_shot=few_shot,
                         inconsistent_policy=inconsistent_policy)


class advCheX_hyp_multi_grade_stage_sep_v1(Dataset):
    """高血压分级+分层分离训练：共享encoder，仅 grade/stage 两个ordinal头。"""

    def __init__(self, images_path, file_path, augment, few_shot=-1):
        self.img_list = []
        self.grade_labels = []
        self.stage_labels = []
        self.grade_list = []
        self.stage_list = []
        self.augment = augment

        grade_counts = np.zeros(4, dtype=np.int64)
        stage_counts = np.zeros(3, dtype=np.int64)
        joint_counts = {f"{g}{s}": 0 for g, s in JOINT_LABELS}
        invalid_counts = {}

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)
            if header is None:
                raise ValueError(f"空CSV文件: {file_path}")
            hmap = {h.strip().lower(): i for i, h in enumerate(header)}
            if "path" not in hmap or "grade" not in hmap or "stage" not in hmap:
                raise ValueError(f"CSV必须包含Path/grade/stage列，当前header={header}")
            path_idx = hmap["path"]
            grade_idx = hmap["grade"]
            stage_idx = hmap["stage"]

            for line in csv_reader:
                if len(line) <= max(path_idx, grade_idx, stage_idx):
                    continue
                img_rel_path = line[path_idx]
                grade = int(line[grade_idx])
                stage = int(line[stage_idx])

                if grade < 0 or grade > 3 or stage < 0 or stage > 2:
                    continue

                grade_counts[grade] += 1
                stage_counts[stage] += 1
                if (grade, stage) in JOINT_LABEL_TO_INDEX:
                    joint_counts[f"{grade}{stage}"] += 1
                else:
                    key = f"g{grade}_s{stage}"
                    invalid_counts[key] = invalid_counts.get(key, 0) + 1

                img_abs_path = os.path.join(images_path, img_rel_path)
                y_grade = [1 if grade >= k else 0 for k in [1, 2, 3]]
                y_stage = [1 if stage >= k else 0 for k in [1, 2]]
                self.img_list.append(img_abs_path)
                self.grade_labels.append(y_grade)
                self.stage_labels.append(y_stage)
                self.grade_list.append(grade)
                self.stage_list.append(stage)

        print(f"[advCheX_hyp_multi_grade_stage_sep_v1] N={len(self.img_list)}", flush=True)
        print(f"[advCheX_hyp_multi_grade_stage_sep_v1] grade_dist={grade_counts.tolist()}", flush=True)
        print(f"[advCheX_hyp_multi_grade_stage_sep_v1] stage_dist={stage_counts.tolist()}", flush=True)
        print(f"[advCheX_hyp_multi_grade_stage_sep_v1] joint6_dist={joint_counts}", flush=True)
        print(
            f"[advCheX_hyp_multi_grade_stage_sep_v1] invalid_joint_count={int(sum(invalid_counts.values()))} details={invalid_counts}",
            flush=True,
        )

        if few_shot > 0:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(len(indexes) * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            _grade_labels = copy.deepcopy(self.grade_labels)
            _stage_labels = copy.deepcopy(self.stage_labels)
            _grade_list = copy.deepcopy(self.grade_list)
            _stage_list = copy.deepcopy(self.stage_list)
            self.img_list = [_img_list[i] for i in indexes]
            self.grade_labels = [_grade_labels[i] for i in indexes]
            self.stage_labels = [_stage_labels[i] for i in indexes]
            self.grade_list = [_grade_list[i] for i in indexes]
            self.stage_list = [_stage_list[i] for i in indexes]
            print(f"少样本模式：选取 {len(self.img_list)} 条数据（总{len(_img_list)}）")

    def __getitem__(self, index):
        img_path = self.img_list[index]
        y_grade = self.grade_labels[index]
        y_stage = self.stage_labels[index]
        grade = self.grade_list[index]
        stage = self.stage_list[index]
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

        target = {
            "y_grade": torch.FloatTensor(y_grade),
            "y_stage": torch.FloatTensor(y_stage),
            "raw_grade": torch.tensor(int(grade), dtype=torch.long),
            "raw_stage": torch.tensor(int(stage), dtype=torch.long),
            "meta": {
                "grade": int(grade),
                "stage": int(stage),
                "path": img_path,
            },
        }
        if getattr(self, "return_path", False):
            return img, target, img_path
        return img, target

    def __len__(self):
        return len(self.img_list)


class advCheX_hyp_multi_stage_v1(Dataset):
    """高血压分期：0~2 -> 两个“是否 ≥k”阈值标签"""

    def __init__(self, images_path, file_path, augment, few_shot=-1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for line in csv_reader:
                img_rel_path = line[0]
                img_abs_path = os.path.join(images_path, img_rel_path)
                self.img_list.append(img_abs_path)
                stage = int(line[1])
                # ordinal encode: >=1, >=2
                lab = [1 if stage >= k else 0 for k in [1, 2]]
                self.img_label.append(lab)

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
        if getattr(self, "return_path", False):
            return img, image_label, img_path
        return img, image_label

    def __len__(self):
        return len(self.img_list)


class advCheX_hyp_multi_stage_v2(Dataset):
    """高血压分期：0~2 -> 两个“是否 ≥k”阈值标签"""

    def __init__(self, images_path, file_path, augment, few_shot=-1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for line in csv_reader:
                img_rel_path = line[0]
                img_abs_path = os.path.join(images_path, img_rel_path)
                self.img_list.append(img_abs_path)
                stage = int(line[1])
                # ordinal encode: >=1, >=2
                lab = [1 if stage >= k else 0 for k in [1, 2]]
                self.img_label.append(lab)

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
        if getattr(self, "return_path", False):
            return img, image_label, img_path
        return img, image_label

    def __len__(self):
        return len(self.img_list)
