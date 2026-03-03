from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import PIL.Image as PImage
import io
import random
import torchvision.transforms.functional as TF

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

class ResizeAndPad:
    def __init__(self, target_size, fill=0, padding_mode='constant'):
        """
        :param target_size: 最终输出的目标大小 (int)
        :param fill: 填充的像素值 (如果是 'constant' 模式)
        :param padding_mode: 填充模式，可选 'constant', 'edge', 'reflect', 'symmetric'
        """
        self.target_size = target_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # 假设输入是 PIL Image
        w, h = img.size
        
        # 1. 计算缩放比例，使最长边等于 target_size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 2. 等比例缩放图像
        img = F.resize(img, (new_h, new_w))
        
        # 3. 计算四周需要填充的像素量，使其居中
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        # 4. 执行填充
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        img = F.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)
        
        return img


def joint_transform(src_img, is_train=True):
    """联合变换：确保 src 和 tgt 经历完全相同的形变、裁剪和翻转"""
    # 1. 保持原版的等比缩放逻辑
    width, height = src_img.size
    tgt_w, tgt_h = crop_size, crop_size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    
    src_img = src_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)

    crop_y = (resized_height - tgt_h) // 2
    crop_x = (resized_width - tgt_w) // 2

    src_img = TF.crop(src_img, crop_y, crop_x, tgt_h, tgt_w)
    # 3. 联合随机翻转 (Joint Random Flip)
    if is_train and random.random() > 0.5:
        src_img = TF.hflip(src_img)

    # 4. 转为 Tensor 并归一化到 [-1, 1]
    src_tensor = TF.to_tensor(src_img).mul_(2).add_(-1)

    return src_tensor


def get_class2label(dataset_name:str):
    if dataset_name == 'dota':
        class2label={
            "Baseball-diamond": 0,
            "Basketball-court": 1,
            "Bridge": 2,
            "Ground-track-field": 3,
            "Harbor": 4,
            "Helicopter": 5,
            "Large-Vehicle": 6,
            "Plane": 7,
            "Roundabout": 8,
            "Ship": 9,
            "Small-Vehicle": 10,
            "Soccer-ball-field": 11,
            "Swimming-Pool": 13,
            "Storage-Tank": 12,
            "Tennis-court": 14,
            }
        return class2label

    elif dataset_name == 'dior':
        class2label={
            "Ship": 13,
            "Vehicle": 18,
            "Tennis court": 16,
            "Storage tank": 15,
            "Baseball field": 2,
            "Harbor": 11,
            "Windmill": 19,
            "Airplane": 0,
            "Bridge": 4,
            "Overpass": 12,
            "Ground track field": 10,
            "Expressway-Service area": 7,
            "Basketballcourt": 3,
            "Airport": 1,
            "Chimney": 5,
            "Expressway-toll-station": 8,
            "Stadium": 14,
            "Dam": 6,
            "Golf field": 9,
            "Train station": 17
            }
        return class2label

    elif dataset_name == 'fgsc23':
        label2class={
            0:"non-ship",
            1:"air carrier",
            2:"destroyer",
            3:"landing craft",
            4:"frigate",
            5:"amphibious transport dock",
            6:"cruiser",
            7:"Tarawa-class amphibious assault ship",
            8:"amphibious assault ship",
            9:"command ship",
            10:"submarine",
            11:"medical ship",
            12:"combat boat",
            13:"auxiliary ship",
            14:"container ship",
            15:"car carrier",
            16:"hovercraft",
            17:"bulk carrier",
            18:"oil tanker",
            19:"fishing boat",
            20:"passenger ship",
            21:"liquefied gas ship",
            22:"barge"}
        class2label = {value: key for key, value in label2class.items()}
        return class2label
    else:
        raise ValueError(f"input dataset_name error: {dataset_name}")

class ImageTextFolder(datasets.ImageFolder):
    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def get_samples_per_class(self):
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # print(f"原始标签索引: {target}")
        target = self.classes[int(target)]
        # print(self.classes)

        # 获取类别名称，并将下划线替换为空格以便更符合自然语言
        class2label = get_class2label(self.dataset_name)  # 替换为你的数据集名称
        # print(class2label)
        label2class = {v: k for k, v in class2label.items()}
        class_name = label2class[int(target)]
        text = f"A high-resolution satellite top-down view of a {class_name} in a remote sensing image."
        return img, text
    
def get_RS_datasets(args, train_path=None, test_path=None):

    if args.pn == '0.06M':
        image_size = 256
    elif args.pn == '0.25M':
        image_size = 512
    elif args.pn == '1M':
        image_size = 1024
    else:
        raise ValueError(f"Unsupported parameter pn: {args.pn}. Expected one of ['0.06M', '0.25M', '1M'].")

    train_transform = transforms.Compose([
        # 替换原本的强行拉伸 Resize，使用等比例缩放+填充
        # 提示：在遥感中，使用 'reflect' (镜像填充) 往往比填黑边 ('constant') 效果更好，可以避免黑边在卷积时产生的明显边缘突变。
        ResizeAndPad(target_size=image_size, padding_mode='reflect'), 
        
        # # --- 强烈建议加入的遥感专属数据增强 ---
        transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),   # 随机垂直翻转
        transforms.RandomChoice([               # 随机旋转 90/180/270 度
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        # -----------------------------------
        
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(2).add(-1)), # 归一化到 [-1, 1]
    ])

    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            # transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(2).add(-1)),
        ]
    )
    if train_path:
        if 'DIOR' in train_path:
            dataset_name = 'dior'
        elif 'DOTA' in train_path:
            dataset_name = 'dota'
        elif 'FGSC' in train_path:
            dataset_name = 'fgsc23'

        train_dataset = ImageTextFolder(root=train_path, transform=train_transform)
        train_dataset.set_dataset_name(dataset_name)
    else:
        train_dataset = None

    if test_path:
        if 'DIOR' in test_path:
            dataset_name = 'dior'
        elif 'DOTA' in test_path:
            dataset_name = 'dota'
        elif 'FGSC' in test_path:
            dataset_name = 'fgsc23'
        test_dataset = ImageTextFolder(root=test_path, transform=test_transform)
        test_dataset.set_dataset_name(dataset_name)
    else:        
        test_dataset = None


    return train_dataset, test_dataset


if __name__ == "__main__":
    train_path = "/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DOTA/train"
    test_path = "/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DOTA/test"
    args = type('Args', (object,), {'pn': '0.06M'})()  # 创建一个简单的对象来模拟 args
    train_dataset, test_dataset = get_RS_datasets(args, train_path, test_path)
    print("train dataset size:", len(train_dataset))
    print("test dataset size:", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for img, txt in train_loader:
        print("图像数据:", img.shape)  # 输出图像数据的形状
        # plt展示4张图像，保存为img.png
        img = (img + 1) / 2
        # img = img.permute(0, 2, 3, 1).numpy()

        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8) # [B, H, W, 3]


        # img = img.numpy().transpose(0, 2, 3, 1)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i in range(4):
            axes[i].imshow(img[i])
            axes[i].axis('off')
        plt.savefig("img.png")
        print("文本数据:", txt)
        break
