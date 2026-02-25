from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


image_size=384
crop_size=256
train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(crop_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(2).add(-1)), # [0, 1] -> [-1, 1]
            ]
        )

test_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


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
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # 获取类别名称，并将下划线替换为空格以便更符合自然语言
        class2label = get_class2label('fgsc23')  # 替换为你的数据集名称
        label2class = {v: k for k, v in class2label.items()}
        class_name = label2class[target]
        text = f"A image of a {class_name}."
        return img, text
    
def get_RS_datasets(train_path=None, test_path=None):
    if train_path:
        train_dataset = ImageTextFolder(root=train_path, transform=train_transform)
    else:
        train_dataset = None
    if test_path:
        test_dataset = ImageTextFolder(root=test_path, transform=test_transform)
    else:        
        test_dataset = None

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_path = "/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DIOR/train"
    test_path = "/picassox/intelligent-cpfs/segmentation/intern_segmentation/dc1/Infinity/data/Asker9527/Remote_Sense_Datasets/DIOR/test"
    train_dataset, test_dataset = get_RS_datasets(train_path, test_path)
    print("train dataset size:", len(train_dataset))
    print("test dataset size:", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for img, txt in train_loader:
        print("图像数据:", img.shape)  # 输出图像数据的形状
        # plt展示4张图像，保存为img.png
        # import matplotlib.pyplot as plt
        # img = img.numpy().transpose(0, 2, 3, 1)
        # fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        # for i in range(4):
        #     axes[i].imshow(img[i])
        #     axes[i].axis('off')
        # plt.savefig("img.png")
        print("文本数据:", txt)
        break
