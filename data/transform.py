# @Time    : 2023/5/11 13:07
# @Author  : emo

from torchvision import transforms


def get_train_transform(mean, std, size):
    return transforms.Compose([
        transforms.Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transform(mean, std, size):
    return transforms.Compose([
        transforms.Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transforms(input_size=224):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transformations = {'train': get_train_transform(mean, std, input_size),
                       'val': get_val_transform(mean, std, input_size)}
    return transformations
