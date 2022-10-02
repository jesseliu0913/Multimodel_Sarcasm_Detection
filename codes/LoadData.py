import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, BertConfig, BertTokenizer
import torchvision
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import time
import os
import Config
import pickle
import PIL


device = Config.device
WORKING_PATH = Config.WORKING_PATH
tokenizer = Config.tokenizer
train_fraction = Config.train_fraction
val_fraction = Config.val_fraction
batch_size = Config.batch_size

"""
read the textfile and then match the image and all the text via dict
"""

def load_data():
    data_set = dict()
    # 1. read the textfile
    for dataset in ["train"]:
        text_file = open(os.path.join(WORKING_PATH, "text_data/", dataset+".txt"), "rb")
        label_file = open(os.path.join(WORKING_PATH, "multilabel_database/", "img_to_five_words.txt"), "rb")
        for text, label in zip(text_file, label_file):
            text_content = eval(text)
            image = text_content[0]
            sentence = text_content[1]
            group = text_content[2]
            label_content = eval(label)

        if os.path.isfile(os.path.join(WORKING_PATH, "dataset_image/", image + ".jpg")):
            data_set[int(image)] = {"text": sentence, "group": group, "label": label_content[1:]}

    for dataset in ["test", "valid"]:
        text_file = open(os.path.join(WORKING_PATH, "text_data/", dataset + ".txt"), "rb")
        label_file = open(os.path.join(WORKING_PATH, "multilabel_database/", "img_to_five_words.txt"), "rb")
        for text, label in zip(text_file, label_file):
            text_content = eval(text)
            image = text_content[0]
            sentence = text_content[1]
            group = text_content[2]
            label_content = eval(label)

            if os.path.isfile(os.path.join(WORKING_PATH, "dataset_image/", image + ".jpg")):
                data_set[int(image)] = {"text": sentence, "group": group, "label": label_content[1:]}

    return data_set


data_set = load_data()
# print(data_set)



# cluster all dattset and save them
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.image_ids = list(data.keys())
        for id in data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"dataset_image/",str(id)+".jpg")
            # now in the dict there have four value for one image id, including text, group, label and image_path



    # load image feature data - resnet 50 result
    def __image_feature_loader(self, id):
        attribute_feature = np.load(os.path.join(WORKING_PATH,"image_feature_data",str(id)+".npy"))
        return torch.from_numpy(attribute_feature)  # switch array to tensor



    # adjust the image
    def image_loader(self, id):
        path=self.data[id]["image_path"]
        img_pil =  PIL.Image.open(path)
        transform = transforms.Compose([transforms.Resize((448,448)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
        img_tensor = transform(img_pil)
        return img_tensor

    def text_loader(self, id):
        return self.data[id]["text"]


    def label_loader(self, id):
        return self.data[id]["label"]


    def __getitem__(self, index):
        id=self.image_ids[index]
        result = tokenizer(self.data[id]["text"], padding='max_length', truncation=True, max_length=196,
                           return_tensors='pt')
        # img = self.__image_loader(id)
        input_ids = result['input_ids'].squeeze(0)
        atten_mask = result['attention_mask'].squeeze(0)
        token_type_ids = result['token_type_ids'].squeeze(0)
        image_feature = self.__image_feature_loader(id)
        label = self.data[id]["label"]
        group = self.data[id]["group"]
        return input_ids, atten_mask, image_feature, token_type_ids, label, group, id

    def __len__(self):
        return len(self.image_ids)



# split the dataset
def train_val_test_split(all_Data, train_fraction, val_fraction):
    train_val_test_count=[int(len(all_Data)*train_fraction),int(len(all_Data)*val_fraction),0]
    train_val_test_count[2]=len(all_Data)-sum(train_val_test_count)
    torch.manual_seed(42)
    return random_split(all_Data, train_val_test_count, generator=torch.Generator().manual_seed(42))


all_Data = MyDataset(data_set)
train_set, val_set, test_set = train_val_test_split(all_Data, train_fraction, val_fraction)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
# play_loader = DataLoader(test_set, batch_size=1, shuffle=True)
# test_data_size = len(test_set)
# print(test_data_size)

"""
example of the data
"""
# if __name__ == "__main__":
#
#     for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in train_loader:
#         print("input_ids", input_ids)
#         print("atten_mask", atten_mask.shape)
#         print("token_type_ids", token_type_ids.shape)
#         print("image feature", image_feature.shape, image_feature.type())
#         print("label", label)
#         print("group", group.shape, group.type())
#         print("image id", id, id.type())
#         break