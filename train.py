import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchsummary import summary

from tqdm.auto import tqdm


from CONFIG import Config

config = Config()


# ==========================================================================
# Hyper-parameters
# ==========================================================================
DEVICE = config.DEVICE
IMG_SIZE = config.IMG_SIZE
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
LEARNING_RATE = config.LEARNING_RATE
WEIGHT_DECAY = config.WEIGHT_DECAY


# ==========================================================================
# Dataset
# ==========================================================================
class myDataset(Dataset):
    def __init__(self, data_folder = "./boxes_v2", 
                       dataset_file=".boxes_v2.txt", 
                       transform=None):
        """
          input:
            - dataset_file: a string of file name - "boxes.txt". In which, a line is (class_index, file_name)
        """

        self.transform = transform
        
        f = open(dataset_file, 'r')
        self.class_index_list = []
        self.image_path_list = []
        for line in f.readlines():
            class_index, file_name = [x.strip() for x in line.split(" ")]
            image_path = os.path.join(data_folder, file_name)
            self.class_index_list.append(int(class_index))
            self.image_path_list.append(image_path)
        f.close()

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        label = self.class_index_list[index]
        image_path = self.image_path_list[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        return image, label
    

# ==========================================================================
# Dataloader
# ==========================================================================
class myDataloader:
    def __init__(self, batch_size=128,
                       random_seed=42,
                       valid_size=0.2,
                       shuffle=True):
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle

    def load_data(self):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        train_tranform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        valid_tranform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]) 

        train_dataset = myDataset(transform=train_tranform) 
        valid_dataset = myDataset(transform=valid_tranform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))


        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.seed(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, sampler=train_sampler
        )

        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=self.batch_size, sampler=valid_sampler
        )

        return (train_loader, valid_loader)
    

# ==========================================================================
# Train function
# ==========================================================================
def train_phase(train_loader, model, criterion, optimizer):

    n_iterations = len(train_loader)
    size = 0
    train_loss = 0
    train_accuracy = 0

    for images, labels in tqdm(train_loader, total=n_iterations):
        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
      
        size += labels.size(0)

        # forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        train_loss += loss.item()
        
        y_hat = F.softmax(outputs, dim=1)
        value_softmax, index_softmax = torch.max(y_hat.data, dim=1)
        train_accuracy += (index_softmax == labels).sum().item()

        # backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del images, labels
        
    train_loss /= n_iterations
    train_accuracy /= size

    return train_loss, train_accuracy * 100


# ==========================================================================
# Test function
# ==========================================================================
def test_phase(test_loader, model, criterion):

    n_iterations = len(test_loader)
    size = 0
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        # turn off model.train() -> turn on model.eval() -> turn off model.eval() -> and then auto turn on model.train() again
        for images, labels in tqdm(test_loader, total=n_iterations):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            size += labels.size(0)

            # forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            y_hat = F.softmax(outputs, dim=1)
            value_softmax, index_softmax = torch.max(y_hat.data, dim=1)
            test_accuracy += (index_softmax == labels).sum().item()

            del images, labels

    test_loss /= n_iterations
    test_accuracy /= size

    return test_loss, test_accuracy * 100 



# ==========================================================================
# Define model
# ==========================================================================
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=512, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=512, out_features=128, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=128, out_features=config.NUM_CLASSES, bias=True),
)


model.to(DEVICE)

# ==========================================================================
# Loss function and Optimizer
# ==========================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# Training + Validation
dataloader = myDataloader(batch_size=BATCH_SIZE)
train_loader, test_loader = dataloader.load_data()

loss_opt = 1e9
acc_opt = -1e9

history_train_loss = []
history_test_loss = []
history_train_acc = []
history_test_acc = []

for epoch in range(NUM_EPOCHS):
    print()
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]:')
    
    
    train_loss, train_accuracy = train_phase(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
    print('Train Loss: {:.4f}, Train Accuracy: {:.4f}%' .format(train_loss, train_accuracy))

    test_loss, test_accuracy = test_phase(test_loader=test_loader, model=model, criterion=criterion)
    print('Valid Loss: {:.4f}, Valid Accuracy: {:.4f}%' .format(test_loss, test_accuracy))

    # scheduler.step()

    history_train_loss.append(train_loss)
    history_test_loss.append(test_loss)
    history_train_acc.append(train_accuracy)
    history_test_acc.append(test_accuracy)
    
    if test_loss < loss_opt and test_accuracy >= acc_opt:
        loss_opt = test_loss
        acc_opt = test_accuracy
        torch.save(model.state_dict(), './best.pt')
    torch.save(model.state_dict(), './last.pt')
