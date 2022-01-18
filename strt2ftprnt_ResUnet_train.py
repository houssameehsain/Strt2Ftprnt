import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import glob
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

img_size = 256
X_img_path = "D:/AI in Urban Design/DL UD dir/Filtered_X"
Y_img_path = "D:/AI in Urban Design/DL UD dir/Filtered_Y"

def make_data(X_img_path, Y_img_path, img_size):
    X_data = []
    Y_data = []

    X_img_count = 0
    Y_img_count = 0

    for i in tqdm(os.listdir(X_img_path), ncols=100, disable=False):
        path = os.path.join(X_img_path, i)
        X = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        X = cv2.resize(X, (img_size, img_size))
        X = np.array(X)
        X = X.reshape((1, img_size, img_size))
        X = X / 255
        X = 1 - X
        X_data.append(X)
        X_img_count += 1

    for i in tqdm(os.listdir(Y_img_path), ncols=100, disable=False):
        path = os.path.join(Y_img_path, i)
        Y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        Y = cv2.resize(Y, (img_size, img_size))
        Y = np.array(Y)
        Y = Y.reshape((1, img_size, img_size))
        Y = Y / 255
        Y = 1 - Y
        Y_data.append(Y)
        Y_img_count += 1

    print('X Image_count:' + str(X_img_count))
    print('Y Image_count:' + str(Y_img_count))

    return X_data, Y_data

class segmentationDataSet(data.Dataset):
    def __init__(self, X_img_path, Y_img_path, img_size):
        self.inputs_path = X_img_path
        self.targets_path = Y_img_path
        self.img_size = img_size
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.inputs, self.targets = make_data(self.inputs_path, self.targets_path, self.img_size)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]
        # Load input and target
        x, y = input_ID, target_ID
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        return x, y

batch_S = 2
test_split = 0.1
ts = 0.1 / 0.9

dataset = segmentationDataSet(X_img_path, Y_img_path, img_size)

dataset_size = len(dataset)
test_size = int(test_split * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

trdataset_size = len(train_dataset)
val_size = int(ts * trdataset_size)
training_size = trdataset_size - val_size

training_dataset, val_dataset = data.random_split(train_dataset, [training_size, val_size])

training_dataloader = data.DataLoader(dataset=training_dataset, batch_size = batch_S, shuffle=True)
x, y = next(iter(training_dataloader))
print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; type: {y.dtype}')
print(f'y = min: {y.min()}; max: {y.max()}')

val_dataloader = data.DataLoader(dataset=val_dataset, batch_size = batch_S, shuffle=True)
x, y = next(iter(val_dataloader))
print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; type: {y.dtype}')
print(f'y = min: {y.min()}; max: {y.max()}')

# Unet architecture
def initial_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
    )
    return conv
def enc_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c)
    )
    return conv
def mid_block(out_c):
    conv = nn.Sequential(
        nn.Conv2d(out_c, out_c, kernel_size=7, padding=3, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, kernel_size=7, padding=3, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c)
    )
    return conv
def dec_block(out_c):
    conv = nn.Sequential(
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, kernel_size=7, padding=3, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c)
    )
    return conv
def end_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=7, padding=3, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
        nn.Conv2d(out_c, out_c, kernel_size=7, padding=3, stride=1, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
    )
    return conv
def sigmoid_block(out_c):
    conv = nn.Sequential(
        nn.Conv2d(out_c, 1, kernel_size=1, padding=0, stride=1, bias=True),
        nn.Sigmoid()
    )
    return conv
def conv_res(in_c, out_c):
    resconv = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1, bias=True)
    return resconv
class Sigmoid_Unet(nn.Module):
    def __init__(self):
        super(Sigmoid_Unet, self).__init__()
        self.initial = initial_block(1, 32)
        self.encblock1 = enc_block(32, 64)
        self.encblock2 = enc_block(64, 128)
        self.encblock3 = enc_block(128, 256)
        self.mid = mid_block(256)
        self.decblock1 = dec_block(128)
        self.decblock2 = dec_block(64)
        self.decblock3 = dec_block(32)
        self.end = end_block(16, 16)
        self.sigmoid = sigmoid_block(16)

        self.transpose1 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, bias=False)
        self.transpose2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, bias=False)
        self.transpose3 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, bias=False)
        self.transpose4 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2, bias=False)

        self.res1 = conv_res(1, 32)
        self.res2 = conv_res(32, 64)
        self.res3 = conv_res(64, 128)
        self.res4 = conv_res(128, 256)
        self.res5 = conv_res(256, 256)
        self.res6 = conv_res(128, 128)
        self.res7 = conv_res(64, 64)
        self.res8 = conv_res(32, 32)
        self.res9 = conv_res(16, 16)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, image):
        x1 = self.initial(image)
        x1res = self.relu(x1 + self.res1(image))
        x2 = self.maxpool(x1res)

        x3 = self.encblock1(x2)
        x2res = self.relu(x3 + self.res2(x2))
        x4 = self.maxpool(x2res)

        x5 = self.encblock2(x4)
        x3res = self.relu(x5 + self.res3(x4))
        x6 = self.maxpool(x3res)

        x7 = self.encblock3(x6)
        x4res = self.relu(x7 + self.res4(x6))
        x8 = self.maxpool(x4res)

        x9 = self.mid(x8)
        x5res = self.relu(x9 + self.res5(x8))
        x10 = self.transpose1(torch.cat([x5res, x8], 1))

        x11 = self.decblock1(x10)
        x6res = self.relu(x11 + self.res6(x10))
        x12 = self.transpose2(torch.cat([x6res, x6], 1))

        x13 = self.decblock2(x12)
        x7res = self.relu(x13 + self.res7(x12))
        x14 = self.transpose3(torch.cat([x7res, x4], 1))

        x15 = self.decblock3(x14)
        x8res = self.relu(x15 + self.res8(x14))
        x16 = self.transpose4(torch.cat([x8res, x2], 1))

        x17 = self.end(x16)
        x9res = self.relu(x17 + self.res9(x16))
        out = self.sigmoid(x9res)
        return out

# hyperparameters
num_epochs = 100
learning_rate = 0.002
lr_decay = 0.1

# Set device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
device = get_device()
print(device)

def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return intersection / union
def diceCoeff(pred, gt, smooth=1e-5):
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
    return loss.sum() / N
def diceCoeffv2(pred, gt, eps=1e-5):
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt)

# model
model = Sigmoid_Unet()
model.to(device)
# criterion
criterion = DiceLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                 threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                 min_lr=0, eps=1e-08, verbose=False)

# Training
class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.DataLoader,
                 validation_DataLoader: torch.utils.data.DataLoader,
                 lr_scheduler: torch.optim.lr_scheduler,
                 epochs: int,
                 epoch: int = 0,
                 batch: int = 0
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.batch = batch

        self.training_loss = []
        self.validation_loss = []
        self.training_acc = []
        self.validation_acc = []
        self.learning_rate = []

    def run_trainer(self):

        progressbar = trange(self.epochs, desc='Progress')
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.training_acc, self.validation_acc, self.learning_rate

    def _train(self):

        self.model.train()  # train mode
        train_losses = []
        train_accuracy = []
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            if i == 0:
                self.batch += 1  # batch counter

            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass

            # Saving training output images
            img1 = torch.unsqueeze(out[0, 0, :, :], 2)
            img1 = img1 * 255
            img1 = img1.cpu()
            np_img1 = img1.detach().numpy()
            cv2.imwrite('D:/AI in Urban Design/DL UD dir/strt2ftprnt_trainOUT/'
                        + 'train' + str(self.epoch) + '-' + str(self.batch) + '-' + str(i+1) + '.jpeg',
                        np_img1)

            acc = dice_metric(out, target)  # calculate accuracy
            acc_value = acc.item()
            train_accuracy.append(acc_value)

            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f}) (acc {acc_value:.4f})')  # update progressbar

        self.training_acc.append(np.mean(train_accuracy))
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):

        self.model.eval()  # evaluation mode
        val_losses = []
        val_accuracy = []
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)

                # Saving validation output images
                img1 = torch.unsqueeze(out[0, 0, :, :], 2)
                img1 = img1 * 255
                img1 = img1.cpu()
                np_img1 = img1.detach().numpy()
                cv2.imwrite('D:/AI in Urban Design/DL UD dir/strt2ftprnt_valOUT/'
                            + 'train' + str(self.epoch) + '-' + str(self.batch) + '-' + str(i + 1) + '.jpeg',
                            np_img1)

                acc = dice_metric(out, target)  # calculate accuracy
                acc_value = acc.item()
                val_accuracy.append(acc_value)

                loss = self.criterion(out, target)
                loss_value = loss.item()
                val_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f}) (acc {acc_value:.4f})')

        self.validation_acc.append(np.mean(val_accuracy))
        self.validation_loss.append(np.mean(val_losses))

        batch_iter.close()

trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader = training_dataloader,
                  validation_DataLoader = val_dataloader,
                  lr_scheduler=scheduler,
                  epochs=num_epochs,
                  epoch=0)
# start training
training_losses, validation_losses, training_acc, validation_acc, lr_rates = trainer.run_trainer()

# save trained model
PATH = 'D:/AI in Urban Design/DL UD dir/DL model/strt2ftprnt_ResUnet.pt'
torch.save(model.state_dict(), PATH)

# Plot loss vs epochs
epoch_list = []
for i in range(len(training_losses)):
    epoch_list.append(i + 1)

# Dice loss and Accuracy plot
plt.plot(epoch_list, training_losses, color='r', label="Training Loss")
plt.plot(epoch_list, validation_losses, color='b', label="Validation Loss")
plt.title("Dice Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy plot
plt.plot(epoch_list, training_acc, color='r', linestyle='--', label="Training Accuracy")
plt.plot(epoch_list, validation_acc, color='b', linestyle='--', label="Validation Accuracy")
plt.title("Dice Metric")
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

# lr plot
plt.plot(epoch_list, lr_rates, color='g', label="Learning rate")
plt.title("Learning rate during training")
plt.xlabel('Epochs')
plt.ylabel('Lr')
plt.legend()
plt.show()
