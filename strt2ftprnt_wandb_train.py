import os
import numpy as np
import cv2

import torch 
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import wandb

# Log in to W&B account
wandb.login(key='') # place wandb login key here

img_size = 256
test_split = 0.1
ts = 0.1 / 0.9
pad = 10

# dataset paths
X_img_path = "D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/Filtered_X" # replace with your own path
Y_img_path = "D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/Filtered_Y" # replace with your own path

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

dataset = segmentationDataSet(X_img_path, Y_img_path, img_size)

dataset_size = len(dataset)
test_size = int(test_split * dataset_size)
train_size = dataset_size - test_size

training_dataset, validation_dataset = data.random_split(dataset, [train_size, test_size])

# PatchGAN Discriminator
class PatchGAN(nn.Module):
    def __init__(self, dnc1, dnc2, dnc3, dnc4, dnc5, kernel_size, stride):
        super(PatchGAN, self).__init__()
        self.conv1 = nn.Conv2d(2, dnc1, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dnc1, dnc2, kernel_size=kernel_size, padding=1, bias=True)
        self.conv3 = nn.Conv2d(dnc2, dnc3, kernel_size=kernel_size, padding=1, bias=True)
        self.conv4 = nn.Conv2d(dnc3, dnc4, kernel_size=kernel_size, padding=1, bias=True)
        self.conv5 = nn.Conv2d(dnc4, dnc5, kernel_size=kernel_size, padding=1, bias=True)
        self.conv6 = nn.Conv2d(dnc5, dnc5, kernel_size=kernel_size, padding=1, bias=True)
        self.conv7 = nn.Conv2d(dnc5, 1, kernel_size=kernel_size, padding=1, bias=True)
        self.Relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=1)
        self.BN1 = nn.BatchNorm2d(dnc2)
        self.BN2 = nn.BatchNorm2d(dnc3)
        self.BN3 = nn.BatchNorm2d(dnc4)
        self.BN4 = nn.BatchNorm2d(dnc5)
        self.BN5 = nn.BatchNorm2d(dnc5)
        self.activation = nn.Sigmoid()

    def forward(self, input, target):
        x1 = torch.cat((input, target), dim=1)
        x2 = self.conv1(x1)
        x2 = self.maxpool(x2)
        x3 = self.Relu(x2)

        x4 = self.conv2(x3)
        x4 = self.maxpool(x4)
        x5 = self.BN1(x4)
        x6 = self.Relu(x5)

        x7 = self.conv3(x6)
        x7 = self.maxpool(x7)
        x8 = self.BN2(x7)
        x9 = self.Relu(x8)

        x10 = self.conv4(x9)
        x11 = self.BN3(x10)
        x12 = self.Relu(x11)

        x13 = self.conv5(x12)
        x13 = self.maxpool(x13)
        x14 = self.BN4(x13)
        x15 = self.Relu(x14)

        x16 = self.conv6(x15)
        x17 = self.BN5(x16)
        x18 = self.Relu(x17)

        x19 = self.conv7(x18)
        out = self.activation(x19)
        return out

# Unet Generator
def dec_block(in_c, out_c, kernel_size, stride, drop=True):
    if drop:
        conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.Dropout(p=0.5, inplace=True)
            )
        return conv
    else:
        conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_c)
        )
        return conv
def enc_block(in_c, out_c, kernel_size, stride, BN=True):
    if BN:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        return conv
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        return conv
def bottleneck(in_c, out_c, kernel_size, stride):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
        nn.ReLU(inplace=True)
    )
    return conv
def end_block(in_c, out_c, kernel_size, stride):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1, bias=True),
        nn.Tanh() # nn.Sigmoid()
    )
    return conv
class Unet(nn.Module):
    def __init__(self, gnc1, gnc2, gnc3, gnc4, kernel_size, stride):
        super(Unet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.enc1 = enc_block(1, gnc1, kernel_size, stride, BN=False)
        self.enc2 = enc_block(gnc1, gnc2, kernel_size, stride, BN=True)
        self.enc3 = enc_block(gnc2, gnc3, kernel_size, stride, BN=True)
        self.enc4 = enc_block(gnc3, gnc4, kernel_size, stride, BN=True)
        self.enc5 = enc_block(gnc4, gnc4, kernel_size, stride, BN=True)
        self.enc6 = enc_block(gnc4, gnc4, kernel_size, stride, BN=True)
        self.enc7 = enc_block(gnc4, gnc4, kernel_size, stride, BN=True)

        self.bottleneck = bottleneck(gnc4, gnc4, kernel_size, stride)

        self.dec1 = dec_block(gnc4, gnc4, kernel_size, stride, drop=True)
        self.dec2 = dec_block(2*gnc4, gnc4, kernel_size, stride, drop=True)
        self.dec3 = dec_block(2*gnc4, gnc4, kernel_size, stride, drop=True)
        self.dec4 = dec_block(2*gnc4, gnc4, kernel_size, stride, drop=False)
        self.dec5 = dec_block(2*gnc4, gnc3, kernel_size, stride, drop=False)
        self.dec6 = dec_block(2*gnc3, gnc2, kernel_size, stride, drop=False)
        self.dec7 = dec_block(2*gnc2, gnc1, kernel_size, stride, drop=False)

        self.out = end_block(2*gnc1, 1, kernel_size, stride)

    def forward(self, image):
        # Encoder
        x1 = self.enc1(image)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
 
        # Bottleneck
        x8 = self.bottleneck(x7)

        # Decoder
        x9 = self.dec1(x8)
        x10 = self.relu(torch.cat((x9, x7), 1))
        x11 = self.dec2(x10)
        x12 = self.relu(torch.cat((x11, x6), 1))
        x13 = self.dec3(x12)
        x14 = self.relu(torch.cat((x13, x5), 1))
        x15 = self.dec4(x14)
        x16 = self.relu(torch.cat((x15, x4), 1))
        x17 = self.dec5(x16)
        x18 = self.relu(torch.cat((x17, x3), 1))
        x19 = self.dec6(x18)
        x20 = self.relu(torch.cat((x19, x2), 1))
        x21 = self.dec7(x20)
        x22 = self.relu(torch.cat((x21, x1), 1))
        # Tanh activation
        out = self.out(x22)
        return out

# Pix2Pix GAN
# class cGAN(nn.Module):
#     def __init__(self):
#         super(cGAN, self).__init__()
#         self.generator = Unet(config.gnc1, config.gnc2, config.gnc3, config.gnc4)
#         self.discriminator = PatchGAN(config.dnc1, config.dnc2, config.dnc3, config.dnc4)

#     def forward(self, image):
#         gen_out = self.generator(image)
#         dis_out = self.discriminator(image, gen_out)
#         return dis_out, gen_out

# Set device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
device = get_device()
print(device)

# Training
def train(epoch, training_DataLoader, generator, discriminator, optimizer_G, optimizer_D, criterion_GAN, criterion_pixwise, config, save):
    discriminator.train()  # train mode
    generator.train()

    d_losses = []
    g_losses = []
    pix_acc = []

    batch_iter = tqdm(enumerate(training_DataLoader), 'Training', total=len(training_DataLoader), leave=False)

    batch = 0
    for i, (x, y) in batch_iter:
        if i == 0:
            batch += 1  # batch counter

        input, target = x.to(device), y.to(device)

        # Training generator
        optimizer_G.zero_grad()
        fake_B = generator(input)

        # Saving training output images
        if epoch % 20 == 0 and save:
            for k in range(int(list(input.size())[0])):
                img1 = torch.unsqueeze(input[k, 0, :, :], 2)
                img1 = img1 * 255
                img1 = img1.cpu()
                np_img1 = img1.detach().numpy()
                np_img1 = cv2.copyMakeBorder(np_img1, pad, pad, pad, int(0.5*pad), cv2.BORDER_CONSTANT)
                img2 = torch.unsqueeze(target[k, 0, :, :], 2)
                img2 = img2 * 255
                img2 = img2.cpu()
                np_img2 = img2.detach().numpy()
                np_img2 = cv2.copyMakeBorder(np_img2, pad, pad, int(0.5*pad), int(0.5*pad), cv2.BORDER_CONSTANT)
                img3 = torch.unsqueeze(fake_B[k, 0, :, :], 2)
                img3 = img3 * 255
                img3 = img3.cpu()
                np_img3 = img3.detach().numpy()
                np_img3 = cv2.copyMakeBorder(np_img3, pad, pad, int(0.5*pad), pad, cv2.BORDER_CONSTANT)
                img = cv2.hconcat([np_img1, np_img2, np_img3])
                cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/strt2ftprnt_train_out/' # replace with your own path
                            + 'train' + str(epoch) + '-' + str(i + 1) + '-' + str(k + 1) + '.jpeg', img)

        pred_fake = discriminator(fake_B, input)

        vld = torch.tensor(np.ones((input.size(0), input.size(1), pred_fake.size(2), pred_fake.size(3))), requires_grad=False)
        fk = torch.tensor(np.zeros((input.size(0), input.size(1), pred_fake.size(2), pred_fake.size(3))), requires_grad=False)
        valid = vld.to(device)
        fake = fk.to(device)

        # GAN loss
        loss_GAN = criterion_GAN(pred_fake.float(), valid.float())
        # Pixel-wise loss
        loss_pixel = criterion_pixwise(fake_B.float(), target.float())

        acc_value = 1 - loss_pixel.item()
        pix_acc.append(acc_value)

        # Total loss
        loss_G = loss_GAN + config.lambda_pix * loss_pixel
        loss_G = loss_G.float()

        loss_value = loss_G.item()
        g_losses.append(loss_value)

        loss_G.backward()
        optimizer_G.step()

        # Training discriminator
        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(target, input)
        loss_real = criterion_GAN(pred_real.float(), valid.float())

        # Fake loss
        pred_fake = discriminator(fake_B.detach().float(), input.float())
        loss_fake = criterion_GAN(pred_fake.float(), fake.float())

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D = loss_D.float()

        loss_value = loss_D.item()
        d_losses.append(loss_value)

        loss_D.backward()
        optimizer_D.step()

        batch_iter.set_description(f'Training: (loss_G {loss_G:.4f})'
                                    f'(loss_D {loss_D:.4f})(accuracy {acc_value:.4f})')  # update progressbar

    batch_iter.close()

    return g_losses, d_losses, pix_acc

def validate(epoch, validation_DataLoader, generator, discriminator, criterion_GAN, criterion_pixwise, config, save):
    discriminator.eval()  # evaluation mode
    generator.eval()

    val_d_losses = []
    val_g_losses = []
    val_pix_acc = []

    batch_iter = tqdm(enumerate(validation_DataLoader), 'Validation', total=len(validation_DataLoader), leave=False)


    for i, (x, y) in batch_iter:
        input, target = x.to(device), y.to(device)

        with torch.no_grad():
            # evaluate generator
            fake_B = generator(input)

            # Saving validation output images
            if epoch % 20 == 0 and save:
                for k in range(int(list(input.size())[0])):
                    img1 = torch.unsqueeze(input[k, 0, :, :], 2)
                    img1 = img1 * 255
                    img1 = img1.cpu()
                    np_img1 = img1.detach().numpy()
                    np_img1 = cv2.copyMakeBorder(np_img1, pad, pad, pad, int(0.5 * pad), cv2.BORDER_CONSTANT)
                    img2 = torch.unsqueeze(target[k, 0, :, :], 2)
                    img2 = img2 * 255
                    img2 = img2.cpu()
                    np_img2 = img2.detach().numpy()
                    np_img2 = cv2.copyMakeBorder(np_img2, pad, pad, int(0.5 * pad), int(0.5 * pad),
                                                    cv2.BORDER_CONSTANT)
                    img3 = torch.unsqueeze(fake_B[k, 0, :, :], 2)
                    img3 = img3 * 255
                    img3 = img3.cpu()
                    np_img3 = img3.detach().numpy()
                    np_img3 = cv2.copyMakeBorder(np_img3, pad, pad, int(0.5 * pad), pad, cv2.BORDER_CONSTANT)
                    img = cv2.hconcat([np_img1, np_img2, np_img3])
                    cv2.imwrite('D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/strt2ftprnt_val_out/' # replace with your own path
                                + 'train' + str(epoch) + '-' + str(i + 1) + '-' + str(k + 1) + '.jpeg', img)

            pred_fake = discriminator(fake_B, input)

            vld = torch.tensor(np.ones((input.size(0), input.size(1), pred_fake.size(2), pred_fake.size(3))), requires_grad=False)
            fk = torch.tensor(np.zeros((input.size(0), input.size(1), pred_fake.size(2), pred_fake.size(3))), requires_grad=False)
            valid = vld.to(device)
            fake = fk.to(device)

            # GAN loss
            loss_GAN = criterion_GAN(pred_fake.float(), valid.float())
            # Pixel-wise loss
            loss_pixel = criterion_pixwise(fake_B.float(), target.float())

            acc_value = 1 - loss_pixel.item()
            val_pix_acc.append(acc_value)

            # Total loss
            loss_G = loss_GAN + config.lambda_pix * loss_pixel
            loss_G = loss_G.float()

            loss_value = loss_G.item()
            val_g_losses.append(loss_value)

            # evaluate discriminator
            # Real loss
            pred_real = discriminator(target, input)
            loss_real = criterion_GAN(pred_real.float(), valid.float())

            # Fake loss
            pred_fake = discriminator(fake_B.detach().float(), input.float())
            loss_fake = criterion_GAN(pred_fake.float(), fake.float())

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D = loss_D.float()

            loss_value = loss_D.item()
            val_d_losses.append(loss_value)

            batch_iter.set_description(f'Validation: (loss_G {loss_G:.4f})'
                                        f'(loss_D {loss_D:.4f})(accuracy {acc_value:.4f})')  # update progressbar

    batch_iter.close()

    return val_g_losses, val_d_losses, val_pix_acc

def run_trainer(training_dataset, validation_dataset, save=False): 
    # hyperparameters
    hyperparameters = dict(num_epochs = 5,
                        g_lr = 1.5e-4,
                        d_lr = 2e-3,
                        lr_decay = 0.1,
                        patch = 16,
                        lambda_pix = 100,
                        b1 = 0.5,
                        b2 = 0.999,
                        batch_S = 20,
                        dnc1 = 512, 
                        dnc2 = 128, 
                        dnc3 = 16, 
                        dnc4 = 256,
                        dnc5 = 256,
                        gnc1 = 32, 
                        gnc2 = 32, 
                        gnc3 = 16, 
                        gnc4 = 256,
                        kernel_size = 3,
                        kernel_size_unet = 4,
                        stride = 2)

    wandb.init(config=hyperparameters)
    # Save model inputs and hyperparameters
    config = wandb.config

    # dataloaders
    training_DataLoader = data.DataLoader(dataset=training_dataset, batch_size=config.batch_S, shuffle=True)
    validation_DataLoader = data.DataLoader(dataset=validation_dataset, batch_size=config.batch_S, shuffle=False)

    # model
    generator = Unet(config.gnc1, config.gnc2, config.gnc3, config.gnc4, config.kernel_size_unet, config.stride)
    discriminator = PatchGAN(config.dnc1, config.dnc2, config.dnc3, config.dnc4, config.dnc5, config.kernel_size, config.stride)
    generator.to(device)
    discriminator.to(device)
    # Log gradients and model parameters wandb
    wandb.watch(generator, log="all", log_freq=10)
    wandb.watch(discriminator, log="all", log_freq=10)

    # criterion
    criterion_GAN = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    criterion_pixwise = nn.L1Loss()

    # optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.b1, config.b2), weight_decay=1e-05)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.b1, config.b2))

    # Scheduler
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=10,
                                                    threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                    min_lr=0, eps=1e-07, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.1, patience=10,
                                                    threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                    min_lr=0, eps=1e-07, verbose=True)

    d_loss = []
    g_loss = []
    pix_accuracy = []
    val_d_loss = []
    val_g_loss = []
    val_pix_accuracy = []

    progressbar = trange(config.num_epochs, desc='Progress')
    epoch = 0
    for i in progressbar:
        epoch += 1  # epoch counter

        # Training
        g_losses, d_losses, pix_acc = train(epoch, training_DataLoader, generator, discriminator, 
                                            optimizer_G, optimizer_D, criterion_GAN, criterion_pixwise, 
                                            config, save)
        g_loss.append(np.mean(g_losses))
        d_loss.append(np.mean(d_losses))
        pix_accuracy.append(np.mean(pix_acc))

        # Validation
        if validation_DataLoader is not None:
            val_g_losses, val_d_losses, val_pix_acc = validate(epoch, validation_DataLoader, generator, discriminator, 
                                                                criterion_GAN, criterion_pixwise, config, save)
            val_g_loss.append(np.mean(val_g_losses))
            val_d_loss.append(np.mean(val_d_losses))
            val_pix_accuracy.append(np.mean(val_pix_acc))

        # Save models
        if epoch % 10 == 0 and save:
            PATH_G = 'D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/DL model/strt2ftprnt_gen_' + \
                    str(epoch) + 'e_' + str(int(100*pix_accuracy[i])) + 't-acc_' + \
                    str(int(100*val_pix_accuracy[i])) + 'v-acc_run02.pt'
            PATH_D = 'D:/AI in Urban Design/DL UD dir/STRT2FTPRNT/DL model/strt2ftprnt_dis_' + \
                    str(epoch) + 'e_' + str(int(100*pix_accuracy[i])) + 't-acc_' + \
                    str(int(100*val_pix_accuracy[i])) + 'v-acc_run02.pt'
            torch.save(generator.state_dict(), PATH_G)
            torch.save(discriminator.state_dict(), PATH_D)

        # Log metrics to visualize performance wandb
        wandb.log({
            'epoch': epoch,
            'train_g_loss': g_loss[i],
            'train_d_loss': d_loss[i],
            'train_accuracy': pix_accuracy[i],
            'val_g_loss': val_g_loss[i],
            'val_d_loss': val_d_loss[i],
            'val_accuracy': val_pix_accuracy[i], 
            'g_lr': optimizer_G.param_groups[0]['lr'],
            'd_lr': optimizer_D.param_groups[0]['lr']
        })

        # Learning rate scheduler 
        if scheduler_G is not None:
            if validation_DataLoader is not None and scheduler_G.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler_G.step(val_g_loss[i])  # learning rate scheduler step with generator val loss
            else:
                scheduler_G.step()  
        if scheduler_D is not None:
            if validation_DataLoader is not None and scheduler_D.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler_D.step(val_d_loss[i])  # learning rate scheduler step with discriminator val loss
            else:
                scheduler_D.step() 
    
    # empty cuda memory
    torch.cuda.empty_cache()


if __name__ == "__main__":
    sweep = True # if false train and save outputs and models to local directories
    if sweep:
        sweep_config = {
                'method': 'bayes', #grid, random, bayes
                'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize'   
                },
                'parameters': {
                    'g_lr': {
                        'min': 1e-4, 'max': 1e-3  # 'values':[1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
                    },
                    'd_lr': {
                        'min': 1e-3, 'max': 9e-2
                    },
                    'dnc1':{
                        'values':[256, 512]
                    },
                    'dnc2':{
                        'values':[128, 256, 512]
                    },
                    'dnc3':{
                        'values':[16, 32, 64]
                    },
                    'dnc4':{
                        'values':[16, 32, 64, 128, 256]
                    },
                    'dnc5':{
                        'values':[16, 32, 64, 128, 256]
                    },
                    'gnc1':{
                        'values':[16, 32, 64, 128]
                    },
                    'gnc2':{
                        'values':[32, 64, 128, 256, 512]
                    },
                    'gnc3':{
                        'values':[32, 64, 128, 256, 512]
                    },
                    'gnc4':{
                        'values':[32, 64, 128, 256, 512]
                    }
                }
            }
        sweep_id = wandb.sweep(sweep_config, project='strt2ftprnt')
        wandb.agent(sweep_id, function=lambda:run_trainer(training_dataset, validation_dataset))

    else:
        # train
        run_trainer(training_dataset, validation_dataset, save=True)

