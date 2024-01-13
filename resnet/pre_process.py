import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import transforms
import torch
import torchvision
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from PIL import ImageFile
import time

# ImageFile.LOAD_TRUNCATED_IMAGES = True

EPOCHS = 90
SEEDS = 20

parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--dataset", required=True,
                    help="Path to folder containing image-net.")
parser.add_argument("-o", "--output", required=True,
                    help="Path to output folder.")
parser.add_argument("-r", "--resume", default=False,
                    help="Resume training.",
                    action=argparse.BooleanOptionalAction)

args = parser.parse_args()
config = vars(args)
dataset_folder_path = config['dataset']
output_path = config['output']
resume = config['resume']

# image-net 2012 specific values.
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
validation_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

training_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.FiveCrop(224),
                # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

print("Starting dataset load.")
dataset_timer = datetime.now()
# training_dataset = torchvision.datasets.imagenet.ImageNet(dataset_folder_path, split="val", transform=training_transform)
training_dataset = torchvision.datasets.imagenet.ImageNet(dataset_folder_path, split="train", transform=training_transform)
validation_dataset = torchvision.datasets.imagenet.ImageNet(dataset_folder_path, split="val", transform=validation_transform)
print(f"Dataset loaded in: {datetime.now() - dataset_timer}")

validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=128,
            num_workers=16,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
training_dataloader = DataLoader(
            training_dataset,
            batch_size=64,
            num_workers=8,
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=0.00001)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    batch_timer = datetime.now()
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    
    end = time.time()
    for i, data in enumerate(training_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data # input is a 5d tensor, labels is 2d
        # print(f"data time: {time.time() - end}")
        # end = time.time()

        # _, ncrops, c, h, w = inputs.size()
        # inputs = inputs.view(-1, c, h, w) # fuse batch size and ncrops
        # labels = torch.reshape(torch.stack([labels for _ in range(ncrops)]).T, (-1,))
        # print(f"reshape time: {time.time() - end}")
        # end = time.time()
        inputs, labels = inputs.cuda(), labels.cuda() # add this line

        # print(f"cuda xfer time: {time.time() - end}")
        # end = time.time()
        # Zero your gradients for every batch!
        model.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # print(f"cuda time: {time.time() - end}")
        # end = time.time()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f'  batch {i + 1} loss: {last_loss} duration: {datetime.now() - batch_timer}')
            batch_timer = datetime.now()
            tb_x = epoch_index * len(training_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'{experiment_path}/{timestamp}')

best_vloss = 1_000_000.

epoch_timer = datetime.now()
times = []
for epoch in range(resume_epoch, EPOCHS):
    print('EPOCH {}:'.format(epoch))
    torch.cuda.synchronize()
    start_epoch = time.time()

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    times.append(elapsed)
    print(elapsed)
    avg_loss = train_one_epoch(epoch, writer)

    running_vloss = 0.0

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_dataloader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.cuda(), vlabels.cuda()
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss} duration: {datetime.now() - epoch_timer}')
    epoch_timer = datetime.now()

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch)
    writer.flush()
    
    writer.add_scalar('LearningRate', optimizer.param_groups[0]["lr"], epoch)
    scheduler.step(avg_vloss)

    # Track best performance
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss

    # Save the model's state.
    model_path = f'{modeling_path}/{timestamp}_{epoch}'
    torch.save(model, model_path)
