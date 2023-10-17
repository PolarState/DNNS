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

EPOCHS = 2
SEEDS = 3

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

def fancy_pca(img):
    print(img)
    covariance_matrix = torch.cov(img)
    (eigenvalues, eigenvectors) = torch.linalg.eig(covariance_matrix)
    sort_perm = eigenvalues[::-1].argsort()
    eigenvalues[::-1].sort()
    eigenvectors = eigenvectors[:, sort_perm]

    # get [p1, p2, p3]
    m1 = torch.column_stack((eigenvectors))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = torch.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    alpha = np.random.normal(0, 0.1)

    # broad cast to speed things up
    m2[:, 0] = alpha * eigenvalues[:]

    print(m2)
    print(m1)

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = m1 * m2

    print(add_vect)

    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)


os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
# One of the pooling operations is not deterministic. So we enable warn only.
torch.use_deterministic_algorithms(mode=True, warn_only=True)

# Function to initalize weights in network via normal distribution.
def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)

resume_model = None
resume_seed = 0
resume_epoch = 0
if resume:
    # List all experiment folders. Experiments are ordered by seed.
    checkpoint_folders = list(filter(lambda x: x.startswith("seed_"), os.listdir(output_path)))
    checkpoint_numbers = sorted([int(folder[len("seed_"):]) for folder in checkpoint_folders], reverse=True)

    if len(checkpoint_numbers) == SEEDS:
        print("All experiments were completed.")
        exit()
    if len(checkpoint_numbers) == 0:
        print(f"No experiments to resume at {output_path}.")
        exit()

    latest_writer_datetime = datetime.max
    latest_writer_name = None
    for number in checkpoint_numbers:
        latest_folder = os.listdir(f"{output_path}/seed_{number}")
        for file in latest_folder:
            try:
                file_time = datetime.strptime(file, '%Y%m%d_%H%M%S')
                if latest_writer_datetime > file_time:
                    latest_writer_datetime = file_time
                    latest_writer_name = file
                    resume_seed = number
                
            except(ValueError):
                print(f"Cannot extract datetime from file: {output_path}/seed_{number}/{file}")
        
        if latest_writer_datetime == datetime.max:
            print(f"No valid writers to resume for seed {number}, trying next seed.")
        else:
            print(f"Found writer to resule for seed {number}. Finding model.")
            break

    if latest_writer_datetime == datetime.max:
        print(f"No valid writers found for any seed.")
        exit()

    model_files = os.listdir(f"{output_path}/seed_{number}/models")
    model_files = sorted(model_files, reverse=True)
    if len(model_files) == 0:
        print("No models found.")
        exit()

    resume_model = torch.load(f"{output_path}/seed_{number}/models/{model_files[0]}")
    resume_epoch = model_files[0][16:]

for seed in range(resume_seed, SEEDS):
    experiment_path = f'{output_path}/seed_{seed}/'
    modeling_path = f'{experiment_path}/models'
    os.makedirs(modeling_path, exist_ok=True)

    # Set progressive seeds for model initalization.
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    if resume and resume_model is not None:
        model = resume_model
    else:
        model = torchvision.models.AlexNet()
        model.apply(init_weights) # Initialize weights with specific function. 
    model.eval().cuda()  # Needs CUDA, don't bother on CPUs

    # Set fixed seeds for everything else.
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)

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
                    transforms.FiveCrop(224),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.ColorJitter(),
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
                num_workers=8,
                shuffle=False,
                drop_last=False,
                pin_memory=True
            )
    training_dataloader = DataLoader(
                training_dataset,
                batch_size=128,
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
        for i, data in enumerate(training_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = data # input is a 5d tensor, labels is 2d

            _, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w) # fuse batch size and ncrops
            labels = torch.reshape(torch.stack([labels for _ in range(ncrops)]).T, (-1,))
            inputs, labels = inputs.cuda(), labels.cuda() # add this line

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
            # if i % 10 == 9:
                last_loss = running_loss / 1000 # loss per batch
                # last_loss = running_loss / 10 # loss per batch
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
    for epoch in range(resume_epoch, EPOCHS):
        print('EPOCH {}:'.format(epoch))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
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

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{modeling_path}/{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)
