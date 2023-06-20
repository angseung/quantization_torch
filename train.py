import os
import platform
import random
import datetime
import json
from typing import Tuple
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from models.resnet import resnet152, fuse_resnet
from utils.quantization_utils import QuantizableModel


def seed_worker(worker_id: None) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


curr_os = platform.system()
print("Current OS : %s" % curr_os)

if "Windows" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif "Darwin" in curr_os:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
elif "Linux" in curr_os:
    device = "cuda" if torch.cuda.is_available() else "cpu"

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

config = {
    "max_epoch": 200,
    "initial_lr": 0.0025,
    "train_batch_size": 64,
    "dataset": "CIFAR-10",
    "train_resume": False,
    "set_random_seed": True,
    "l2_reg": 0.0,
    "dropout_rate": [None, None, None, None],
    "scheduling": "normal",
    "augment": False,
}

if config["set_random_seed"]:
    random_seed = 1
    g = torch.Generator()
    g.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # multi-GPU
    np.random.seed(random_seed)
    # torch.use_deterministic_algorithms(True)

Dataset = config["dataset"]
max_epoch = config["max_epoch"]
batch_size = config["train_batch_size"]

# Data Preparing  !!!
print("==> Preparing data..")

if Dataset == "ImageNet":
    input_size = 224
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    trainset = torchvision.datasets.ImageNet(
        root="/data_yper/imagenet/", split="train", transform=transform_train
    )
    testset = torchvision.datasets.ImageNet(
        root="/data_yper/imagenet/", split="val", transform=transform_test
    )

# TODO: modify padding in RandomCrop
elif Dataset == "CIFAR-10":
    input_size = 32
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.ToTensor(), normalize]
    )

    if config["augment"]:
        ## override normalize weights
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        transform_train = transforms.Compose(
            [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
            ]
        )

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar-10/", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./cifar-10/", train=False, download=True, transform=transform_test
    )

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    worker_init_fn=seed_worker,
    generator=g,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0
)


# Training
def train(epoch, dir_path=None) -> Tuple[float, float]:
    print(f"\nEpoch: {epoch}, curr_lr: {scheduler.get_lr()}")
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    with tqdm(trainloader, unit="batch") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Train Epoch {epoch}")

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            tepoch.set_postfix(
                loss=train_loss / (batch_idx + 1), accuracy=100.0 * correct / total
            )

    with open(dir_path + "/log.txt", "a") as f:
        f.write(
            "Epoch [%03d] |Train| Loss: %.3f, Acc: %.3f \t"
            % (epoch, train_loss / (batch_idx + 1), 100.0 * correct / total)
        )

    return train_loss / (batch_idx + 1), 100.0 * correct / total


def test(epoch, dir_path=None) -> Tuple[float, float]:
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(testloader, unit="batch") as tepoch:
            for batch_idx, (inputs, targets) in enumerate(tepoch):
                tepoch.set_description(f"Test Epoch {epoch}")

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                tepoch.set_postfix(
                    loss=test_loss / (batch_idx + 1), accuracy=100.0 * correct / total
                )
    acc = 100.0 * correct / total

    # Save checkpoint.
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, "./" + dir_path + "/ckpt.pth")

        best_acc = acc

    with open(dir_path + "/log.txt", "a") as f:
        f.write("|Test| Loss: %.3f, Acc: %.3f \n" % (test_loss / (batch_idx + 1), acc))

    return (test_loss / (batch_idx + 1)), acc


# Model
print("==> Building model..")

resnet = resnet152()
fuse_resnet(resnet, is_qat=True)
resnet = QuantizableModel(resnet, is_qat=True).prepare()
nets = {"resnet152": resnet}

for netkey in nets.keys():
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    if config["train_resume"]:
        log_path = f"outputs/{os.listdir('outputs')[-1]}"
    else:
        log_path = f"outputs/{netkey}_{now}"

    net = nets[netkey]

    os.makedirs(log_path, exist_ok=True)

    if not config["train_resume"]:
        with open(log_path + "/log.txt", "w") as f:
            f.write(f"Networks : {netkey}_{now}\n")
            f.write("Net Train Configs: \n %s\n" % json.dumps(config))

    elif config["train_resume"]:
        with open(log_path + "/log.txt", "a") as f:
            f.write("Train resumed from this point...\n")

    # for multiple gpus...
    if device == "cuda" and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)  # Not support ONNX converting
        # cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if config["scheduling"] in ["warm", "warm_and_restart"]:
        optimizer = optim.Adam(
            net.parameters(),
            lr=1e-8,
            weight_decay=config["l2_reg"],  # for warm-up
        )
    elif config["scheduling"] == "normal":
        optimizer = optim.Adam(
            net.parameters(),
            lr=config["initial_lr"],
            weight_decay=config["l2_reg"],  # for non-warm-up
        )

    # For Original CosAnnealing...
    if config["scheduling"] == "normal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(max_epoch * 1.0),
            verbose=1,
        )

    if config["train_resume"]:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        checkpoint = torch.load(
            log_path + "/ckpt.pth", map_location=torch.device("cpu")
        )
        net.load_state_dict(checkpoint["net"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, max_epoch):
        net = net.to(device)
        train_loss, train_acc = train(epoch, log_path)
        test_loss, test_acc = test(epoch, log_path)

        scheduler.step()
