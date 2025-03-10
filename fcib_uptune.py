import argparse
import random
import os
from glob import glob
from tqdm import tqdm

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.v2 as T

import datetime
# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from fmcib.models import fmcib_model
from extractor_transform import CenterPadTransform

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

TRAIN_SPLIT = 'train'
TEST_SPLIT = 'test'
TQDM_DISABLE = False
FCIB_OUTPUT_SIZE = 4096
NUM_CLASSES = 2

TARGET_SIZE = 50

exp_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# checkpoint_dir = Path(f'./checkpoints/{exp_time}')
log_dir = 'runs/full'
writer = None

class FCIBUptune(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = fmcib_model()

        if config['fine_tune_mode'] == 'head':
            for param in self.model.parameters():
                param.requires_grad = False

        self.classifier_dropout = torch.nn.Dropout(config['dropout_prob'])
        self.classifier_interim = torch.nn.Linear(config['fcib_output_size'], config['head_interim_size'])
        self.interim_af = F.gelu
        self.classifier_head = torch.nn.Linear(config['head_interim_size'], config['num_classes'])

    def forward(self, input):
        output = self.model(input)
        output_dropout = self.classifier_dropout(output)
        classifier_interim = self.classifier_interim(output_dropout)
        interim_activated = self.interim_af(classifier_interim)
        classifier_output = self.classifier_head(interim_activated)
        return classifier_output, classifier_interim



class FCIBTuneDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, transform=None):
        super().__init__()
        # self.dataset_dir = dataset_dir
        # self.split = split
        self.dataset_dir = os.path.join(dataset_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []

        file_paths = sorted(glob(os.path.join(self.dataset_dir, "*.npy")))
        # Load images and assign labels based on filename
        for file_path in file_paths:
            image = np.load(file_path).astype(np.float32)
            image_tensor = torch.tensor(image, dtype=torch.float32)
            self.images.append(image_tensor)  # Load .npy file

            # Assign labels based on filename prefix
            filename = os.path.basename(file_path)
            if filename.startswith("lesion_"):
                # label = torch.tensor([1, 0], dtype=torch.float32)
                label = 1
            elif filename.startswith("free_"):
                # label = torch.tensor([0, 1], dtype=torch.float32)
                label = 0
            else:
                raise ValueError(f"Unexpected filename format: {filename}")
            self.labels.append(label)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
            image = image.unsqueeze(0)
        label = self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.images)


# Evaluate the model on dev examples.
def model_eval(args, dataloader, model, device):
    model.eval()  # Switch to eval model, will turn off randomness like dropout.
    y_true = []
    y_pred = []
    cumulative_loss = 0
    num_batches = 0
    for step, (images, labels) in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        images = images.to(device)
        # labels = labels.to(device)

        logits, _ = model(images)
        logits = logits.detach().cpu()
        loss = F.cross_entropy(
            logits, labels, reduction='sum',
            weight=torch.tensor([1.0, args.tumor_class_weight])) / args.batch_size
        cumulative_loss += loss.item()
        num_batches += 1

        preds = np.argmax(logits.numpy(), axis=1).flatten()
        # labels_true = np.argmax(labels, axis=1).flatten()
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds)

    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return cumulative_loss / num_batches, acc, f1, precision, recall, y_pred, y_true


def save_model(model, optimizer, args, config):
    # save_info = {
    #     'model': model.state_dict(),
    #     'optim': optimizer.state_dict(),
    #     'args': args,
    #     'system_rng': random.getstate(),
    #     'numpy_rng': np.random.get_state(),
    #     'torch_rng': torch.random.get_rng_state(),
    # }
    os.makedirs(args.weight_save_path, exist_ok=True)
    out_file = os.path.join(args.weight_save_path, f'weights_{get_model_name(args)}-{exp_time}')
    torch.save(model.state_dict(), f'{out_file}.gpu')
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict_cpu, f'{out_file}.cpu')


def train(args):
    train_transform = transforms.Compose([
        CenterPadTransform(target_size=(TARGET_SIZE, TARGET_SIZE, TARGET_SIZE), do_random_shift=True),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.5),
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomAffine(degrees=15, scale=(0.9, 1.1)),
        # T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    ])

    test_transform = transforms.Compose([
        CenterPadTransform(target_size=(TARGET_SIZE, TARGET_SIZE, TARGET_SIZE))
    ])

    train_dataset = FCIBTuneDataset(args.dataset_dir, TRAIN_SPLIT, train_transform)
    val_dataset = FCIBTuneDataset(args.dataset_dir, TEST_SPLIT, test_transform)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    config = {'dropout_prob': args.dropout_prob,
              'fcib_output_size': FCIB_OUTPUT_SIZE,
              'head_interim_size': args.head_interim_size,
              'num_classes': NUM_CLASSES,
              'fine_tune_mode': args.fine_tune_mode,
              }
    model = FCIBUptune(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'Running model on a device {device}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_eval_acc = 0

    img_ctr = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for images, labels in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            # loss = F.cross_entropy(logits, labels.view(-1), reduction='sum') / args.batch_size
            loss = F.cross_entropy(
                logits, labels, reduction='sum',
                weight=torch.tensor([1.0, args.tumor_class_weight]).to(device)) / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            # Increase the total number of images on which we trained the model
            img_ctr += num_batches * args.batch_size

        train_loss = train_loss / (num_batches)
        train_eval_loss, train_acc, train_f1, train_precision, train_recall, train_y_pred, train_y_true = model_eval(
            args, train_dataloader, model, device)
        eval_eval_loss, eval_acc, eval_f1, eval_precision, eval_recall, eval_y_pred, eval_y_true = model_eval(
            args, val_dataloader, model, device)

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            save_model(model, optimizer, args, config)

        print()
        print(f"Epoch {epoch}")
        print("Train loss :: {train_loss :.3f}, F1 :: {train_f1 :.3f}, Precision :: {train_precision :.3f}, Recall :: {train_recall :.3f}")
        print("Eval  loss :: {eval_loss :.3f},  F1 :: {eval_f1 :.3f},  Precision :: {eval_precision :.3f},  Recall :: {eval_recall :.3f}")
        print(f'Truth   sample: f{eval_y_true[0:40]}')
        print(f'Predict sample: f{eval_y_pred[0:40]}')

        writer.add_scalar('Training Loss', train_loss, img_ctr)
        writer.add_scalar('Evaluation Loss', eval_eval_loss, img_ctr)
        writer.add_scalar('Training F1', train_f1, img_ctr)
        writer.add_scalar('Evaluation F1', eval_f1, img_ctr)


    with open('train_summary.txt', "a+") as file:
        file.write(f'{get_model_name(args)}, train F1 :: {train_f1 :.3f}, eval F1 :: {eval_f1 :.3f}\n')
        file.write(f'Truth   sample: f{eval_y_true[0:40]}\n')
        file.write(f'Predict sample: f{eval_y_pred[0:40]}\n\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dataset_dir", type=str, default='/mnt/data/GammaKnife/fcib_finetune')
    parser.add_argument("--weight_save_path", type=str, default='/mnt/data/GammaKnife/fcib_finetune_weights')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout_prob", type=float, default=0.3)
    parser.add_argument("--head_interim_size", type=int, default=20)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--tumor_class_weight", type=float, help="In the loss, weight applied to a tumor class",
                        default=3.0)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='head:  task specific head parameters are updated; full-model: all parameters are updated as well',
                        choices=('head', 'full'), default="head")
    args = parser.parse_args()
    return args


def get_model_name(args):
    return f'e{args.epochs}_bs{args.batch_size}_dp{args.dropout_prob}_his{args.head_interim_size}_lr{args.lr}_ftm{args.fine_tune_mode}_ctw{args.tumor_class_weight}'

if __name__ == "__main__":
    args = get_args()
    print('Arguments: ')
    print(args)
    writer = SummaryWriter(log_dir=os.path.join(
        log_dir, f'experiment_{get_model_name(args)}-{exp_time}'))
    train(args)
