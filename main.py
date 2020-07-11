from src.dataset import VRMDataset, label_mapper_VRM
from src.word2vec import word2vec
from src import vrm_config
from src.model import build_bilstm_ram, ChunkCrossEntropyLoss_VRM
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import argparse
import csv
import tqdm

PRECISION_THRESHOLD = 0.70
IGNORE_INDEX = -1

def str2bool(s):
    return s.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='VRM Recognition Training With Pytorch')

parser.add_argument('--command', default="train", help="train or test")
parser.add_argument('--net', default="BiLSTM-RAM", help="The network architecture")
parser.add_argument('--multiplier', default=1, type=int, help='hidden size multiplier')

# Params for Optimizer
parser.add_argument('--optim',default='SGD', help='Optimizer type')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optim')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for optim')
parser.add_argument('--beta', default=1e-5, type=float, help='Beta for optim')
parser.add_argument('--betas_0', default=0.9, type=float, help='Betas for Adam')
parser.add_argument('--betas_1', default=0.999, type=float, help='Betas for Adam')


# Params for loading dataset
parser.add_argument('--dataset', default='VRM', help='dataset configuration')
parser.add_argument('--data_path', help='dataset path')
parser.add_argument('--embedding_path',help='get embedding from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str, help="Scheduler. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="40,60", type=str, help="milestones for MultiStepLR. percentage values.")
parser.add_argument('--fixed_milestones')
# Params for Cosine Annealing
parser.add_argument('--t_max', default=100, type=float, help='T_max value for Cosine Annealing Scheduler. percentage values')

# Train and Test params
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int, help='the number epochs')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=1.0, type=float, help='the number epochs')
parser.add_argument('--logdir', default='log/', help='Directory for logging')
parser.add_argument('--log_stride', default=100, type=int, help='Logging steps')
parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--checkpoint_folder', default='checkpoint/', help='Directory for saving checkpoint models')
parser.add_argument('--checkpoint_stride', default=10, type=int, help='saving model epochs')
parser.add_argument('--checkpoint_path', default=None, help='Explicit Checkpoint path input by user')



args = parser.parse_args()

# not support multi-gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

def train(model,
            device,
            dataloader,
            criterion,
            optimizer,
            scheduler,
            writer,
            checkpoint_dir,
            epoch_size,
            log_stride,
            checkpoint_stride,
            use_chunk=False):

    best_precision = 0
    best_model_state = None
    previous_lr = None
    for epoch in tqdm.tqdm(range(epoch_size)):

        model.train()
        scheduler.step()
        average_loss = 0
        last_log = 0

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar("lr/train",current_lr,epoch)

        if previous_lr and previous_lr != current_lr:
            model.load_state_dict(best_model_state)

        previous_lr = current_lr

        total = len(dataloader)

        for n, data in enumerate(tqdm.tqdm(dataloader)):
            optimizer.zero_grad()

            utterances, labels = model.formatter(data)

            utterances = [utterance.to(device) for utterance in utterances]
            labels = labels.to(device)
            preds = model(utterances)

            loss = criterion(preds, labels)
            average_loss += loss

            loss.backward()
            optimizer.step()

            if (n+1) % log_stride == 0:

                last_log = (average_loss/log_stride).cpu().item()
                writer.add_scalar("loss/train", last_log, epoch*len(dataloader)+n)
                average_loss = 0       


        if (epoch+1)% checkpoint_stride == 0:
            torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(epoch)),
                )


    torch.save(model.state_dict(), os.path.join(checkpoint_dir,"checkpoint_{}.pth".format(epoch_size-1)))
    writer.close()



def converter(x, y, z):
    if x < 0:
        f = 'U'
    elif x and y and z:
        f = 'D'
    elif x and y and not z:
        f = 'E'
    elif x and not y and z:
        f = 'A'
    elif x and not y and not z:
        f = 'C'
    elif not x and y and z:
        f = 'Q'
    elif not x and y and not z:
        f = 'K'
    elif not x and not y and z:
        f = 'I'
    elif not x and not y and not z:
        f = 'R'

    return f


def get_latest_version(checkpoint_dir,work_id=None):

    ls = os.listdir(checkpoint_dir)

    if work_id:
        versions = [int(chk.split('_')[-1].split('.')[0]) for chk in ls if chk.startswith(work_id)]
    else:
        versions= [int(chk.split('_')[-1].split('.')[0]) for chk in ls]

    version = 0
    if len(versions) != 0:
        versions.sort()
        version = versions[-1]
    return version


if __name__=='__main__':

    work_id = args.net +"_"+ args.dataset
    print(args.command + " on " + work_id)

    by_conversation = False
    valby_conv = False
    lastpooling = False

    config = vrm_config
    # VRM setting
    builder = build_bilstm_ram
    by_conversation = config.by_conversation
    criterion = ChunkCrossEntropyLoss_VRM(num_chunk= config.chunk_size, weights=config.weights,ignore_index= IGNORE_INDEX)

    print("setting model...")

    model = builder(DEVICE,config,args.multiplier)
    model.to(DEVICE)

    print("getting embedding...")
    wv = word2vec(args.embedding_path,embedding_len=config.embedding_len, sent_len=config.sent_len)
    to_vector = wv.to_vector

    print("getting data...")
    df = pd.read_csv(args.data_path, error_bad_lines=False)
    print("setting dataset...")

    label_mapper = label_mapper_VRM

    dataset= VRMDataset(df,to_vector,label_mapper,config.sent_len, config.pos_len, config.max_dialogue_len,
                chunk_size=config.chunk_size,by_conversation=by_conversation)


    if args.command =='train':

        if args.resume_point:
            model.load_state_dict(torch.load(args.resume_point))

        dataloader = DataLoader(dataset,batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True)

        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.betas_0,args.betas_1),
                                weight_decay=args.weight_decay)

        else:
            optimizer = None

        if args.scheduler == 'multi-step':
            if args.fixed_milestones:
                milestones = [ int(v.strip()) for v in args.milestones.split(",")]
            else:
                milestones = [ int((int(v.strip())/100)*args.num_epochs) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,gamma=args.gamma)

        else:
            scheduler = None

        chk_version = get_latest_version(args.checkpoint_folder,work_id)+1
        log_version = get_latest_version(args.logdir,work_id)+1

        checkpoint_dir = os.path.join(args.checkpoint_folder,work_id+"_"+str(chk_version))
        os.mkdir(checkpoint_dir)
        log_dir = os.path.join(args.logdir,work_id+"_"+str(log_version))

        writer = SummaryWriter(log_dir)

        print("start training...")

        train(model,
            DEVICE,
            dataloader,
            criterion,
            optimizer,
            scheduler,
            writer,
            checkpoint_dir=checkpoint_dir,
            epoch_size=args.num_epochs,
            log_stride=args.log_stride,
            checkpoint_stride=args.checkpoint_stride)







