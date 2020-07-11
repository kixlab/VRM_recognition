from src.dataset import VRMDataset, label_mapper_VRM
from src.word2vec import word2vec
from src import vrm_config
from src.model import build_bilstm_ram, ChunkCrossEntropyLoss_VRM
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
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
parser.add_argument('--val_data_path', help='dataset path')
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

parser.add_argument('--resume_point', default=None)
parser.add_argument('--testby_conv', default=None)
parser.add_argument('--use_conv_val', default=None)


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
            val_epoch_size=10,
            val_dataloader=None,
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

        if val_epoch_size < 1 :
            val_stride = int(total*val_epoch_size)

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

            if val_epoch_size < 1:

                if (n+1) % val_stride == 0:
                    prec_f, precision = val_in_train(model, device, val_dataloader, writer)

                    if precision > PRECISION_THRESHOLD and precision > best_precision:
                        best_precision = precision
                        best_model_state = model.state_dict()
                        torch.save(
                            model.state_dict(),
                            os.path.join(checkpoint_dir, "checkpoint_{}_{}.pth".format(precision,epoch)),
                        )

                    model.train()

        if val_epoch_size > 1 and val_dataloader:

            if (epoch+1)% val_epoch_size == 0:
                prec_f, precision = val_in_train(model, device, dataloader, writer)

                if precision > PRECISION_THRESHOLD and precision > best_precision:
                    best_precision = precision
                    best_model_state = model.state_dict()
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_dir, "checkpoint_{}_{}.pth".format(precision,epoch)),
                    )


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

def val_in_train(model, device, dataloader, writer):

    model.eval()
    forms = []
    intents = []
    gt_forms = []
    gt_intents = []

    # correct_nums = [0, 0, 0, 0, 0, 0]
    # total_nums = [0, 0, 0, 0, 0, 0]
    # precisions = [0, 0, 0, 0, 0, 0]

    for n, data in enumerate(tqdm.tqdm(dataloader)):
        utterances, labels = model.formatter(data)
        utterances = [utterance.to(device) for utterance in utterances]
        labels = labels.to(device)[0]

        with torch.no_grad():
            preds = model(utterances)

    #     for axis in range(6):
    #         _labels = labels[...,axis]
    #         _preds = [p[..., 2*axis:2*axis+2] for p in preds]

    #         for i, pred in enumerate(_preds):
    #             label = _labels[0][i].item()
    #             if label != IGNORE_INDEX:
    #                 _, predicted_label = pred.max(-1)
    #                 if predicted_label.item() == label:
    #                     correct_nums[axis] += 1
    #                 total_nums[axis] += 1

    # for axis in range(6):
    #      precisions[axis] = correct_nums[axis] / total_nums[axis]

    # print(correct_nums)
    # print(total_nums)

    # return precisions

        for i, p in enumerate(preds):
            if labels[i, 0].item() != IGNORE_INDEX:
                is_correct_form = []
                gt = []
                for axis in range(3): # form
                    label = labels[i, axis].item()
                    _, predicted_label = p[...,2*axis:2*axis+2].max(-1)
                    predicted_label = predicted_label.item()
                    is_correct_form.append(int(label == predicted_label))
                    gt.append(label)

                forms.append(is_correct_form)
                gt_forms.append(gt)

            else:
                forms.append([0, 0, 0])
                gt_forms.append([-1, -1, -1])

            if labels[i, 3].item() != IGNORE_INDEX:
                is_correct_intent = []
                gt = []
                for axis in range(3, 6):  # intent
                    label = labels[i, axis].item()
                    _, predicted_label = p[..., 2 * axis:2 * axis + 2].max(-1)
                    predicted_label = predicted_label.item()
                    is_correct_intent.append(int(label == predicted_label))
                    gt.append(label)

                intents.append(is_correct_intent)
                gt_intents.append(gt)

            else:
                intents.append([0, 0, 0])
                gt_intents.append([-1, -1, -1])


    forms = np.array(forms)
    intents = np.array(intents)
    gt_forms = np.array(gt_forms)
    gt_intents = np.array(gt_intents)

    c_ind_form = np.where(np.mean(forms, axis=1) == 1)
    c_ind_intent = np.where(np.mean(intents, axis=1) == 1)
    c_ind_both = np.intersect1d(c_ind_form, c_ind_intent)

    ind_valid_form = np.where(np.mean(gt_forms, axis=1) >= 0)
    ind_valid_intent = np.where(np.mean(gt_intents, axis=1) >= 0)
    ind_valid_both = np.intersect1d(ind_valid_form, ind_valid_intent)

   # print(len(ind_valid_both))

    gt_forms_c = gt_forms[c_ind_form]
    gt_intents_c = gt_intents[c_ind_intent]

    concat_c = np.concatenate((gt_forms[c_ind_both], gt_intents[c_ind_both]), axis=1)
    concat_gt = np.concatenate((gt_forms[ind_valid_both], gt_intents[ind_valid_both]), axis=1)

    list0 = np.array(['DD', 'DE', 'DA', 'DC', 'DQ', 'DK', 'DI', 'DR',
                      'ED', 'EE', 'EA', 'EC', 'EQ', 'EK', 'EI', 'ER',
                      'AD', 'AE', 'AA', 'AC', 'AQ', 'AK', 'AI', 'AR',
                      'CD', 'CE', 'CA', 'CC', 'CQ', 'CK', 'CI', 'CR',
                      'QD', 'QE', 'QA', 'QC', 'QQ', 'QK', 'QI', 'QR',
                      'KD', 'KE', 'KA', 'KC', 'KQ', 'KK', 'KI', 'KR',
                      'ID', 'IE', 'IA', 'IC', 'IQ', 'IK', 'II', 'IR',
                      'RD', 'RE', 'RA', 'RC', 'RQ', 'RK', 'RI', 'RR'])

    list = np.array(['D', 'E', 'A', 'C', 'Q', 'K', 'I', 'R'])

    stat_c = np.zeros(8 * 8)
    stat_gt = np.zeros(8 * 8)

    for c in concat_c:
        x, y, z = c[:3]
        fl = converter(x, y, z)
        x, y, z = c[3:]
        il = converter(x, y, z)
        stat_c[np.where(list0 == ''.join([fl, il]))] += 1

    for c in concat_gt:
        x, y, z = c[:3]
        fl = converter(x, y, z)
        x, y, z = c[3:]
        il = converter(x, y, z)
        stat_gt[np.where(list0 == ''.join([fl, il]))] += 1

    #print(list0[np.where(stat_c > 0)])
    #print(list0[np.where(stat_gt > 0)])
    #print(np.where(stat_c > 0), np.where(stat_gt > 0))
    #print(stat_c)
    #print(stat_gt)
    form_c = np.zeros(8)
    intent_c = np.zeros(8)

    form_t = np.zeros(8)
    intent_t = np.zeros(8)

    for gt in gt_forms:
        x, y, z = gt
        form_t[np.where(list == converter(x, y, z))] += 1

    for gt in gt_forms_c:
        x, y, z = gt
        form_c[np.where(list == converter(x, y, z))] += 1

    for gt in gt_intents:
        x, y, z = gt
        intent_t[np.where(list == converter(x, y, z))] += 1

    for gt in gt_intents_c:
        x, y, z = gt
        intent_c[np.where(list == converter(x, y, z))] += 1

    # print(form_t)
    # print(np.sum(form_t))
    # print(form_c)
    # print(intent_t)
    # print(np.sum(intent_t))
    # print(intent_c)

    prec_f = np.sum(form_c) / np.sum(form_t)
    prec_i = np.sum(intent_c) / np.sum(intent_t)

    return prec_f, prec_i

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

    val_dataset = None
    by_conversation = False
    valby_conv = False
    lastpooling = False

    if args.use_conv_val:
        valby_conv = True

    config = vrm_config
    # VRM setting
    builder = build_bilstm_ram
    by_conversation = config.by_conversation
    criterion = ChunkCrossEntropyLoss_VRM(num_chunk= config.chunk_size, weights=config.weights,ignore_index= IGNORE_INDEX)

    if args.testby_conv:
        by_conversation = True

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

    if args.val_data_path:
        val_df = pd.read_csv(args.val_data_path, error_bad_lines=False)
        val_dataset = VRMDataset(val_df,to_vector,label_mapper,config.sent_len,config.pos_len,config.max_dialogue_len,
                chunk_size=config.chunk_size,by_conversation=by_conversation)

    if args.command =='train':

        if args.resume_point:
            model.load_state_dict(torch.load(args.resume_point))

        dataloader = DataLoader(dataset,batch_size = args.batch_size, num_workers=args.num_workers, shuffle=True)

        if val_dataset:
            val_dataloader = DataLoader(val_dataset, batch_size =1, num_workers=1, shuffle=False)

        else:
            val_dataloader= None

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
            checkpoint_stride=args.checkpoint_stride,
            val_epoch_size=args.validation_epochs,
            val_dataloader=val_dataloader)


    elif args.command =='test':
        dataloader = DataLoader(dataset,batch_size = 1, num_workers=args.num_workers, shuffle=False)
        if args.checkpoint_path:
            chk_path = args.checkpoint_path
            #output_path = os.path.join(args.output_folder,work_id+"_comm_output.csv")
            model.load_state_dict(torch.load(chk_path))
            print("load the file located at %s" % chk_path)
            print("running on test set...")
            prec_f, prec_i = val_in_train(model, DEVICE, dataloader, writer=None)
            print("form prec: %s%%, intent prec: %s%%" % (prec_f * 100, prec_i *100))

        else:
            print("No checkpoint path")




