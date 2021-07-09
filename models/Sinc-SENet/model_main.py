"""
    Author: Knoxliu (dengkailiu@whu.edu.cn)
    All rights reserved.
"""
import argparse
import sys
import os
import shutil
import data_utils
import time
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import librosa
from collections import defaultdict
from functools import reduce
import math

import torch
from tensorboardX import SummaryWriter
from models import  SEResNet, MFCCModel, CQCCModel, Sincnetmodelv1, \
    Sincnetmodelv2, SpectrogramModel, sincnet_ori, CLDNN
from data_utils import ASVFile

# cut off
def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        random_start = np.random.randint(x_len - max_len)
        return x[random_start:random_start + max_len]
    # need to pad
    num_repeats = (max_len // x_len) + 1
    # float type ---> num_repeats
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# def evaluate_accuracy(data_loader, model, device):
#     num_correct = 0.0
#     num_total = 0.0
#     model.eval()
#     for batch_x, batch_y, batch_meta in data_loader:
#         batch_size = batch_x.size(0)
#         num_total += batch_size
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.view(-1).type(torch.int64).to(device)
#         batch_out = model(batch_x)
#         _, batch_pred = batch_out.max(dim=1)
#         num_correct += (batch_pred == batch_y).sum(dim=0).item()
#     return 100 * (num_correct / num_total)


def validate(val_loader, model, device, focal_obj):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (batch_x, batch_y, _) in enumerate(val_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # compute output
            output = model(batch_x)

            # loss
            if focal_obj:
                loss = focal_obj(output, batch_y)
            else:
                loss = F.nll_loss(output, batch_y)
            acc1 = accuracy(output, batch_y, topk=(1,))
            losses.update(loss.item(), batch_x.size(0))
            top1.update(acc1[0], batch_x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('===> Val time: {batch_time.avg:.3f}'.format(batch_time=batch_time))
    return top1.avg


# generating score file.
def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()

    fname_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    utt2scores = defaultdict(list)
    meta_dict = {}
    for i, (batch_x, batch_y, batch_meta) in enumerate(data_loader):
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 0] - batch_out[:, 1]
                       ).data.cpu().numpy().ravel()

        # score = batch_out[:, 0]  # use log-probability of the bonafide class for scoring
        for i in range(batch_size):
            utt_id = batch_meta.file_name[i].split('-')[0]
            tmp_meta = ASVFile(speaker_id=batch_meta.speaker_id[i],
                               file_name=utt_id,
                               path=batch_meta.path[i],
                               sys_id=batch_meta.sys_id[i],
                               key=batch_meta.key[i])
            utt2scores[utt_id].append(batch_score[i].item())
            meta_dict[utt_id] = tmp_meta

    # print(type(dataset), type(dataset.sysid_dict_inv), dataset.sysid_dict_inv)
    for utt_id, value in utt2scores.items():
        avg_score = reduce(lambda x, y: x + y, value) / len(value)

        # add outputs
        fname_list.append(utt_id)
        key_list.append('bonafide' if meta_dict[utt_id].key == 1 else 'spoof')
        # print(type(meta_dict[utt_id].sys_id), meta_dict[utt_id].sys_id)
        # print(type(dataset.sysid_dict_inv[meta_dict[utt_id].sys_id]))
        sys_id_list.append(dataset.sysid_dict_inv[meta_dict[utt_id].sys_id.item()])
        # print(type(avg_score))
        score_list.append(avg_score)

    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            fh.write('{} {} {} {}\n'.format(f, s, k, cm))
    print('Result saved to {}'.format(save_path))


def train_epoch(train_loader, model, epoch, device, optim, log_interval, focal_obj=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    # optim = torch.optim.Adam(model.parameters(), lr=lr)
    # optim = ScheduledOptim(
    #         torch.optim.Adam(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True), 1000)
    # weight = torch.FloatTensor([1.0, 9.0]).to(device)
    # criterion = nn.NLLLoss(weight=weight)

    end = time.time()
    tmp_time = 0
    for i, (batch_x, batch_y, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #  get batch size
        batch_size = batch_x.size(0)

        # create variables
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # compute output
        batch_out = model(batch_x)

        # loss
        if focal_obj: batch_loss = focal_obj(batch_out, batch_y)
        else: batch_loss = F.nll_loss(batch_out, batch_y)

        # measure accuracy and record loss
        acc1 = accuracy(batch_out, batch_y, topk=(1, ))
        losses.update(batch_loss.item(), batch_size)
        top1.update(acc1[0], batch_size)

        #  compute gradient and record loss
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        lr = optim.update_learning_rate()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        tmp_time += batch_time.val

        # print info every log_interval batches.
        if i % log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                  'Data {data_time_val:.3f} ({data_time_avg:.3f})\t'
                  'LR {lr:.6f}\t'
                  'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                   epoch, i, len(train_loader),
                   batch_time_val=batch_time.val, batch_time_avg=batch_time.avg,
                   data_time_val=data_time.val, data_time_avg=data_time.avg,
                   lr=lr,
                   loss_val=losses.val, loss_avg=losses.avg,
                   top1_val=top1.val.item(), top1_avg=top1.avg.item()))

    return losses.avg, top1.avg, tmp_time


def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=512, win_length=512, hop_length=256)
    a = np.abs(s)**2
    #melspect = librosa.feature.melspectrogram(S=a)
    feat = librosa.power_to_db(a)
    return feat


def compute_mfcc_feats(x):
    mfcc = librosa.feature.mfcc(x, sr=16000, n_mfcc=24)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)
    return feats


def get_wav(x):
    return np.reshape(x, (1, -1))

def frame_div(x):
    s_len = len(x)
    fs = 16000
    N = int(0.015 * fs)
    interval = int(0.005 * fs)
    N_frame = (s_len - N) // interval + 1

    # hamming window
    # n_lin = np.linspace(0, N, num=N, endpoint=False)
    # win = 0.54-0.46*np.cos(2*math.pi*n_lin/(N - 1))

    input = np.zeros([N_frame, N])
    for i in range(N_frame):
        input[i, :] = np.asarray(x[interval*i:(interval*i+N)])
    return input


class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta += 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UCLANESL ASVSpoof2019  model')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--model_name', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='spect')
    parser.add_argument('--is_test', action='store_true', default=False)
    parser.add_argument('--dataset_sz', type=int, default=0)
    # parser.add_argument('--eer_criteria', action='store_true', default=False)
    parser.add_argument('--multi_class', action='store_true', default=False)
    parser.add_argument('--eer_criteria', action='store_true', default=False)
    parser.add_argument('--focal_obj', action='store_true', default=False)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    track = args.track
    pretrained = args.model_name
    eer_criteria = args.eer_criteria
    focal_obj = args.focal_obj

    assert args.features in ['mfcc', 'spect', 'wave-ori', 'spect-ori', 'cqcc',
                             'wave', '2d_wave'], 'Not supported feature'
    model_tag = 'model_{}_{}_{}_{}_{}_{}'.format(
        track, args.features, args.num_epochs, args.batch_size, args.lr, args.dataset_sz)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    # model_save_path
    model_save_path = os.path.join('models', model_tag)
    assert track in ['logical', 'physical'], 'Invalid track given'
    is_logical = (track == 'logical')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
        print("create model directory successfully!\n")

    if args.features == 'mfcc':
        feature_fn = compute_mfcc_feats
        model_cls = MFCCModel
    elif args.features == 'spect':
        feature_fn = get_log_spectrum
        model_cls = SEResNet
    elif args.features == 'spect-ori':
        feature_fn = get_log_spectrum
        model_cls = SpectrogramModel
    elif args.features == 'cqcc':
        feature_fn = None  # cqcc feature is extracted in Matlab script
        model_cls = CQCCModel
    elif args.features == 'wave':
        # feature_fn = get_wav
        # model_cls = Sincnetmodelv1
        feature_fn = frame_div
        model_cls = Sincnetmodelv2
        # feature_fn = CLDNN
    elif args.features == 'wave-ori':
        feature_fn = get_wav
        model_cls = sincnet_ori
    elif args.features == '2d_wave':
        feature_fn = frame_div
        model_cls = CLDNN

    print("the input analysis object is '{}' and the corresponding model "
          "selected is '{}'\n".format(args.features, model_cls))

    # preprocess and feature extraction
    transforms = transforms.Compose([
        lambda x: librosa.util.normalize(x),
        lambda x: feature_fn(x),
        lambda x: Tensor(x)
    ])

    # detect and select device('cuda' or 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load eval(or dev) dataset
    eval_set = data_utils.ASVDataset(is_train=False, is_logical=is_logical,
                                    transform=transforms, feature_name=args.features,
                                     is_test=args.is_test, dataset_sz=args.dataset_sz, multi_class=args.multi_class)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=True)

    model = model_cls().to(device)
    print(args)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n===> Model total parameter: {}\n'.format(model_params))
    optim = ScheduledOptim(
            torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.999), eps=1e-09, weight_decay=1e-4, amsgrad=True), 1000)

    start_epoch = 0
    best_eer = 10000
    best_acc1 = 0
    t = 0
    if pretrained:
        mpretrained = os.path.join(model_save_path, pretrained)
        if os.path.isfile(mpretrained):
            print("===> loading checkpoint at '{}'".format(model_save_path))
            checkpoint = torch.load(mpretrained)
            start_epoch = checkpoint['epoch']
            t = checkpoint['train_time']
            if eer_criteria:
                best_eer = checkpoint['best_eer']
            else:
                best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("===> loaded checkpoint '{}' (epoch {})"
                  .format(pretrained, checkpoint['epoch']))
        else:
            print("===> no checkpoint found at '{}'".format(model_save_path))


    if args.eval:
        res_path = './res'
        if not os.path.exists(res_path):
            os.mkdir(res_path)
            print("create result directory successfully!\n")

        assert args.eval_output is not None, 'You must provide an output path'
        assert pretrained is not None, 'You must provide model checkpoint'
        eval_type = 'eval' if args.is_test else 'dev'
        t1 = time.time()
        produce_evaluation_file(eval_set, model, device,
                                '{}/{}_{}_{}'.format(res_path, model_tag, eval_type, args.eval_output))
        t2 = time.time()
        print('Testing process consumes {} seconds'.format(t2-t1))
        sys.exit(0)

    # load train set
    train_set = data_utils.ASVDataset(is_train=True, is_logical=is_logical, transform=transforms,
                                      feature_name=args.features, dataset_sz=args.dataset_sz)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True)

    # training params settings
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_epoch = 0
    early_stopping, max_patience = 0, 6 #for early stopping
    print('===> Training starting...')
    print('--------------------------------------------------------------------------')

    fig_path = './fig'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    params_path = os.path.join(fig_path, model_tag)
    if not os.path.exists(params_path):
        os.mkdir(params_path)
    val_acc = np.zeros((100, 2))
    loss_curve = np.zeros((100, 2))
    for epoch in range(start_epoch, num_epochs):
        val_acc[epoch, 0] = epoch + 1
        loss_curve[epoch, 0] = epoch + 1
        running_loss, train_accuracy, total_time = train_epoch(train_loader, model,
                                                   epoch, device, optim, 1)
        t += total_time
        print('=> Validation starting...')

        acc1 = validate(eval_loader, model, device, focal_obj)

        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', acc1, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        val_acc[epoch, 1] = acc1
        loss_curve[epoch, 1] = running_loss
        print('epoch: {} - running loss: {} - train_accuracy: {:.2f} - acc1: {:.2f}\n'
		.format(epoch, running_loss, train_accuracy.item(), acc1.item()))

        # remember best acc@1/eer and save checkpoint
        if eer_criteria:
            is_best = acc1 < best_eer
            best_eer = min(acc1, best_eer)
        else:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

        # adjust learning rate + early stopping
        if is_best:
            early_stopping = 0
            best_epoch = epoch + 1
        else:
            early_stopping += 1
            if epoch - best_epoch > 2:
                optim.increase_delta()
                best_epoch = epoch + 1
        if early_stopping == max_patience:
            print('--------------------------------------------------------------------------')
            print('=>Early stop!\n')
            print('Training process consumes {} seconds'.format(t))
            break

        # save model
        if not is_best:
            continue
        model_dict = {'epoch': epoch, 'state_dict': model.state_dict(),
                      'optimizer': optim.state_dict(), 'train_time': t}
        if eer_criteria: model_dict['best_eer'] = best_eer
        else: model_dict['best_acc1'] = best_acc1
        print('=>Model saving...\n')
        torch.save(model_dict, os.path.join(model_save_path,
                                            'epoch_{}.pth'.format(epoch)))
        if is_best:
            print("===> save to checkpoint at {}\n".format(model_save_path + '/' + 'final.pth'))
            shutil.copyfile(os.path.join(model_save_path,
                                         'epoch_{}.pth'.format(epoch)),
                                         model_save_path + '/' + 'final.pth')

        if epoch == num_epochs - 1:
            print('--------------------------------------------------------------------------')
            print('training complete!')
            print('Training process consumes {} seconds'.format(t))
            break
    val_txt = os.path.join(params_path, 'val.txt')
    loss_txt = os.path.join(params_path, 'loss.txt')
    np.savetxt(val_txt, val_acc, fmt='%.8f')
    np.savetxt(loss_txt, loss_curve, fmt='%.8f')
    writer.close()
