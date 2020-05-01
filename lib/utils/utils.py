import os
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def metric(out, tar, thre=0.5, train_type='SegNet'):

    batch_size = len(out)

    # Sigmoid
    pred = torch.sigmoid(out)

    # to Numpy
    pred = pred.detach().to('cpu').numpy()
    target = tar.detach().to('cpu').numpy()

    # to int
    pred = np.int64(np.where(pred > thre, 1, 0))
    target = np.int64(target)

    if train_type == 'SegNet':
        pred = pred.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)

    # confusion_matrix
    cfm = confusion_matrix(target, pred)

    return cfm['cfm']


def confusion_matrix(tar, pred):
    # tar: Target [Batch_size, features]
    # pred: Prediction [Batch_size, features]
    assert len(pred) == len(tar), 'pred.shape & target.shape must the same.'

    # True Positive
    tp = (pred == 1) * 1 + (tar == 1) * 1

    # False Negative
    fn = (tar == 1) * 1 + (pred == 0) * 1

    # False Positive
    fp = (tar == 0) * 1 + (pred == 1) * 1

    # True Negative
    tn = (pred == 0) * 1 + (tar == 0) * 1

    tp, fn, fp, tn = [np.sum(np.sum(m[m == 2]) // 2)
                      for m in [tp, fn, fp, tn]]

    return {'cfm': [tp, fn, fp, tn]}


def compute_mm(tp, fn, fp, tn):

    # prescision
    prec = tp / (tp + fp)

    # recall
    recall = tp / (tp + fn)

    # F1-Score
    f1 = (2 * prec * recall) / (prec + recall)

    # # Accuracy
    # acc = (tp + tn) / cnt

    return prec, recall, f1


def select_device(device='', apex=False):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        cuda_str = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                cuda_str = ' ' * len(cuda_str)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (cuda_str, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def save_checkpoint(states, is_best, output_dir, filename='', snapshot=None):
    if snapshot and states['epoch'] % snapshot == 0:
        torch.save(states, os.path.join(output_dir, filename + '_checkpoint_{0}.pth'.format(states['epoch'])))

    if is_best and 'state_dict' in states:
        torch.save(states, os.path.join(output_dir, filename + '_model_best.pth'))

