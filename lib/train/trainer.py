from __future__ import absolute_import

import time
from ..utils.utils import *
from ..utils.progress.progress.bar import Bar  # https://github.com/verigak/progress


def train(cfg, train_loader, model, criterion, optimizer, epoch, device):

    # Batch data container.
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    tpes = AverageMeter()
    fnes = AverageMeter()
    fpes = AverageMeter()
    tnes = AverageMeter()

    # switch to train mode.
    model.train()  # Train mode.
    model.to(device)

    # visualize tool
    bar = Bar('Train', max=len(train_loader))

    end = time.time()
    for i, (input_img, label_pixel, label, info) in enumerate(train_loader):

        batch_size = len(input_img)

        # measure data loading time
        data_time.update(time.time() - end)

        # choose device
        input_img = input_img.to(device)
        label_pixel = label_pixel.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # prediction
        seg_out, c_out = model(input_img)

        ' Loss '
        loss = criterion(seg_out if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else c_out,
                         label_pixel if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else label)
        losses.update(loss.item(), batch_size)  # record loss

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy  (seg_out, c_out,  label_pixel, label)
        tp, fn, fp, tn = metric(seg_out if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else c_out,
                                label_pixel if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else label,
                                train_type=cfg.TRAIN.TRAIN_MODEL)
        tpes.update(tp)
        fnes.update(fn)
        fpes.update(fp)
        tnes.update(tn)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        msg = 'Epoch: [{0}][{1}/{2}]  | Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | ' \
              'Speed: {speed:.1f} samples/s | Data: {data_time.val:.3f}s ({data_time.avg:.3f}s) | ' \
              'Loss: {loss.val:.5f} ({loss.avg:.5f})'.format(
               epoch, i, len(train_loader), batch_time=batch_time, speed=batch_size / batch_time.val,
               data_time=data_time, loss=losses)
        bar.suffix = msg
        bar.next()

    bar.finish()

    # Compute metric
    prec, recall, f1 = compute_mm(tpes.sum, fnes.sum, fpes.sum, tnes.sum)
    msg = 'Training_Metic || ' \
          'Precision-TP: {in0:.3f} |' \
          'Recall: {in1:.3f} |' \
          'F1-Score: {in2:.3f} |' \
          .format(in0=prec, in1=recall, in2=f1)
    print(msg)

    return losses.avg


def validate(cfg, val_loader, model, criterion, device):

    # Batch data container.
    batch_time = AverageMeter()
    losses = AverageMeter()

    tpes = AverageMeter()
    fnes = AverageMeter()
    fpes = AverageMeter()
    tnes = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.to(device)

    # visualize tool
    bar = Bar('Validate', max=len(val_loader))

    end = time.time()
    with torch.no_grad():
        for i, (input_img, label_pixel, label, info) in enumerate(val_loader):

            batch_size = len(input_img)

            # choose device
            input_img = input_img.to(device)
            label_pixel = label_pixel.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # prediction
            seg_out, c_out = model(input_img)

            ' Loss '
            loss = criterion(seg_out if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else c_out,
                             label_pixel if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else label)
            losses.update(loss.item(), batch_size)  # record loss

            # measure accuracy
            tp, fn, fp, tn = metric(seg_out if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else c_out,
                                    label_pixel if cfg.TRAIN.TRAIN_MODEL == 'SegNet' else label,
                                    train_type=cfg.TRAIN.TRAIN_MODEL)
            tpes.update(tp)
            fnes.update(fn)
            fpes.update(fp)
            tnes.update(tn)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            msg = 'Test: [{0}/{1}] | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | ' \
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})' \
                  .format(i, len(val_loader), batch_time=batch_time, loss=losses)

            bar.suffix = msg
            bar.next()

        bar.finish()

        # Compute
        prec, recall, f1 = compute_mm(tpes.sum, fnes.sum, fpes.sum, tnes.sum)
        msg = 'Valid_Metic || ' \
              'Precision-TP: {in0:.3f} |' \
              'Recall: {in1:.3f} |' \
              'F1-Score: {in2:.3f} |' \
            .format(in0=prec, in1=recall, in2=f1)
        print(msg)

        return losses.avg, prec, recall, f1

