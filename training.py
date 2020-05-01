from __future__ import print_function, absolute_import

# Import
from lib.data_helper.data_loader import Data_loader
from lib.train.trainer import train, validate
import torch.backends.cudnn as cudnn
from lib.models import model, loss
from lib.utils.utils import *
from lib.config import cfg
import time

# Select Device
cudnn.benchmark = cfg.CUDNN.BENCHMARK
device = select_device(str(cfg.DEVICE))  # 'cpu', '0', '1'

# model
model = model.SegDecNet(cfg, device, train_type=cfg.TRAIN.TRAIN_MODEL, load_segnet=True)

# loss
criterion = loss.get_loss()

# Choose the Optimizer
if cfg.TRAIN.OPTIMIZER == 'rms':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
elif cfg.TRAIN.OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

elif cfg.TRAIN.OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR)

else:
    print('Unknown solver: {}'.format(cfg.TRAIN.OPTIMIZER))
    assert False

# DataLoader
train_loader = torch.utils.data.DataLoader(
    Data_loader(cfg, is_train=True),
    batch_size=cfg.TRAIN.TRAIN_BATCH, shuffle=True,
    num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)

val_loader = torch.utils.data.DataLoader(
    Data_loader(cfg, is_train=False),
    batch_size=cfg.TRAIN.TRAIN_BATCH, shuffle=False,
    num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)


# Train & Val
idx = []
best_f1 = 0
lr = cfg.TRAIN.LR
for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):

    # Adjust Learning Rate.
    lr = adjust_learning_rate(optimizer, epoch, lr, cfg.TRAIN.SCHEDULE, cfg.TRAIN.LR_FACTOR)
    print('\nEpoch: %d | Learn Rate: %.8f' % (epoch + 1, lr))

    # --------------------------------------------------------------------------------------

    # Training
    end = time.time()
    train_loss = train(cfg, train_loader, model, criterion, optimizer, epoch, device)
    train_time = time.time() - end

    # Evaluate
    end = time.time()
    valid_loss, prec, recall, f1 = validate(cfg, val_loader, model, criterion, device)
    val_time = time.time() - end

    # remember best mm and save checkpoint
    is_best = f1 > best_f1
    best_f1 = max(f1, best_f1)

    states = {'epoch': epoch + 1,
              'model': cfg.TRAIN.TRAIN_MODEL,
              'state_dict': model.state_dict(),
              'perf': f1,
              'optimizer': optimizer.state_dict()}

    save_checkpoint(states, is_best,
                    snapshot=cfg.TRAIN.snapshot,
                    filename=cfg.TRAIN.TRAIN_MODEL,
                    output_dir=cfg.CHECKPOINT_PATH)
