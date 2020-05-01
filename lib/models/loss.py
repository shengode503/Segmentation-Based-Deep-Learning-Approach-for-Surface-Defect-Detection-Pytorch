import torch.nn as nn
import torch


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        loss = self.criterion(pred, target)

        return loss

def get_loss():
    # return BCELoss() if train_type == 'SegNet' else CrossEntropyLoss()
    return BCELoss()


if __name__ == '__main__':
    pass




