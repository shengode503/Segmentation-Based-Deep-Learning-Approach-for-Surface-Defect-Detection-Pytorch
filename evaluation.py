from __future__ import print_function, absolute_import

# Import
from lib.data_helper.data_loader import Data_loader
from lib.models import model
from lib.utils.utils import *
from lib.config import cfg
import pickle

# Select Device
device = select_device('cpu')  # 'cpu', '0', '1'  # str(device)

# model
m = model.SegDecNet(cfg, device, train_type='DecNet', load_segnet=False)
state = torch.load(os.path.join(cfg.CHECKPOINT_PATH, 'DecNet_model_best.pth'), map_location=device)
m.load_state_dict(state['state_dict'])

m = m.to(device)
m.eval()

# DataLoader
train_loader = torch.utils.data.DataLoader(
    Data_loader(cfg, is_train=False),
    batch_size=3, shuffle=False,
    num_workers=0, pin_memory=False)


with torch.no_grad():

    result = {
        'SegNet': {
            'pred': [],
            'tar': []},
        'DecNet': {
            'pred': [],
            'tar': []}
              }

    for i, (image, label_pixel, label, info) in enumerate(train_loader):

        print('# batch_idx -----> [{0}]'.format(i))

        batch_size = len(image)

        # choose device
        input_img = image.to(device)
        label_pixel = label_pixel.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # prediction
        seg_out, c_out = m(input_img)

        # -------------------------------------------------------

        # sigmoid
        seg_out = torch.sigmoid(seg_out)
        c_out = torch.sigmoid(c_out)

        # to Numpy
        seg_out = seg_out.detach().to('cpu').numpy()
        c_out = c_out.detach().to('cpu').numpy()

        label = label.detach().to('cpu').numpy()
        label_pixel = label_pixel.detach().to('cpu').numpy()

        # > 0.5 == 1, < 0.5 == 0
        seg_out = np.int64(np.where(seg_out > 0.5, 1, 0))
        c_out = np.int64(np.where(c_out > 0.5, 1, 0))

        result['SegNet']['pred'].append(seg_out)
        result['DecNet']['pred'].append(c_out)

        result['SegNet']['tar'].append(label_pixel)
        result['DecNet']['tar'].append(label)

    # -------------------------------------------------------

    # # Save
    # with open('Result.pkl', 'wb') as f:
    #     pickle.dump(result, f)

    # for DecNet
    pred = result['DecNet']['pred']
    tar = result['DecNet']['tar']

    Pred, Tar = [], []
    for i in range(len(pred)):
        p = pred[i]
        t = tar[i]
        for pp, tt in zip(p, t):
            Pred += pp.tolist()
            Tar += tt.tolist()
    pred = np.asarray(Pred)
    tar = np.asarray(Tar)

    tp, fn, fp, tn = confusion_matrix(tar, pred)['cfm']
    Prec, recall, f1 = compute_mm(tp, fn, fp, tn)

    print("# -------> DecNet's Result:")
    print('Precision:', Prec)
    print('Recall:', recall)
    print('F1_Score:', f1)

    # -------------------------------------------

    # for SegNet
    pred = result['SegNet']['pred']
    tar = result['SegNet']['tar']

    Pred = np.zeros([1, pred[0].shape[1], pred[0].shape[2]])
    Tar = np.zeros_like(Pred)
    for i in range(len(pred)):
        Pred = np.vstack((Pred, pred[i]))
        Tar = np.vstack((Tar, tar[i]))
    pred = Pred[1, :, :]
    tar = Tar[1, :, :]

    tp, fn, fp, tn = confusion_matrix(tar, pred)['cfm']
    Prec, recall, f1 = compute_mm(tp, fn, fp, tn)

    print("# -------> SegNet's Result:")
    print('tp: {0}, fn: {1}, fp: {2}, tn: {3}'.format(tp, fn, fp, tn))
    print('Precision:', Prec)
    print('Recall:', recall)
    print('F1_Score:', f1)
