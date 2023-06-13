import time
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import uniform, normal
import pandas as pd
from utils import get_graph, combine_ZA  # *

cuda = False # True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def eval(args, epoch, dataloader, toTrModels, nonTrModels, centers):
    # (1) Per-class accuracy
    bi_cm = np.zeros((17, 2))  # Binary classify
    open_cm = np.zeros((17, 11))  # 10 + 1 classify
    sr_cm = np.zeros((17, 17))  # 50 classify
    # Sample average accuracy

    # prepare target data
    att_cents = centers['att_cents']
    te_tgt_iter = enumerate(dataloader)
    n_batches = len(dataloader)

    for itern in range(n_batches):
        (feats_tgt, lbls_tgt, atts_tgt, clu_tgt) = te_tgt_iter.__next__()[1]

        feats_tgt_var = Variable(feats_tgt.type(FloatTensor))
        # lbls_tgt_var = Variable(lbls_tgt.type(LongTensor))
        # atts_tgt_var = Variable(atts_tgt.type(FloatTensor))
        # clu_tgt_var = Variable(clu_tgt.type(LongTensor))

        # for tgt
        t_z = toTrModels['GenZ'](feats_tgt_var)
        if args.GenA_type == 'GCN':
            t_vertices = feats_tgt_var
            t_adj = get_graph(a=t_vertices, b=t_vertices, dist='euclidean', alpha=0.2, graph_type='adjacency')
            t_att_pred = toTrModels['GenA'](t_z, t_adj)
        else:
            t_att_pred = toTrModels['GenA'](t_z)
            # ''' Attributes propagation to refine the attributes'''
            propagator = get_graph(t_z, t_z, dist='euclidean', alpha=0.2)
            propagator = F.normalize(propagator, p=1, dim=1)
            t_att_pred = torch.mm(propagator, t_att_pred)

        if args.combine_za:
            t_f = combine_ZA(t_z, t_att_pred)
        else:
            t_f = t_z

        # (1) Binary Classify
        t_su_prob, t_su_pred = toTrModels['ClfSU'](t_f)  # binary classify
        t_pred_shr_inds = t_su_pred == 0
        t_pred_unk_inds = t_su_pred == 1
        # (2) OpenSet Classify
        t_prob, t_y_pred, t_mul_dis = toTrModels['Clf'](t_f)  # (NxC)(Nx1)(Nx1), 10 + 1 classify
        # (3) Feature Generation Classify
        if args.binary:
            threshold = 0.5
            t_att_pred[t_att_pred >= threshold] = 1
            t_att_pred[t_att_pred < threshold] = 0
        else:
            pass
        yt_pred_att = Variable(FloatTensor(t_att_pred.size(0)).fill_(-1), requires_grad=False)
        yt_pred_att_shr = nonTrModels['ProtClf'](t_att_pred.detach(), att_cents[:10, :], dist='cosine', T=0.1)[1]
        yt_pred_att_unk = nonTrModels['ProtClf'](t_att_pred.detach(), att_cents[10:, :], dist='cosine', T=0.1)[1] + 10
        yt_pred_att_shr = yt_pred_att_shr.type(FloatTensor)
        yt_pred_att_unk = yt_pred_att_unk.type(FloatTensor)
        yt_pred_att[t_pred_shr_inds] = yt_pred_att_shr[t_pred_shr_inds]
        yt_pred_att[t_pred_unk_inds] = yt_pred_att_unk[t_pred_unk_inds]

        # Update confusion matrix
        for i in range(len(lbls_tgt)):
            yi_true = lbls_tgt[i].item() # gt label: [0 - 16]
            yi_bi_pred = t_su_pred[i].item()  # binary shr/unk: 0 or 1
            bi_cm[yi_true, yi_bi_pred] += 1.0

            yi_open_pred = t_y_pred[i].item() # 40 + 1
            open_cm[yi_true, yi_open_pred] += 1.0

            yi_att_pred = int(yt_pred_att[i].item()) # 40 + 10
            sr_cm[yi_true, yi_att_pred] += 1.0

    # (1) Per-class accuracy
    bi_cm = bi_cm.astype(np.float) / np.sum(bi_cm, axis=1, keepdims=True) # Binary classify
    shr_Bi_Acc = np.sum(bi_cm[:10, 0]) / 10
    unk_Bi_Acc = np.sum(bi_cm[10:, 1]) / 7

    open_cm = open_cm.astype(np.float) / np.sum(open_cm, axis=1, keepdims=True)  # Binary classify
    shr_Open_Acc = np.sum(open_cm[:10, :10].diagonal()) / 10
    unk_Open_Acc = np.sum(open_cm[10:, 10]) / 7



    sr_cm = sr_cm.astype(np.float) / np.sum(sr_cm, axis=1, keepdims=True)  # Binary classify
    shr_SR_Acc = np.sum(sr_cm[:10, :10].diagonal()) / 10
    unk_SR_Acc = np.sum(sr_cm[10:, 10:].diagonal()) / 7

    print("Ep {epoch:d}(s/u): {shr_Bi_Acc:.2f}/{unk_Bi_Acc:.2f} Open: " \
             "OS*={shr_Open_Acc:.2f}/OS^={unk_Open_Acc:.2f}/OS={Open_Acc:.2f}" \
             " SR: S={shr_SR_Acc:.2f}/U={unk_SR_Acc:.2f}/H={SR_Acc:.2f}".format(
                epoch=epoch, shr_Bi_Acc=shr_Bi_Acc*100, unk_Bi_Acc=unk_Bi_Acc*100,
                shr_Open_Acc=shr_Open_Acc*100, unk_Open_Acc=unk_Open_Acc*100, Open_Acc= (shr_Open_Acc*10+unk_Open_Acc)/11 * 100,
                shr_SR_Acc=shr_SR_Acc*100, unk_SR_Acc=unk_SR_Acc*100, SR_Acc=(2*shr_SR_Acc*unk_SR_Acc)/(shr_SR_Acc+unk_SR_Acc)*100)
    )

    # Log results
    result = "Ep {epoch:d}(s/u): {shr_Bi_Acc:.2f}/{unk_Bi_Acc:.2f} Open: " \
             "OS*={shr_Open_Acc:.2f}/OS^={unk_Open_Acc:.2f}/OS={Open_Acc:.2f}" \
             " SR: S={shr_SR_Acc:.2f}/U={unk_SR_Acc:.2f}/H={SR_Acc:.2f}".format(
                epoch=epoch, shr_Bi_Acc=shr_Bi_Acc*100, unk_Bi_Acc=unk_Bi_Acc*100,
                shr_Open_Acc=shr_Open_Acc*100, unk_Open_Acc=unk_Open_Acc*100, Open_Acc= (shr_Open_Acc*10+unk_Open_Acc)/11 * 100,
                shr_SR_Acc=shr_SR_Acc*100, unk_SR_Acc=unk_SR_Acc*100, SR_Acc=(2*shr_SR_Acc*unk_SR_Acc)/(shr_SR_Acc+unk_SR_Acc)*100)

    results_path = './results/' + args.att_type + '/step3/N2AwA/' + args.src+'2'+args.tgt
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    with open(results_path+'/eval_record.txt', 'a') as f:
        f.write(result)
        f.write('\n')




