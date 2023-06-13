from torch.nn.functional import one_hot
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable

cuda = False  #True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



def get_clu_centers(args, toTrModels, dataloaders, centers, label='init'):# label='init' | 'clf'
    print("z,a cents updated!")
    for key in toTrModels:
        toTrModels[key].eval()
    if label == 'init':  # Calculate based on init pseudo labels
        zt_clu_cents = torch.zeros((17, 512)).type(FloatTensor)
        at_clu_cents = torch.zeros((17, 85)).type(FloatTensor)
        n_clu_samples = torch.zeros((17)).type(FloatTensor)

        te_tgt_iter = enumerate(dataloaders['te_loader_tgt'])
        for i, (feats_tgt, lbls_tgt, atts_tgt, clu_tgt) in te_tgt_iter:
            feats_tgt_var = Variable(feats_tgt.type(FloatTensor))
            clu_tgt_var = Variable(clu_tgt.type(LongTensor))

            t_z = toTrModels['GenZ'](feats_tgt_var)
            if args.GenA_type == 'GCN':
                t_vertices = feats_tgt_var  # centers['xt_clu_cents'][clu_tgt_var] # Can use predicted labels, init labels maybe easy to train
                t_adj = get_graph(a=t_vertices, b=t_vertices, dist='euclidean', alpha=0.2, graph_type='adjacency')
                t_att_pred = toTrModels['GenA'](t_z, t_adj)  # .detach())#.detach()
            else:
                t_att_pred = toTrModels['GenA'](t_z)
                # propagator = get_graph(feats_tgt_var, feats_tgt_var, dist='euclidean', alpha=0.2)
                propagator = get_graph(t_z, t_z, dist='euclidean', alpha=0.2)
                propagator = F.normalize(propagator, p=1, dim=1)
                t_att_pred = torch.mm(propagator, t_att_pred)

            for j in range(len(t_z)):
                clu_lbl = clu_tgt_var[j].type(LongTensor)
                n_clu_samples[clu_lbl] += 1
                zt_clu_cents[clu_lbl, :] += t_z[j, :]#.clone()
                at_clu_cents[clu_lbl, :] += t_att_pred[j, :]#.clone()

        zt_clu_cents = zt_clu_cents / n_clu_samples.unsqueeze(1).repeat(1, 512)
        at_clu_cents = at_clu_cents / n_clu_samples.unsqueeze(1).repeat(1, 85)
        return zt_clu_cents, at_clu_cents

    elif label == 'clf':
        zt_clu_cents = torch.zeros((17, 512)).type(FloatTensor)
        at_clu_cents = torch.zeros((17, 85)).type(FloatTensor)
        n_clu_samples = torch.zeros((17)).type(FloatTensor)

        te_tgt_iter = enumerate(dataloaders['te_loader_tgt'])
        for i, (feats_tgt, _, _, clu_tgt) in te_tgt_iter:
            feats_tgt_var = Variable(feats_tgt.type(FloatTensor).detach())
            clu_tgt_var = Variable(clu_tgt.type(LongTensor).detach())

            t_z = toTrModels['GenZ'](feats_tgt_var)
            if args.GenA_type == 'GCN':
                t_vertices = feats_tgt_var  # centers['xt_clu_cents'][clu_tgt_var] # Can use predicted labels, init labels maybe easy to train
                t_adj = get_graph(a=t_vertices, b=t_vertices, dist='euclidean', alpha=0.2, graph_type='adjacency')
                t_att_pred = toTrModels['GenA'](t_z, t_adj)  # .detach())#.detach()
            else:
                t_att_pred = toTrModels['GenA'](t_z)
                # ''' Attributes propagation to refine the attributes'''
                propagator = get_graph(t_z, t_z, dist='euclidean', alpha=0.2)
                propagator = F.normalize(propagator, p=1, dim=1)
                t_att_pred = torch.mm(propagator, t_att_pred)

            t_f = combine_ZA(t_z, t_att_pred)
            t_prob, t_y_pred, _ = toTrModels['Clf'](t_f)  # (NxC)(Nx1)(Nx1)
            t_su_prob, t_su_pred = toTrModels['ClfSU'](t_f)

            for j in range(len(t_z)):
                if t_su_pred[j] == 0: # predicted as share
                    clu_lbl = t_y_pred[j].type(LongTensor)
                    n_clu_samples[clu_lbl] += torch.max(t_prob[j])
                    zt_clu_cents[clu_lbl, :] += t_z[j, :]#.clone()
                    at_clu_cents[clu_lbl, :] += t_att_pred[j, :]#.clone()
                else: # t_su_pred == 1, unshared
                    clu_lbl = clu_tgt_var[j].type(LongTensor) # Consider classify again with z prototypes
                    n_clu_samples[clu_lbl] += 1
                    zt_clu_cents[clu_lbl, :] += t_z[j, :]#.clone()
                    at_clu_cents[clu_lbl, :] += t_att_pred[j, :]#.clone()

        for k in range(17):
            if n_clu_samples[k] == 0:
                print("Lost center: ", k)
                try:
                    zt_clu_cents[k,:] = centers['zt_clu_cents'][k,:].detach()
                    at_clu_cents[k,:] = centers['at_clu_cents'][k,:].detach()
                    n_clu_samples[k] += 1
                except Exception:
                    raise ValueError("zt/at_clu_cents does not exist.")
            else:
                pass

        zt_clu_cents = zt_clu_cents / n_clu_samples.unsqueeze(1).repeat(1, 512)
        at_clu_cents = at_clu_cents / n_clu_samples.unsqueeze(1).repeat(1, 85)

        alpha = 0.001
        old_zt_clu_cents = centers['zt_clu_cents'].detach()
        old_at_clu_cents = centers['at_clu_cents'].detach()
        new_zt_clu_cents = (1-alpha)*old_zt_clu_cents + alpha * zt_clu_cents
        new_at_clu_cents = (1-alpha)*old_at_clu_cents + alpha * at_clu_cents

        return new_zt_clu_cents, new_at_clu_cents


def get_graph(a, b, dist='euclidean', alpha=0.2, graph_type='propagator'): #propagator | ajacency
    weights = get_adjacency(a, b, dist=dist, alpha=alpha).float() # mask
    if graph_type == 'adjacency':
        adj = F.normalize(weights, p=1, dim=1)
        return adj
    elif graph_type == 'propagator':
        n = weights.shape[1]
        identity = torch.eye(n, dtype=weights.dtype).type(FloatTensor)
        isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
        # checknan(laplacian=isqrt_diag)
        S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
        # checknan(normalizedlaplacian=S)
        propagator = identity - alpha * S
        propagator = torch.inverse(propagator[None, ...])[0]
        # checknan(propagator=propagator)
        return propagator
    else:
        return None

def get_adjacency(a,b,dist='euclidean',alpha=0.2):
    dist_map = get_dist_map(a,b,dist=dist)
    mask = dist_map != 0
    rbf_scale = 1
    weights = torch.exp(- dist_map * rbf_scale / dist_map[mask].std())
    mask = torch.eye(weights.size(1)).type(FloatTensor)
    weights = weights * (1-mask) #~mask
    return weights

def get_dist_map(a, b, dist='euclidean'):
    if dist == 'abs':
        dist_map = torch.cdist(a, b, p=1)
    elif dist == 'euclidean':
        dist_map = torch.cdist(a, b, p=2)
    elif dist == 'cosine':
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        dist_map = 1 - torch.mm(a_norm, b_norm.transpose(0, 1))
    elif dist == 'cosine_sim':
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        dist_map = torch.mm(a_norm, b_norm.transpose(0, 1))
    else:
        raise Exception("Distance NOT defined!")
    return dist_map


def combine_ZA(z, a, method='cat'):
    if method == 'z':
        return z
    elif method == 'cat':
        z = F.normalize(z, dim=1, p=2)
        a = F.normalize(a, dim=1, p=2)
        return torch.cat([z, a], dim=1)
    # elif method == 'bmm':
    #     z = F.normalize(z, dim=1, p=2)
    #     a = F.normalize(a, dim=1, p=2)
    #     return torch.bmm(z.unsqueeze(2), a.unsqueeze(1)).view(-1, z.size(1) * a.size(1))
    else:
        raise Exception("Combine method not exists.")


def resume_pretrained_weights(args, models, step='step3'):
    for m in models:
        saved_weights = torch.load('./saved_weights/'+args.att_type+'/'+step+'/N2AwA/'+args.src+'2'+args.tgt+'_'+m+'.pth')
        models[m].load_state_dict(saved_weights)
    return models


def count_epoch_on_large_dataset(train_loader_source, train_loader_target):
    batch_number_t = len(train_loader_target)
    batch_number_s = len(train_loader_source)
    if batch_number_s > batch_number_t:
        return 'source', batch_number_s
    else:
        return 'target', batch_number_t


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    if args.lr_plan == 'step':
        exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
        lr = args.lr * (0.1 ** exp)
    elif args.lr_plan == 'dao':
        lr = args.lr / math.pow((1 + 10 * epoch / args.epochs), 0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_oneHot(y, n_classes):
    return one_hot(y, num_classes=n_classes).type(FloatTensor)

