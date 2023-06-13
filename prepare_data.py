import torch
import numpy as np
import pandas as pd
import torch.utils.data as data


class FeatureDataset(data.Dataset):
    """
    Args:
        data_dir (String): data file
        labels_dir (String): labels file
        att_dir (String): attributes file
        clu_dir (string): clu lbls file
    """

    def __init__(self, data_dir, labels_dir, att_dir, clu_dir):
        self.features = pd.read_csv(data_dir, header=None, index_col=None).values  # Pre-trained ResNet-50 features of data
        self.labels = pd.read_csv(labels_dir, header=None, index_col=None)[0].values
        self.attributes = pd.read_csv(att_dir).values
        self.clu_labels = pd.read_csv(clu_dir, header=None, index_col=None)[0].values  # Clustering results/labels based on the ResNet-50 features of the target domain data
        self.classes, self.counts = np.unique(self.labels, return_counts=True)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, attribute) where target is class_index of the target class.
        """
        feats = self.features[index, :]
        lbls = self.labels[index]
        atts = self.attributes[index, :]
        clu_lbls = self.clu_labels[index]

        return feats, lbls, atts, clu_lbls

    def __len__(self):
        return len(self.labels)


def generate_dataloader(args):
    # Data loading code
    # (1) Source domain
    src_feat_dir = args.data_path_source + '/features/' + args.src + '10_feats.csv'
    src_lbl_dir = args.data_path_source + '/features/' + args.src + '10_labels.csv'
    if args.att_type == 'binary':
        src_att_dir = args.data_path_source + '/attributes/' + args.src + '10_att_bi.csv'
    elif args.att_type == 'continuous':
        raise Exception("No continuous attributes!")
    else:
        raise ValueError("The attributes type is not defined.")
    src_clu_dir = src_lbl_dir  # Source domain directly use ground-truth labels as clustering results

    source_train_dataset = FeatureDataset(data_dir=src_feat_dir, labels_dir=src_lbl_dir, att_dir=src_att_dir, clu_dir=src_clu_dir)

    # (2) Target domain - training stage (gt 'lbl' and 'att' not used in training)
    tgt_feat_dir = args.data_path_target + '/features/' + args.tgt + '_feats.csv'
    tgt_lbl_dir = args.data_path_target + '/features/' + args.tgt + '_labels.csv'  # GT not used in training
    if args.att_type == 'binary':
        tgt_att_dir = args.data_path_target + '/attributes/' + args.tgt + '_att_bi.csv'  # GT not used in training
    elif args.att_type == 'continuous':
        raise Exception("No continuous attributes!")
    else:
        raise ValueError("The attributes type is not defined.")

    if args.init_prot_type == 'center':
        raise Exception("No center initialized pseudo!")
    elif args.init_prot_type == 'sample':
        tgt_clu_dir = args.data_path_target + '/pseudo/' + args.src+'2'+args.tgt + '_sample_pseudo.csv'
    else:
        raise ValueError("init_prot_type does not exist.")

    target_train_dataset = FeatureDataset(data_dir=tgt_feat_dir, labels_dir=tgt_lbl_dir, att_dir=tgt_att_dir, clu_dir=tgt_clu_dir)

    # (3) Target domain - test stage (gt 'lbl' and 'att' not used in training.)
    tgt_feat_dir = args.data_path_target + '/features/' + args.tgt + '_feats.csv'
    tgt_lbl_dir = args.data_path_target + '/features/' + args.tgt + '_labels.csv'  # GT not used in training
    if args.att_type == 'binary':
        tgt_att_dir = args.data_path_target + '/attributes/' + args.tgt + '_att_bi.csv'  # GT not used in training
    elif args.att_type == 'continuous':
        raise Exception("No continuous attributes!")
    else:
        raise ValueError("The attributes type is not defined.")

    if args.init_prot_type == 'center':
        raise Exception("No center initialized pseudo!")
    elif args.init_prot_type == 'sample':
        tgt_clu_dir = args.data_path_target + '/pseudo/' + args.src+'2'+args.tgt + '_sample_pseudo.csv'
    else:
        raise ValueError("init_prot_type does not exist.")

    target_test_dataset = FeatureDataset(data_dir=tgt_feat_dir, labels_dir=tgt_lbl_dir, att_dir=tgt_att_dir, clu_dir=tgt_clu_dir)

    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    return source_train_loader, target_train_loader, target_test_loader
