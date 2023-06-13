import argparse
import torch

def opts():
    parser = argparse.ArgumentParser(description='Semantic Recovery for OpenSet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # datasets
    parser.add_argument('--data_path_source', type=str, default='./data/N2AwA/', help='root of source training set')
    parser.add_argument('--data_path_target', type=str, default='./data/N2AwA/', help='root of target training set')
    parser.add_argument('--src', type=str, default='3D2', help='source training set')
    parser.add_argument('--tgt', type=str, default='AwA2', help='target training set')
    parser.add_argument('--src_nc', type=int, default=10, help='src class number')
    parser.add_argument('--tgt_nc', type=int, default=17, help='tar class number')
    parser.add_argument('--shr_nc', type=int, default=10, help='src class number')
    parser.add_argument('--unk_nc', type=int, default=7, help='tar class number')

    # data specification
    parser.add_argument('--init_prot_type', type=str, default='sample', help='Type of init prototype classifcation, [center | sample]')
    parser.add_argument('--binary', type=bool, default=True, help='Binary the predicted or reconstructed attributes or not')
    parser.add_argument('--att_type', type=str, default='binary', help='binary or continuous attributes')
    parser.add_argument('--src_soft_select', action='store_true', help='whether to softly select source instances')
    parser.add_argument('--src_hard_select', action='store_true', help='whether to hardly select source instances')
    parser.add_argument('--src_mix_weight', action='store_true', help='whether to mix 1 and soft weight')
    parser.add_argument('--tao_param', type=float, default=0.5, help='threshold parameter of cosine similarity')
    # general optimization options
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--no_da', action='store_true', help='whether to not use data augmentation')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_plan', type=str, default='dao', help='learning rate decay plan of step or dao')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120], help='decrease learning rate at these epochs for step decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_true', help='whether to use nesterov SGD')
    parser.add_argument('--eps', type=float, default=1e-6, help='a small value to prevent underflow')
    # specific optimization options
    parser.add_argument('--ao', action='store_true', help='whether to use alternative optimization')
    parser.add_argument('--cluster_method', type=str, default='kmeans', help='clustering method of kmeans or spherical_kmeans or kernel_kmeans to choose')
    parser.add_argument('--cluster_iter', type=int, default=5, help='number of iterations of K-means')
    parser.add_argument('--cluster_kernel', type=str, default='rbf', help='kernel to choose when using kernel K-means')
    parser.add_argument('--gamma', type=float, default=None, help='bandwidth for rbf or polynomial kernel when using kernel K-means')
    parser.add_argument('--sample_weight', action='store_true', help='whether to adapt sample weight when using kernel K-means')
    parser.add_argument('--initial_cluster', type=int, default=1, help='target or source class centroids for initialization of K-means')
    parser.add_argument('--init_cen_on_st', action='store_true', help='whether to initialize learnable cluster centers on both source and target instances')
    parser.add_argument('--src_cen_first', action='store_true', help='whether to use source class centroids as initial target cluster centers at the first epoch')
    parser.add_argument('--src_cls', action='store_true', help='whether to classify source instances when clustering target instances')
    parser.add_argument('--src_fit', action='store_true', help='whether to use convex combination of true label vector and predicted label vector as training guide')
    parser.add_argument('--src_pretr_first', action='store_true', help='whether to perform clustering over features extracted by source pre-trained model at the first epoch')
    parser.add_argument('--learn_embed', action='store_true', help='whether to apply embedding clustering')
    parser.add_argument('--no_second_embed', action='store_true', help='whether to not apply embedding clustering on output features of the first FC layer')
    parser.add_argument('--alpha', type=float, default=1.0, help='degrees of freedom of Student\'s t-distribution')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of auxiliary target distribution or assigned cluster labels')
    parser.add_argument('--embed_softmax', action='store_true', help='whether to use softmax to normalize soft cluster assignments for embedding clustering')
    parser.add_argument('--div', type=str, default='kl', help='measure of prediction divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--gray_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and gray images on the target domain')
    parser.add_argument('--aug_tar_agree', action='store_true', help='whether to enforce the consistency between RGB and augmented images on the target domain')
    parser.add_argument('--sigma', type=float, default=0.1, help='standard deviation of Gaussian for data augmentation operation of blurring')
    # checkpoints
    parser.add_argument('--resume', type=str, default='', help='checkpoints path to resume')
    parser.add_argument('--log', type=str, default='./checkpoints/N2AwA/', help='log folder')
    parser.add_argument('--stop_epoch', type=int, default=200, metavar='N', help='stop epoch for early stop (default: 200)')
    # architecture
    parser.add_argument('--GenA_type', type=str, default='FC', help='GenA type:[FC, MulDis, GCN]')
    parser.add_argument('--arch', type=str, default='resnet50', help='model name')
    parser.add_argument('--num_neurons', type=int, default=128, help='number of neurons of fc1')
    parser.add_argument('--step1', type=str, default='train', help='whether to train step 1') # 'train' | 'load' | 'resume'
    parser.add_argument('--step2', type=str, default='train', help='whether to train step 2')
    parser.add_argument('--step3', type=str, default='train', help='whether to train step 3')
    parser.add_argument('--combine_za', type=bool, default=True, help='Whether or not combine z and att to Clf')
    # i/o
    parser.add_argument('--print_freq', type=int, default=10, metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args()
    args.pretrained = True

    args.src_cls = True
    args.src_cen_first = True
    args.learn_embed = True
    args.embed_softmax = True
    args.log = args.log + '_' + args.src + '2' + args.tgt + '_bs' + str(args.batch_size) + '_lr' + str(args.lr)

    return args
