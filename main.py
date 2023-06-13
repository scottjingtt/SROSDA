import random
import torch.backends.cudnn as cudnn
from networks import build_model
from opts import opts # options for the project
from prepare_data import generate_dataloader # prepare the data and dataloader
from evaluation import *
from step3 import train_step3
from utils import get_clu_centers, resume_pretrained_weights  # *

args = opts()
subdomain = ['AwA', 'painting', 'real']  # 'clipart', 'infograph', 'quickdraw', 'sketch'
class_num = 17
src = subdomain[0]
tgt = subdomain[2]
args.src = src
args.tgt = tgt

args.step3 = 'train'
args.att_type = 'binary'  # 'binary'|'continuous'
args.combine_za = True  # f=[z,a] | f=z
args.init_prot_type = 'sample'  # 'center' | 'sample'
args.GenA_type = 'FC'  # 'GCN', 'FC', 'MulDis'

cuda = False  # True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def main():
    # (1) process data and prepare dataloaders
    train_loader_source, train_loader_target, test_loader_target = generate_dataloader(args)
    dataloaders = {'tr_loader_src': train_loader_source,
                   'tr_loader_tgt': train_loader_target,
                   'te_loader_tgt': test_loader_target}
    # (1.1) Attributes centers groungd truth: (50 x 85)
    if args.att_type == 'binary':
        df_att = pd.read_csv('./data/N2AwA/attributes/attributes_bi.csv', header=None,index_col=None)
    elif args.att_type == 'continuous':
        raise Exception("No continuous attributes")
    else:
        raise ValueError("att type does not exist")

    att_cents = Variable(torch.tensor(df_att.values).type(FloatTensor))

    # (1.2) Init prot centers pseudo: (50 x 512), first 40 are shared classes
    if args.init_prot_type == 'center':
        raise Exception("No center based clu prot")
        # xt_clu_cents = pd.read_csv('./data/N2AwA/pseudo/'+args.src+'2'+args.tgt+'_sample_xt_clu_cents17.csv').values
        # xt_clu_cents = Variable(torch.tensor(xt_clu_cents).type(FloatTensor))
    elif args.init_prot_type == 'sample':
        xt_clu_cents = pd.read_csv('./data/N2AwA/pseudo/' + args.src + '2' + args.tgt + '_sample_xt_clu_cents17.csv').values
        xt_clu_cents = Variable(torch.tensor(xt_clu_cents).type(FloatTensor))
    else:
        raise ValueError("init proto type does not exist")
    centers = {'att_cents': att_cents, 'xt_clu_cents': xt_clu_cents}


    # (2) Buid the mdoels
    x_dim = 2048
    z_dim = 512
    a_dim = 85
    if args.combine_za:
        f_dim = z_dim + a_dim
    else:
        f_dim = z_dim
    shr_nc = args.shr_nc

    GenZ = build_model(args, 'GenZ', input_size=x_dim, h1=1024, h2=z_dim).to(device)
    if args.GenA_type == 'GCN':  # To be explored in the future
        GenA = build_model(args, 'GCN', input_size=z_dim, h1=256, h2=a_dim).to(device)
    elif args.GenA_type == 'FC':
        GenA = build_model(args, 'GenA', input_size=z_dim, h1=256, h2=a_dim).to(device)
    elif args.GenA_type == 'MulDis':
        GenA = build_model(args, 'MulDis', input_size=z_dim, h1=256, h2=a_dim).to(device)  # Binary classifier
    else:
        raise ValueError("GenA_type not exists.")
    Clf = build_model(args, 'Clf', input_size=f_dim, h1=256, h2=shr_nc+1).to(device)
    ClfSU = build_model(args, 'ClfSU', input_size=f_dim, h1=256, h2=2).to(device)  # Seen/Unseen Classifier

    ProtClf = build_model(args, 'Prot').to(device)

    toTrModels = {'GenZ': GenZ, 'Clf': Clf, 'GenA': GenA, 'ClfSU': ClfSU}
    nonTrModels = {'ProtClf': ProtClf}  # No trainable params in the Prototype-classifier


    # (3) Specify loss functions and initialization
    BCELoss = torch.nn.BCELoss().to(device)
    CELoss = torch.nn.CrossEntropyLoss().to(device)
    MSELoss = torch.nn.MSELoss().to(device)
    KLDivLoss = torch.nn.KLDivLoss(reduction='sum').to(device)
    lossFunctions = {'BCELoss': BCELoss, 'CELoss': CELoss, 'MSELoss': MSELoss, 'KLDivLoss': KLDivLoss}

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)

    # (4) Create optimizers
    optimizers = {}
    for m in toTrModels:
        opt_name = 'opt_' + m
        optimizers[opt_name] = torch.optim.Adam(toTrModels[m].parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)


    # # (5) Create log directory
    # if not os.path.isdir(args.log):
    #     os.makedirs(args.log)
    # log = open(os.path.join(args.log, 'train_log.txt'), 'w')
    # state = {k: v for k, v in args._get_kwargs()}
    # log.write(json.dumps(state) + '\n')
    # log.close()
    #
    # log = open(os.path.join(args.log, 'train_log.txt'), 'a')
    # log.write('\n-------------------------------------------\n')
    # log.write(time.asctime(time.localtime(time.time())))
    # log.write('\n-------------------------------------------')
    # log.close()

    cudnn.benchmark = True

    # (6). Start training
    centers['zt_clu_cents'], centers['at_clu_cents'] = get_clu_centers(args, toTrModels, dataloaders, centers, label='init')  # label='init' | 'clf'
    # (6.1) Training (step1&2 are pre-training steps. Deprecated due to none contribution)
    if args.step3 == 'train':
        print('begin training step3')
        train_step3(args, dataloaders, toTrModels, nonTrModels, lossFunctions, optimizers, centers)
    elif args.step3 == 'resume':
        print("Keep training step3!")
        resume_pretrained_weights(args, toTrModels, step='step3')
        train_step3(args, dataloaders, toTrModels, nonTrModels, lossFunctions, optimizers, centers)
    elif args.step3 == 'load':
        print(" Step 3 trained weights already exits!!!")
        resume_pretrained_weights(args, toTrModels, step='step3')
    else:
        print(" Nothing to do with Step 3...")

    eval(args, -3, dataloaders['te_loader_tgt'], toTrModels, nonTrModels, centers)

    # log = open(os.path.join(args.log, 'test_log.txt'), 'a')
    # log.write('\n***   best val acc: %3f   ***' % best_prec1)
    # log.write('\n***   best test acc: %3f   ***' % best_test_prec1)
    # log.write('\n***   cond best test acc: %3f   ***' % cond_best_test_prec1)
    # # end time
    # log.write('\n-------------------------------------------\n')
    # log.write(time.asctime(time.localtime(time.time())))
    # log.write('\n-------------------------------------------\n')
    # log.close()
    #
    #
    #


if __name__ == '__main__':
    main()


