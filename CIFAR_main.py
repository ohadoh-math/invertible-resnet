"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import threading
import logging
import contextlib
import numpy
import torch
import termcolor
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Subset
from save_data import save_data, METADATA_DIR
import numpy as np
import torchvision
import torchvision.transforms as transforms
import visdom
import os
import sys
import time
import argparse
import pdb
import random
import json
from models.utils_cifar import train, test, std, mean, get_hms, interpolate
from models.conv_iResNet import conv_iResNet as iResNet
from models.conv_iResNet import multiscale_conv_iResNet as multiscale_iResNet

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation',
                    action='store_true', help='perform density estimation')
parser.add_argument('--optimizer', default="adamax", type=str, help="optimizer", choices=["adam", "adamax", "sgd"])
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--coeff', default=0.9, type=float, help='contraction coefficient for linear layers')
parser.add_argument('--numTraceSamples', default=1, type=int, help='number of samples used for trace estimation')
parser.add_argument('--numSeriesTerms', default=1, type=int, help='number of terms used in power series for matrix log')
parser.add_argument('--powerIterSpectralNorm', default=5, type=int, help='number of power iterations used for spectral norm')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='coefficient for weight decay')
parser.add_argument('--drop_rate', default=0.1, type=float, help='dropout rate')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--init_batch', default=1024, type=int, help='init batch size')
parser.add_argument('--init_ds', default=2, type=int, help='initial downsampling')
parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')
parser.add_argument('--inj_pad', default=0, type=int, help='initial inj padding')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--nBlocks', nargs='+', type=int, default=[4, 4, 4])
parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 2, 2])
parser.add_argument('--nChannels', nargs='+', type=int, default=[16, 64, 256])
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-interpolate', '--interpolate', dest='interpolate', action='store_true', help='train iresnet')
parser.add_argument('-drop_two', '--drop_two', dest='drop_two', action='store_true', help='2d dropout on')
parser.add_argument('-nesterov', '--nesterov', dest='nesterov', action='store_true',
                    help='nesterov momentum')
parser.add_argument('-norm', '--norm', dest='norm', action='store_true',
                    help='compute norms of conv operators')
parser.add_argument('-analysisTraceEst', '--analysisTraceEst', dest='analysisTraceEst', action='store_true',
                    help='analysis of trace estimation')
parser.add_argument('-multiScale', '--multiScale', dest='multiScale', action='store_true',
                    help='use multiscale')
parser.add_argument('-fixedPrior', '--fixedPrior', dest='fixedPrior', action='store_true',
                    help='use fixed prior, default is learned prior')
parser.add_argument('-noActnorm', '--noActnorm', dest='noActnorm', action='store_true',
                    help='disable actnorm, default uses actnorm')
parser.add_argument('--nonlin', default="elu", type=str, choices=["relu", "elu", "sorting", "softplus"])
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--save_dir', default=None, type=str, help='directory to save results')
parser.add_argument('--vis_port', default=8097, type=int, help="port for visdom")
parser.add_argument('--vis_server', default="localhost", type=str, help="server for visdom")
parser.add_argument('--log_every', default=10, type=int, help='logs every x iters')
parser.add_argument('-log_verbose', '--log_verbose', dest='log_verbose', action='store_true',
                    help='verbose logging: sigmas, max gradient')
parser.add_argument('-deterministic', '--deterministic', dest='deterministic', action='store_true',
                    help='fix random seeds and set cuda deterministic')
parser.add_argument('--trunc', type=float, default=1., help='Truncate the data and test sets by percentage.')


def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

try_make_dir('results')

def anaylse_trace_estimation(model, testset, use_cuda, extension):
    # setup range for analysis
    numSamples = np.arange(10)*10 + 1
    numIter = np.arange(10)
    # setup number of datapoints
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    # TODO change
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        # compute trace
        out_bij, p_z_g_y, trace, gt_trace = model(inputs[:, :, :8, :8],
                                                       exact_trace=True)
        trace = [t.cpu().numpy() for t in trace]
        np.save('gtTrace'+extension, gt_trace)
        np.save('estTrace'+extension, trace)
        return
    

def test_spec_norm(model, in_shapes, extension):
    i = 0
    j = 0
    params = [v for v in model.module.state_dict().keys() \
              if "bottleneck" and "weight" in v \
              and not "weight_u" in v \
              and not "weight_orig" in v \
              and not "bn1" in v and not "linear" in v]
    logging.info(len(params))
    logging.info(len(in_shapes))
    svs = [] 
    for param in params:
      if i == 0:
        input_shape = in_shapes[j]
      else:
        input_shape = in_shapes[j]
        input_shape[1] = int(input_shape[1] // 4)

      convKernel = model.module.state_dict()[param].cpu().numpy()
      input_shape = input_shape[2:]
      fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
      t_fft_coeff = np.transpose(fft_coeff)
      U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
      Dflat = np.sort(D.flatten())[::-1] 
      logging.info("Layer "+str(j)+" Singular Value "+str(Dflat[0]))
      svs.append(Dflat[0])
      if i == 2:
        i = 0
        j+= 1
      else:
        i+=1
    np.save('singular_values'+extension, svs)
    return


def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init
    """
    batches = []
    seen = 0
    for x, y in dataloader:
        batches.append(x)
        seen += x.size(0)
        if seen >= batch_size:
            break
    batch = torch.cat(batches)
    return batch

def get_model(
    multiScale,
    in_shape,
    nBlocks,
    nStrides,
    nChannels,
    nClasses,
    init_ds,
    inj_pad,
    coeff,
    numTraceSamples,
    numSeriesTerms,
    powerIterSpectralNorm,
    densityEstimation,
    noActnorm,
    fixedPrior,
    nonlin,
):
    if multiScale:
        model = multiscale_iResNet(
            in_shape,
            nBlocks,
            nStrides,
            nChannels,
            init_ds == 2,
            inj_pad,
            coeff,
            densityEstimation,
            nClasses,
            numTraceSamples,
            numSeriesTerms,
            powerIterSpectralNorm,
            actnorm=(not noActnorm),
            learn_prior=(not fixedPrior),
            nonlin=nonlin,
        )
    else:
        model = iResNet(
            nBlocks=nBlocks,
            nStrides=nStrides,
            nChannels=nChannels,
            nClasses=nClasses,
            init_ds=init_ds,
            inj_pad=inj_pad,
            in_shape=in_shape,
            coeff=coeff,
            numTraceSamples=numTraceSamples,
            numSeriesTerms=numSeriesTerms,
            n_power_iter=powerIterSpectralNorm,
            density_estimation=densityEstimation,
            actnorm=(not noActnorm),
            learn_prior=(not fixedPrior),
            nonlin=nonlin,
        )
    return model


def cifar10_classification_model():
    """
    CIFAR10 classification
    """
    return get_model(
        False,
        (3, 32, 32),
        [7,7,7],
        [1,2,2],
        [32,64,128],
        10,
        1,
        13,
        0.9,
        1,
        1,
        1,
        False,
        False,
        False,
        "elu",
    )


def main():
    args = parser.parse_args()
    assert 0. < args.trunc, Exception("Can't truncate with negative percentage", args.trunc)

    logging.basicConfig(
        format="%(asctime)s (pid=%(process)d:tid=%(thread)d) [%(levelname)s:%(funcName)s:%(filename)s:%(lineno)d] %(message)s",
        level=logging.DEBUG,
    )

    if args.deterministic:
        logging.info("MODEL NOT FULLY DETERMINISTIC")
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        torch.backends.cudnn.deterministic=True

    dens_est_chain = [
        lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
        lambda x: x / 256.,
        lambda x: x - 0.5
    ]
    if args.dataset == 'mnist':
        assert args.densityEstimation, "Currently mnist is only supported for density estimation"
        mnist_transforms = [transforms.Pad(2, 0), transforms.ToTensor(), lambda x: x.repeat((3, 1, 1))]
        transform_train_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
        transform_test_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform_train_mnist)
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=False, transform=transform_test_mnist)
        args.nClasses = 10
        in_shape = (3, 32, 32)
    else:
        if args.dataset == 'svhn':
            train_chain = [transforms.Pad(4, padding_mode="symmetric"),
                           transforms.RandomCrop(32),
                           transforms.ToTensor()]
        else:
            train_chain = [transforms.Pad(4, padding_mode="symmetric"),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]

        test_chain = [transforms.ToTensor()]
        if args.densityEstimation:
            transform_train = transforms.Compose(train_chain + dens_est_chain)
            transform_test = transforms.Compose(test_chain + dens_est_chain)
        else:
            clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
            transform_train = transforms.Compose(train_chain + clf_chain)
            transform_test = transforms.Compose(test_chain + clf_chain)


        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            args.nClasses = 10
        elif args.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)
            args.nClasses = 100
        elif args.dataset == 'svhn':
            trainset = torchvision.datasets.SVHN(
                root='./data', split='train', download=True, transform=transform_train)
            testset = torchvision.datasets.SVHN(
                root='./data', split='test', download=True, transform=transform_test)
            args.nClasses = 10
        in_shape = (3, 32, 32)

    if args.trunc < 1:
        orig_train_len = len(trainset)
        orig_test_len = len(testset)

        random_train_indices = list(
            numpy.random.choice(
                numpy.arange(len(trainset)),
                int(args.trunc * len(trainset)),
                replace=False,
            )
        )
        trainset = Subset(trainset, random_train_indices)

        random_test_indices = list(
            numpy.random.choice(
                numpy.arange(len(testset)),
                int(args.trunc * len(testset)),
                replace=False,
            )
        )
        testset = Subset(testset, random_test_indices)

        logging.info(
            "truncated train set size to %.02f%%: %d -> %d",
            100 * args.trunc,
            orig_train_len,
            len(trainset),
        )
        logging.info(
            "truncated test set size to %.02f%%: %d -> %d",
            100 * args.trunc,
            orig_test_len,
            len(testset),
        )
    else:
        logging.info(termcolor.colored("not truncating dataset", "red"))


    # setup logging with visdom
    viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
    assert viz.check_connection(), "Could not make visdom"

    if args.deterministic:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                                  shuffle=True, num_workers=2, worker_init_fn=np.random.seed(1234))
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                                 shuffle=False, num_workers=2, worker_init_fn=np.random.seed(1234))
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)

    model = get_model(
        args.multiScale,
        in_shape,
        args.nBlocks,
        args.nStrides,
        args.nChannels,
        args.nClasses,
        args.init_ds,
        args.inj_pad,
        args.coeff,
        args.numTraceSamples,
        args.numSeriesTerms,
        args.powerIterSpectralNorm,
        args.densityEstimation,
        args.noActnorm,
        args.fixedPrior,
        args.nonlin,
    )

    # init actnrom parameters
    init_batch = get_init_batch(trainloader, args.init_batch)

    logging.info("cuda specs")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        init_batch = init_batch.cuda()
    else:
        in_shapes = model.get_in_shapes()

    logging.info("initializing actnorm parameters...")
    logging.info("batch device = %r", init_batch.device)

    save_data(METADATA_DIR/"init-batch.h5", init_batch=init_batch)
    with torch.no_grad():
        # there is some wierd exception going on here in our setup.
        # calling `model` the frist time raises that error and the next time it seems fine.
        with contextlib.suppress(Exception):
            model(init_batch)

        model(init_batch, ignore_logdet=True)

    if use_cuda:
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
        cudnn.benchmark = True
        in_shapes = model.module.get_in_shapes()

    logging.info("initialized")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_objective = checkpoint['objective']
            logging.info('objective: '+str(best_objective))
            model = checkpoint['model']
            if use_cuda:
                model.module.set_num_terms(args.numSeriesTerms)
            else:
                model.set_num_terms(args.numSeriesTerms)
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    try_make_dir(args.save_dir)
    if args.analysisTraceEst:
        anaylse_trace_estimation(model, testset, use_cuda, args.extension)
        return

    if args.norm:
        test_spec_norm(model, in_shapes, args.extension) 
        return

    if args.interpolate:
        interpolate(model, testloader, testset, start_epoch, use_cuda, best_objective, args.dataset)
        return

    if args.evaluate:
        test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
        if use_cuda:
            model.module.set_num_terms(args.numSeriesTerms)
        else:
            model.set_num_terms(args.numSeriesTerms)
        model = torch.nn.DataParallel(model.module)
        test(best_objective, args, model, start_epoch, testloader, viz, use_cuda, test_log)
        return

    logging.info('|  Train Epochs: ' + str(args.epochs))
    logging.info('|  Initial Learning Rate: ' + str(args.lr))

    elapsed_time = 0
    test_objective = -np.inf

    logging.info("optimizer = %r", args.optimizer)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

    with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))

    train_log = open(os.path.join(args.save_dir, "train_log.txt"), 'w')

    logging.info("TRAINING!")
    for epoch in range(1, 1+args.epochs):
        start_time = time.time()
        logging.debug("start << training epoch %d", epoch)
        train(args, model, optimizer, epoch, trainloader, trainset, viz, use_cuda, train_log)
        logging.debug("finish >> training epoch %d", epoch)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        logging.info('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    logging.info('Testing model')
    test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
    test_objective = test(test_objective, args, model, epoch, testloader, viz, use_cuda, test_log)
    logging.info('* Test results : objective = %.2f%%' % (test_objective))
    with open(os.path.join(args.save_dir, 'final.txt'), 'w') as f:
        f.write(str(test_objective))

    model_file = os.path.join(args.save_dir, 'model.tr')
    logging.info("saving model to file %s", model_file)
    torch.save(model.cpu(), model_file)


if __name__ == '__main__':
    main_tid = threading.main_thread().ident
    try:
        main()
    except Exception:
        if threading.main_thread().ident == threading.current_thread().ident:
            logging.error("critical error", exc_info=True)
        sys.exit(1) #raise

