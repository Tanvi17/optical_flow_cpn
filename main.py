import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from data_loader import *
from image_transform import preprocessing
import models
import torchvision.transforms as transforms
# import datasets
from loss.multiscale import multiscaleEPE, realEPE
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.common import flow2rgb, AverageMeter, save_checkpoint


DATASET_PATH = '/media/common/datasets/scene_flow_datasets/FlyingChairs_release/'
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
print(model_names)

parser = argparse.ArgumentParser(description='PyTorch CPN Training on FlyingChairs dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', metavar='DIR', default=DATASET_PATH, type=str,
                    help='path to dataset')
parser.add_argument('--n_iter', default=0,
                    help='number of times train eval is iterated')
parser.add_argument('--solver', default='adam', choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=8, type=int, metavar='N', 
                    help='batch size')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR', 
                    help='initial learning rate')

parser.add_argument('--seed_split', default=42)
parser.add_argument('--div_flow', default=20.0, type=float)
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))


##to use or not
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split_file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split_value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument(
    "--split_seed",
    type=int,
    default=None,
    help="Seed the train-val split to enforce reproducibility (consistent restart too)",
)



parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--alpha', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
#for pretrained model
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')


args = parser.parse_args()
best_EPE = -1
n_iter = args.n_iter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

def main():
    global args, best_EPE
    args = parser.parse_args()

    #save models
    params = 'model_opt{}_epochs{}_bs{}_lr{}'.format(
        args.solver,
        args.epochs,
        args.batch_size,
        args.lr)
    
    # add timestamp
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    model_path = os.path.join(timestamp, params)
    model_path = os.path.join('saved_models',model_path)
    print('=> saving model at {}'.format(model_path))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if args.seed_split is not None:
        np.random.seed(args.seed_split)

    train_writer = SummaryWriter(os.path.join(model_path, 'train'))
    test_writer = SummaryWriter(os.path.join(model_path, 'test'))
    output_writers = []

    # Data loading code and preprocessing
    '''
    img path - PIL img - tensor - 
    1. ArrayToTensor: Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    2. Hist. Eq.: Tensor -> Tensor
    3. Normalize
    4. Subtract -0.5
    '''
    input_image_transform = transforms.Compose([
        preprocessing.HistogramEqualization(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0]),
        preprocessing.CustomRange()
    ])

    #for more: https://pytorch.org/vision/master/transforms.html
    co_transform = transforms.Compose([
        transforms.RandomCrop((320,448)),
        transforms.RandomAffine(degrees=(5,10), translate=(0.2,0.2)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ])

    target_transform = transforms.Compose([
        preprocessing.ReadFromFile(),
        transforms.ToTensor(),
        # transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = flying_chairs(root=args.data, 
                                        transform=input_image_transform, 
                                        target_transform=target_transform,
                                        co_transform=co_transform, split=None)

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))

    
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.cpn.CPN(network_data).to(device)

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]

    # if device.type == "cuda":
    #     model = torch.nn.DataParallel(model).cuda()
    #     cudnn.benchmark = True

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    if args.evaluate:
        best_EPE = validate(val_loader, model, 0, output_writers)
        return

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        train_loss, train_EPE = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)

        # evaluate on validation set

        with torch.no_grad():
            EPE = validate(val_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean EPE', EPE, epoch)

        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best, model_path)


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)
        input = torch.cat(input,1).to(device)

        # compute output
        output = model((input, target))
        if args.sparse:
            # Since Target pooling is not very precise when sparse,
            # take the highest resolution prediction and upsample it instead of downsampling target
            h, w = target.size()[-2:]
            output = [F.interpolate(output[0], (h,w)), *output[1:]]

        loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)
        # record loss and EPE
        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_iter)
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t EPE {6}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses, flow2_EPEs))
        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_EPEs.avg


def validate(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = torch.cat(input,1).to(device)

        # compute output
        output = model((input, target))
        flow2_EPE = args.div_flow*realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print('output writers', output_writers)
        if i < len(output_writers):  # log first output of first batches
            if epoch == args.start_epoch:
                mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
                output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))

    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg

if __name__ == '__main__':
    main()
