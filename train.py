import argparse
import os
import shutil
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.loss import LabelSmoothingCrossEntropy
from thop import profile
from torchstat import stat
from ptflops import get_model_complexity_info
import torch.optim as optim
from utils import WarmUpLR

my_model = resnext50_32x4d(num_classes=1000)
pre_model = torchvision.models.resnext50_32x4d(pretrained=True)
pre_dict = pre_model.state_dict()
my_model_dict = my_model.state_dict()

need_pre_dict = {k: v for k, v in pre_dict.items() if k in my_model_dict}
my_model_dict.update(need_pre_dict)
my_model.load_state_dict(my_model_dict)
model = my_model


print(model)
# Training settings
parser = argparse.ArgumentParser(description='PyTorch scen classification training')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./log_aid20_r8', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

pretrained_size = 288
#UCM20
train_means = [0.4858, 0.4915, 0.4518]
train_stds= [0.1739, 0.1640, 0.1564]

test_means = [0.4827, 0.4886, 0.4492]
test_stds= [0.1731, 0.1630, 0.1545]


#AID50
# train_means = [0.3989, 0.4096, 0.3691]
# train_stds= [0.1586, 0.1460, 0.1408]
#
# test_means = [0.3967, 0.4089, 0.3679]
# test_stds= [0.1573, 0.1442, 0.1397]

# #NWPU45_10
# train_means = [0.3655, 0.3785, 0.3413]
# train_stds= [0.1452, 0.1355, 0.1320]
#
# test_means = [0.3683, 0.3813, 0.3438]
# test_stds= [0.1454, 0.1356, 0.1320]

#NWPU45_20
# train_means = [0.3655, 0.3785, 0.3413]
# train_stds= [0.1452, 0.1355, 0.1320]
#
# test_means = [0.3684, 0.3812, 0.3438]
# test_stds= [0.1454, 0.1356, 0.1320]

train_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.RandomRotation(270),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(train_means, train_stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.CenterCrop(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(test_means, test_stds)
                       ])

train_data = datasets.ImageFolder(root = "data/AID50/train/",
                                  transform = train_transforms)

test_data = datasets.ImageFolder(root = "data/AID50/test/",
                                 transform = test_transforms)

train_iterator = data.DataLoader(train_data,
                                 shuffle = True,
                                 batch_size = args.batch_size)

test_iterator = data.DataLoader(test_data,
                                batch_size = args.batch_size)


from torch.optim import lr_scheduler
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

criterion = LabelSmoothingCrossEntropy()
# criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)


n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
from torchsummaryX import summary

def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_iterator):
        if args.cuda:
            data, target = data.to(device), target.to(device)

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # print(summary(model, data))
        # flops, params = profile(model, inputs=(data.to(device),))
        macs, params = get_model_complexity_info(model, (3,288,288), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        # print("flops:.6f", macs, "params:.6f", params, "\n")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        loss = criterion(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {}/{}\tTrain lr: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch,args.epochs,optimizer.param_groups[0]['lr'], batch_idx * len(data), len(train_iterator.dataset),
                100. * batch_idx / len(train_iterator), loss.item()))
def test():
    model.eval()
    test_loss = 0
    correct = 0
    # import time
    # start = time.time()
    time_z = 0
    for data, target in test_iterator:

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        with torch.no_grad():
            import time
            start = time.time()
            output = model(data)
            end = time.time()
        time_z = time_z + (end - start)
        # print("time: {}".format((end - start)/len(data)))
        test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # end = time.time()
    # time = end - start
    test_loss /= len(test_iterator.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Time: {}\n'.format(
        test_loss, correct, len(test_iterator.dataset),
        100.00 * correct / len(test_iterator.dataset),
        time_z/len(test_iterator.dataset)

    ))
    return correct / float(len(test_iterator.dataset))



def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 10))
    if epoch == 0:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train(epoch)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model
    }, is_best, filepath=args.save)























#
# import torch.nn.functional as F
# def get_predictions(model, iterator):
#
#     model.eval()
#
#     images = []
#     labels = []
#     probs = []
#
#     with torch.no_grad():
#
#         for (x, y) in iterator:
#
#             x = x.to(device)
#
#             y_pred = model(x)
#
#             y_prob = F.softmax(y_pred, dim = -1)
#             top_pred = y_prob.argmax(1, keepdim = True)
#
#             images.append(x.cpu())
#             labels.append(y.cpu())
#             probs.append(y_prob.cpu())
#
#     images = torch.cat(images, dim = 0)
#     labels = torch.cat(labels, dim = 0)
#     probs = torch.cat(probs, dim = 0)
#
#     return images, labels, probs
# images, labels, probs = get_predictions(model, test_iterator)
#
# pred_labels = torch.argmax(probs, 1)
#
#
# def plot_confusion_matrix(labels, pred_labels, classes):
#     fig = plt.figure(figsize=(50, 50))
#     ax = fig.add_subplot(1, 1, 1)
#     cm = confusion_matrix(labels, pred_labels)
#     bm = ConfusionMatrixDisplay(cm, classes)
#     bm.plot(values_format='d', cmap='Blues', ax=ax)
#     fig.delaxes(fig.axes[1])  # delete colorbar
#     plt.xticks(rotation=90)
#     plt.xlabel('Predicted Label', fontsize=80)
#     plt.ylabel('True Label', fontsize=80)
#
#
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import os
# classes = os.listdir('data/train/')
#
# plot_confusion_matrix(labels, pred_labels, classes)