import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')

# optimizer
parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'rmsprop', 'adam', 'radam'])
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int, metavar='LR', help='initial model last layer rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov', action='store_false', help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M', help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M', help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M', help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

# training
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--start_epoch", default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=30, type=int, metavar='N')

parser.add_argument('--image-size', type=int, default=288)
parser.add_argument('--arch', default='alexnet', choices=['resnet34', 'resnet18', 'resnet50'])
parser.add_argument('--num_classes', default=2, type=int)

args = parser.parse_args()
