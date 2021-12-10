import argparse


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, required=True)
    args.add_argument('--batch', type=int, default=8)
    args.add_argument('--lr', type=float, default=3e-5)
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--size', type=int, default=600)
    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--norm', type=bool, default=False)
    args.add_argument('--datapath', type=str, default='/root/datasets/open')
    args.add_argument('--mask', action='store_true')
    args.add_argument('--patch', action='store_true')
    args.add_argument('--self', action='store_true')
    args.add_argument('--resume', action='store_true')
    return args.parse_args()
