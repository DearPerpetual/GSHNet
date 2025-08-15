from test import testing
import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/media/tyh/CTGU-Work/GSHNet/datasets', type=str, help='data path')
    parser.add_argument('--img_size', default=384, type=int, help='network input size')
    parser.add_argument('--method', default='model', type=str, help='M3Net with different backbone')
    parser.add_argument('--pretrained_model', default='./pretrained_model/', type=str, help='load Pretrained model')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--save_model', default='output/model/', type=str, help='save model path')
    parser.add_argument('--test', default=False, type=bool, help='test or not')
    parser.add_argument('--save_test', default='output/preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_methods', type=str, default='PowerLine')
    parser.add_argument('--record', default='./record.txt', type=str, help='record file')
    args = parser.parse_args()

    if args.test:
        testing(args=args)
