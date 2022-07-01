# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to data',
                        default='/Users/wangzi/PycharmProjects/few-shot-learning/Prototypical-Networks-for-Few-shot-Learning-PyTorch/data')
    parser.add_argument('-dname', '--dataset_name',
                        type=str,
                        help='which dataset to use',
                        default='miniImagenet')
    parser.add_argument('-model', '--model_name',
                        type=str,
                        help='which dataset to use',
                        default='vit')

    parser.add_argument('-height', '--height',
                        type=int,
                        help='image resized height',
                        default=64)

    parser.add_argument('-width', '--width',
                        type=int,
                        help='image resized width',
                        default=64)

    parser.add_argument('-channel', '--channel',
                        type=int,
                        help='image resized width',
                        default=3)

    parser.add_argument('-dist', '--dist',
                        type=str,
                        help='which dist loss to use',
                        default='euclidean')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')

    parser.add_argument('-opt', '--optimizer',
                        type=str,
                        help='which optimizer to use',
                        default='Adam')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.0001)

    parser.add_argument('-wd', '--weight_decay',
                        type=float,
                        help='l2 regulazation rate',
                        default=0.0)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=50)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.9)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-use_aux_loss', '--use_aux_loss',
                        type=bool,
                        help='whether to use aux loss',
                        default=False)

    parser.add_argument('-use_join_loss', '--use_join_loss',
                        type=bool,
                        help='whether to use join loss',
                        default=False)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=20)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=15)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=16)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)

    parser.add_argument('-total_classes', '--total_classes',
                        type=int,
                        help='number of dataset total classes, mini-imagenet: 100',
                        default=100)

    parser.add_argument('-warm_up_epochs', '--warm_up_epochs',
                        type=int,
                        help='warm up epochs',
                        default=5)
    parser.add_argument('-balance_scale', '--balance_scale',
                        type=float,
                        help='scale cross entropy loss to aux euclidean distance loss',
                        default=1.0)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)
    parser.add_argument('-pretrained', '--pretrained',
                        type=bool,
                        help='if or not to use pretrained model',
                        default=False)
    parser.add_argument('--cuda',
                        type=str,
                        help='use gpu or cpu',
                        default='-1')

    parser.add_argument('--comment',
                        type=str,
                        help='comment',
                        default='')
    return parser
