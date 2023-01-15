# This is a sample Python script.
import argparse

from components.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to train DOT/TOD on a specific dataset. The weight and result are saved '
                    'in the current directory')
    parser.add_argument('-tr', '--train_path',
                        help='the path to the data. it must contains a img folder containing the train image and a '
                             'bboxes folder containing the bounding boxes')
    parser.add_argument('-ts', '--test_path',
                        help='the path to the data. it must contains a img folder containing the train image and a '
                             'bboxes folder containing the bounding boxes')
    parser.add_argument('-w', '--weights', help='the path to the weights if needed')
    parser.add_argument('-e', '--episode', help='number of episodes', default='500')
    parser.add_argument('-nb', '--nb_class', help='number of class')
    parser.add_argument('-a', '--learning_rate', help='base learning rate', default='0.0005')
    parser.add_argument('-g', '--gamma', help='gamma factor of the discounted rewards', default='0.1')
    parser.add_argument('-lrg', '--lr_gamma', help='the factor between 0 an 1 that the learning rate will decay each '
                                                   '100 episode', default='0.8')
    parser.add_argument('-pm', '--plot_metrics', action='store_true')

    args = parser.parse_args()

    trainer = Trainer(int(args.nb_class), learning_rate=float(args.learning_rate),
                      gamma=float(args.gamma), lr_gamma=float(args.lr_gamma))
    "../../Seal/Train"
    trainer.train(int(args.episode), args.train_path, plot_metric=args.plot_metrics)
    trainer.evaluate(args.test_path, plot_metric=args.plot_metrics)
