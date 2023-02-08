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
    parser.add_argument('-ed', '--episode_dot', help='number of episodes for dot training', default='1000')
    parser.add_argument('-et', '--episode_tod', help='number of episodes for tod training', default='2000')
    parser.add_argument('-nb', '--nb_class', help='number of class')
    parser.add_argument('-a', '--learning_rate', help='base learning rate', default='0.00008')
    parser.add_argument('-gd', '--gamma_dot', help='gamma factor of the discounted rewards for dot', default='0.1')
    parser.add_argument('-gt', '--gamma_tod', help='gamma factor of the discounted rewards for tod', default='0.5')

    parser.add_argument('-epsd', '--epsilon_dot', help='epsilon for the e-greedy of dot', default='0.3')
    parser.add_argument('-epst', '--epsilon_tod', help='epsilon fot the e-greedy of tod', default='0.4')

    parser.add_argument('-lrg', '--lr_gamma', help='the factor between 0 an 1 that the learning rate will decay each '
                                                   '100 episode', default='1.')
    parser.add_argument('-pm', '--plot_metrics', action='store_true')

    args = parser.parse_args()

    trainer = Trainer(int(args.nb_class), learning_rate=float(args.learning_rate),
                      gamma_dot=float(args.gamma_dot), gamma_tod=float(args.gamma_tod),
                      lr_gamma=float(args.lr_gamma), e_tod=float(args.epsilon_tod), e_dot=float(args.epsilon_dot))

    trainer.train_dot(int(args.episode_dot), args.train_path, plot_metric=args.plot_metrics)
    trainer.train_tod(int(args.episode_tod), args.train_path, plot_metric=args.plot_metrics)
    trainer.evaluate(args.test_path, plot_metric=args.plot_metrics)
