import argparse


def parse_opts_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output.json', type=str,
                        help='Output json from prediction phase')
    parser.add_argument('--labeled_videos', default='', type=str,
                        help='A JSON containing a list of tuples (video name, ground truth label)')
    parser.add_argument('--confidence_threshold', default=0.8, type=float,
                        help='If the prediction score is equal or greater than this threshold, than it will be counted '
                             'as a TP, otherwise as a FP.')

    args = parser.parse_args()

    return args
