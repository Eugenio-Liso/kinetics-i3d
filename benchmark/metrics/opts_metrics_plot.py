import argparse


def parse_opts_metrics_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='metrics.csv', type=str,
                        help='CSV containing metrics')
    parser.add_argument('--classes_list', default='', type=str, required=True,
                        help='A text file containing the possible outcomes. One class per row.')

    args = parser.parse_args()

    return args
