import argparse


def parse_opts_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output.json', type=str, required=True,
                        help='Output json from prediction phase')
    parser.add_argument('--labeled_videos', default='', type=str, required=True,
                        help='A JSON containing a list of tuples (video name, ground truth label)')
    parser.add_argument('--classes_list', default='', type=str, required=True,
                        help='A text file containing the possible outcomes. One class per row.')
    parser.add_argument('--output_csv', default='metrics.csv', type=str,
                        help='The output path of the csv with the calculated results')
    parser.add_argument('--output_mean_times', default='', type=str, required=True,
                        help='Output json from prediction phase with mean execution times')

    args = parser.parse_args()

    return args
