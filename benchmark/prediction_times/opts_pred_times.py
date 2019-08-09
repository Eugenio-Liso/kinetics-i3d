import argparse


def parse_opts_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_times', default='output_times.json', type=str,
                        help='Output json from prediction phase with execution times')
    parser.add_argument('--max_videos_rows', default=3, type=int,
                        help='Max number of rows per column displayed')

    args = parser.parse_args()

    return args
