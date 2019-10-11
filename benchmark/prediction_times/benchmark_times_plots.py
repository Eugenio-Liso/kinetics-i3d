import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], "../../logging_utils"))

from math import ceil
import matplotlib.pyplot as plt
import json
from opts_pred_times import parse_opts_benchmark
from logger_factory import getBasicLogger
import numpy as np
import seaborn as sns
sns.set()

logger = getBasicLogger(os.path.basename(__file__))


def build_benchmark_results(exec_times, max_videos_in_row):
    num_videos = len(exec_times)

    logger.info("Number of videos to display: {}".format(num_videos))

    if num_videos <= max_videos_in_row:
        columns = 1
    else:
        columns = int(ceil(len(exec_times) / max_videos_in_row))
    _ = plt.figure('Inference Times')

    logger.info(
        "Chart will have {} rows and {} columns. Note that this configuration may be not respected by Matplotlib "
        "renderer.".format(num_videos, columns))

    build_plot(columns, exec_times, num_videos)
    display_results()


def build_plot(columns, exec_times, rows):
    for current_video, exec_time_single_video in enumerate(exec_times, start=1):
        plt.subplot(rows, columns, current_video)

        extract_data(exec_time_single_video, False)


def extract_data(exec_time_single_video, single_plot):
    for video, times in exec_time_single_video.items():
        if single_plot:
            plt.figure(video)

        samples_points_y = []

        for sample_prediction in times:
            # tensor = sample[0]
            single_execution_time = sample_prediction[1]
            samples_points_y.append(single_execution_time)

        batches_count_x = range(1, len(samples_points_y) + 1)  # Exclusive upper bound

        # Calculate the simple average of the data
        y_mean = [np.mean(samples_points_y)] * len(batches_count_x)

        plt.plot(batches_count_x, samples_points_y, label='Data', marker='o')
        plt.plot(batches_count_x, y_mean, label='Mean', linestyle='--')
        plt.ylabel('Prediction time (sec)')
        plt.xlabel('Number of frame batches')

        # Make a legend
        plt.legend(loc='upper right')


def display_results():
    plt.subplots_adjust(wspace=0.5, hspace=1)
    plt.show()


if __name__ == '__main__':
    opt = parse_opts_benchmark()
    output_json_exec_times_path = opt.output_times

    logger.info("Input json of execution times: {}".format(output_json_exec_times_path))

    with open(output_json_exec_times_path, 'r') as f:
        json_exec_times = json.load(f)

    max_videos = opt.max_videos_rows

    build_benchmark_results(json_exec_times, max_videos)
