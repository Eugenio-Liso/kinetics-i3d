import csv

import matplotlib.pyplot as plt
import numpy as np
from opts_metrics_plot import parse_opts_metrics_plot
import seaborn as sns
sns.set()

# from matplotlib.ticker import FormatStrFormatter


def get_metric(class_names, idx_metrics, row):
    result = []
    for _ in class_names:
        result.append(float(row[idx_metrics]))
        idx_metrics += 1
    return result, idx_metrics


def build_plot(idx_chart, classes_metric, class_names, x_axis, title):
    _ = plt.subplot(idx_chart)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    barlist = plt.bar(x_axis, classes_metric)
    plt.xticks(x_axis, class_names)

    # Think of a general way to actually change colors dynamically
    if len(classes_metric) == 3:
        barlist[0].set_color('b')
        barlist[1].set_color('g')
        barlist[2].set_color('r')

    plt.title(title)


if __name__ == '__main__':
    opt = parse_opts_metrics_plot()

    _ = plt.figure('Testing Metrics')

    input_csv = opt.input_csv
    classes_list = opt.classes_list

    with open(classes_list, 'r') as f:
        class_names = []
        for row in f:
            class_names.append(row[:-1])

    with open(input_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Did not found any good way to take the last live of a CSV. I know it's inefficient
        for row in csv_reader:
            if len(row) != 0 and row[0] == 'Mean_metrics_for_videos':
                mean_predicition_time = row[1]
                mean_accuracy = row[2]

                idx_metrics = 3

                classes_precision, idx_metrics = get_metric(class_names, idx_metrics, row)
                classes_recall, idx_metrics = get_metric(class_names, idx_metrics, row)
                classes_fscore, _ = get_metric(class_names, idx_metrics, row)

                length_chart = len(class_names)
                x_axis = np.arange(length_chart)

                build_plot(131, classes_precision, class_names, x_axis, "Classes Precision")
                build_plot(132, classes_recall, class_names, x_axis, "Classes Recall")
                build_plot(133, classes_fscore, class_names, x_axis, "Classes F-Score")

                plt.subplots_adjust(wspace=0.5, hspace=1)
                plt.show()
            else:
                continue
