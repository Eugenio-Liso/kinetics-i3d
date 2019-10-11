import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], "../../logging_utils"))

import json
from opts_metrics import parse_opts_benchmark
from logger_factory import getBasicLogger
import csv
from sklearn.metrics import precision_recall_fscore_support
from metrics_aggregator import SimpleAverage

logger = getBasicLogger(os.path.basename(__file__))


def create_column_metric_csv_header(column_prefix, class_names, header):
    for target_class in class_names:
        header.append(f"{column_prefix}_{target_class}")


def create_column_metric_csv_content(contents, csv_row):
    for result in contents:
        csv_row.append(result)


if __name__ == '__main__':

    opt = parse_opts_benchmark()
    output_json_predictions = opt.output
    ground_truth_labels = opt.labeled_videos
    classes_list = opt.classes_list
    output_csv = opt.output_csv
    output_mean_times = opt.output_mean_times

    logger.info("Input json of predictions: {}".format(output_json_predictions))
    logger.info("Input video labels: {}".format(ground_truth_labels))

    with open(output_json_predictions, 'r') as f:
        json_predictions = json.load(f)

    with open(ground_truth_labels, 'r') as f:
        labels = json.load(f)

    with open(output_mean_times, 'r') as f:
        output_mean_times_json = json.load(f)

    with open(classes_list, 'r') as f:
        class_names = []
        for row in f:
            class_names.append(row[:-1])

    if os.path.exists(output_csv):
        os.remove(output_csv)

    with open(output_csv, 'w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ["VIDEO_NAME", "MEAN_PREDICTION_TIME", "ACCURACY"]

        create_column_metric_csv_header("PRECISION", class_names, header)
        create_column_metric_csv_header("RECALL", class_names, header)
        create_column_metric_csv_header("F-SCORE", class_names, header)

        writer.writerow(header)

        assert len(json_predictions) == len(labels), \
            "Labels size must be equal to output json size, i.e. one label per video prediction "

        average_accuracy = SimpleAverage()
        average_mean_time = SimpleAverage()
        predicted_labels = []
        target_labels = []

        for prediction_single_video in json_predictions:
            video_name = prediction_single_video['video']
            clips = prediction_single_video['clips']

            if video_name not in labels:
                raise ValueError("Video {} not found in ground truth labels".format(video_name))

            ground_truth = labels[video_name].casefold()

            total_predictions = len(clips)  # num of predictions done
            correct_predictions = 0

            predicted_labels_for_single_video = []

            for single_prediction in clips:
                predicted_label = single_prediction['label'].casefold()  # for ignore case comparisons

                if predicted_label == ground_truth:
                    correct_predictions += 1

                predicted_labels_for_single_video.append(predicted_label)

            ground_truth_list_repeated = [ground_truth] * len(predicted_labels_for_single_video)

            precision, recall, fscore, _ = \
                precision_recall_fscore_support(ground_truth_list_repeated,
                                                predicted_labels_for_single_video,
                                                labels=class_names)

            final_accuracy = (correct_predictions / total_predictions)
            mean_time_for_video = output_mean_times_json[video_name]

            logger.info("Mean Prediction Time for video: {} is {}".format(video_name, mean_time_for_video))
            logger.info("Accuracy for video: {} is {}%".format(video_name, final_accuracy))
            logger.info("Precision for video: {} is {}".format(video_name, precision))
            logger.info("Recall for video: {} is {}".format(video_name, recall))
            logger.info("F-Score for video: {} is {}".format(video_name, fscore))

            average_accuracy.update(correct_predictions, total_predictions)
            average_mean_time.update(mean_time_for_video)
            predicted_labels.extend(predicted_labels_for_single_video)
            target_labels.extend(ground_truth_list_repeated)

            csv_row = [video_name, output_mean_times_json[video_name], final_accuracy]

            create_column_metric_csv_content(precision, csv_row)
            create_column_metric_csv_content(recall, csv_row)
            create_column_metric_csv_content(fscore, csv_row)

            writer.writerow(csv_row)

        writer.writerow([])

        avg_accuracy = average_accuracy.average()
        avg_mean_time = average_mean_time.average()

        precision, recall, fscore, _ = \
            precision_recall_fscore_support(target_labels,
                                            predicted_labels,
                                            labels=class_names)

        final_row = ['Mean_metrics_for_videos', avg_mean_time, avg_accuracy]

        create_column_metric_csv_content(precision, final_row)
        create_column_metric_csv_content(recall, final_row)
        create_column_metric_csv_content(fscore, final_row)

        writer.writerow(final_row)
