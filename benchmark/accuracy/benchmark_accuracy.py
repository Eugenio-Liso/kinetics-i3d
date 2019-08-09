import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], "../../logging_utils"))

import json
from opts_accuracy import parse_opts_benchmark
from logger_factory import getBasicLogger

logger = getBasicLogger(os.path.basename(__file__))

if __name__ == '__main__':

    opt = parse_opts_benchmark()
    output_json_predictions = opt.output
    ground_truth_labels = opt.labeled_videos
    confidence_threshold = opt.confidence_threshold

    logger.info("Input json of predictions: {}".format(output_json_predictions))
    logger.info("Input video labels: {}".format(ground_truth_labels))

    with open(output_json_predictions, 'r') as f:
        json_predictions = json.load(f)

    with open(ground_truth_labels, 'r') as f:
        labels = json.load(f)

    assert len(json_predictions) == len(labels), \
        "Labels size must be equal to output json size, i.e. one label per video prediction "

    for prediction_single_video in json_predictions:
        video_name = prediction_single_video['video']
        clips = prediction_single_video['clips']

        if video_name not in labels:
            raise ValueError("Video {} not found in ground truth labels".format(video_name))

        ground_truth = labels[video_name].casefold()

        total_predictions = len(clips)  # num of predictions done
        correct_predictions = 0
        true_positives = 0

        for single_prediction in clips:
            predicted_label = single_prediction['label'].casefold()  # for ignore case comparisons

            if predicted_label == ground_truth:
                correct_predictions += 1

                class_scores = single_prediction['scores']
                if max(class_scores) >= confidence_threshold:
                    true_positives += 1

        final_accuracy = (correct_predictions / total_predictions) * 100
        precision = (true_positives / total_predictions) * 100

        logger.info("Accuracy for video: {} is {}%".format(video_name, final_accuracy))
        logger.info("Precision for video: {} is {}%".format(video_name, precision))
