import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], "../../logging_utils"))

import matplotlib.pyplot as plt
import json
from opts_pred_times import parse_opts_benchmark
from logger_factory import getBasicLogger
import benchmark_times_plots as bp

logger = getBasicLogger(os.path.basename(__file__))

if __name__ == '__main__':

    opt = parse_opts_benchmark()
    output_json_exec_times_path = opt.output_times

    logger.info("Input json of execution times: {}".format(output_json_exec_times_path))

    with open(output_json_exec_times_path, 'r') as f:
        json_exec_times = json.load(f)

    for exec_time_single_video in json_exec_times:
        bp.extract_data(exec_time_single_video, True)

    bp.display_results()
