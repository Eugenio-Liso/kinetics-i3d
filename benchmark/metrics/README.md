# Metrics 

## Generate CSV containing the metrics

```python
python benchmark/metrics/benchmark_metrics.py \
--output <output json from testing> \
--labeled_videos <labels_file> \
--classes_list <class_list file> \
--output_mean_times <output_times from testing>
```

The classes list file is a simple file containing a set of labels, one per row.

The labels file contains a JSON with the following structure:

```bash
{
  "name_video1": "label1"
  "name_video2": "label2"
  ...
}
```

The video name must match the one you used in the prediction phase.

The label must be a valid label (i.e. one that can be predicted).

# Plotting the results

```python
python benchmark/metrics/plot_metrics.py \
--input_csv <csv produced by previous script> \
--classes_list <class_list file>
```