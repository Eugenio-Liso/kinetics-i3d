# Accuracy 

## Log file

```python
python benchmark/accuracy/benchmark_accuracy.py --output <output_json_with_predictions> --labeled_videos <file_containing_labels> [--confidence_threshold 10]
```

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

The confidence threshold is used to output a more solid metric on how the model is performing. 
If a score is equal or higher than the specified threshold (it has a default value specified in `opts_accuracy.py`),
then it is considered as a true positive, otherwise it is a false positive.