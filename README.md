# This is a forked repo from: https://github.com/deepmind/kinetics-i3d

- Setup environment
```bash
conda install tensorflow-gpu=1.13.1
conda install -c hcc dm-sonnet=1.32
conda install -c conda-forge tensorflow-probability=0.6.0
conda install scikit-learn=0.21.2 matplotlib=3.1.0 seaborn=0.9.0

# For preprocessing
conda install opencv=3.4.4 -c conda-forge # 4.0.1 does not have DualTVL1OpticalFlow_create

# Additional dependencies that will not be needed in general
#conda install imageio=2.5.0 ipython=7.7.0
```

# How to preprocess videos and run inference
The main ideas behind the implementation comes from here: https://github.com/deepmind/kinetics-i3d/pull/5/commits/f1fa01a332179e82cd655e7cd2f2f0c1c04f0c74.

However, there is also another implementation that comes from [Colab](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/action_recognition_with_tf_hub.ipynb#scrollTo=USf0UvkYIlKo
). By default, the implementation that mostly 
follows the author advices should be chosen (i.e. the one already used right now).
See the preprocessing/preprocess.py file.

For technical reasons, the first frame in RGB videos is removed. The reason is that, why a video could have N 
frames, the optical flow of it has N-1 frames. So we have to match the array dimensions.

You can choose to run the preprocessing with videos or frames as inputs. If you are giving frames as inputs, the directory 
structure should be as follows:

```bash
frames_dir
├── class_1
│   ├── video_id1
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   ├── image_00003.jpg
│   │   ├── image_00004.jpg
│   │   ├── image_00005.jpg
│   │   ├── image_00006.jpg
│   │   ├── image_00007.jpg
│   ├── video_id2
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   ├── image_00003.jpg
...
├── class_2
└── class_N
```

After the preprocessing, run the script `evaluate_sample.py` or `custom_evaluation.sh` just like in the section below.
# Run without preprocessing

See `custom_evaluation.sh` to know the script inputs. Sample call:

```bash
bash custom_evaluation.sh joint true data/rgb data/flow
```

Otherwise, just use the `evaluate_sample.py` script. Check the parameters of that script (prefixed by `tf.flags`).
