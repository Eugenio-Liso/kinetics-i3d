#!/usr/bin/env bash

type_prediction=$1
imagenet_pretrained=$2
input_video_rgb_or_flow=$3
input_video_flow=$4


if [[ -z ${input_video_rgb_or_flow} && -z ${input_video_flow} ]]; then
    echo "No input path has been set correctly. Exiting..."
elif [[ -z ${input_video_flow} ]]; then # Only RGB
    python evaluate_sample.py --imagenet_pretrained ${imagenet_pretrained} --eval_type ${type_prediction} --input_video_rgb "${input_video_rgb_or_flow}" --input_video_flow ""
else
    #RGB + FLOW
    python evaluate_sample.py --imagenet_pretrained ${imagenet_pretrained} --eval_type ${type_prediction} --input_video_rgb "${input_video_rgb_or_flow}" --input_video_flow "${input_video_flow}"
fi
#python evaluate_sample.py --imagenet_pretrained true --eval_type rgb --input_video_rgb data/customVideoRGB.npy > out/customVideo_rgb_logs.txt
#python evaluate_sample.py --imagenet_pretrained true --eval_type flow --input_video_flow data/customVideoFlow.npy > out/customVideo_flow_logs.txt
#python evaluate_sample.py --imagenet_pretrained true --eval_type joint --input_video_rgb data/customVideoRGB.npy --input_video_flow data/customVideoFlow.npy > out/customVideo_joint_logs.txt

# Sample test from devs
# python evaluate_sample.py --imagenet_pretrained ${imagenet_pretrained} --eval_type ${type_prediction} --input_video_rgb "${input_video_rgb_or_flow}" --input_video_flow data/v_CricketShot_g04_c01_flow.npy > out/sampleTestVideo_joint_logs.txt
