#!/usr/bin/env bash
type_prediction=$1
imagenet_pretrained=$2
input_folder_rgb=$3
input_folder_flow=$4
if [[ -z ${input_folder_rgb} && -z ${input_folder_flow} ]]; then
    echo "No input path has been set correctly. Exiting..."
elif [[ ${type_prediction} == 'rgb' ]]; then # Only RGB
    python evaluate_sample.py --imagenet_pretrained "${imagenet_pretrained}" --eval_type "${type_prediction}" --input_folder_rgb "${input_folder_rgb}"
elif [[ ${type_prediction} == 'flow' ]]; then
    python evaluate_sample.py --imagenet_pretrained "${imagenet_pretrained}" --eval_type "${type_prediction}" --input_folder_flow "${input_folder_rgb}"
elif [[ ${type_prediction} == 'joint' ]]; then
    #RGB + FLOW
    python evaluate_sample.py --imagenet_pretrained "${imagenet_pretrained}" --eval_type "${type_prediction}" --input_folder_rgb "${input_folder_rgb}" --input_folder_flow "${input_folder_flow}"
else
  echo "Type of prediction: ${type_prediction} is not supported"
fi
#python evaluate_sample.py --imagenet_pretrained true --eval_type rgb --input_video_rgb data/customVideoRGB.npy > out/customVideo_rgb_logs.txt
#python evaluate_sample.py --imagenet_pretrained true --eval_type flow --input_video_flow data/customVideoFlow.npy > out/customVideo_flow_logs.txt
#python evaluate_sample.py --imagenet_pretrained true --eval_type joint --input_video_rgb data/customVideoRGB.npy --input_video_flow data/customVideoFlow.npy > out/customVideo_joint_logs.txt
# Sample test from devs
# python evaluate_sample.py --imagenet_pretrained ${imagenet_pretrained} --eval_type ${type_prediction} --input_video_rgb "${input_video_rgb_or_flow}" --input_video_flow data/CricketShot.npy > out/sampleTestVideo_joint_logs.txt