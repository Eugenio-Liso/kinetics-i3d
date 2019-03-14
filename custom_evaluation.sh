#python evaluate_sample.py --imagenet_pretrained true --eval_type rgb --input_video_rgb data/customVideoRGB.npy > out/customVideo_rgb_logs.txt
#python evaluate_sample.py --imagenet_pretrained true --eval_type flow --input_video_flow data/customVideoFlow.npy > out/customVideo_flow_logs.txt
#python evaluate_sample.py --imagenet_pretrained true --eval_type joint --input_video_rgb data/customVideoRGB.npy --input_video_flow data/customVideoFlow.npy > out/customVideo_joint_logs.txt

# Sample test from devs
python evaluate_sample.py --imagenet_pretrained true --eval_type joint --input_video_rgb data/v_CricketShot_g04_c01_rgb.npy --input_video_flow data/v_CricketShot_g04_c01_flow.npy > out/sampleTestVideo_joint_logs.txt
