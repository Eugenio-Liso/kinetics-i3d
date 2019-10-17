import cv2 as cv
import imageio
import numpy as np
from IPython import display
import os
import subprocess
from opts_preprocess import parse_opts_prediction

# Should not be changed
imagenet_pretrained = "true"


# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


# "Unofficial" implementation from
# https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/action_recognition_with_tf_hub.ipynb#scrollTo=USf0UvkYIlKo
def load_video_rgb(path, max_frames=0, resize=(224, 224)):
    cap = cv.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv.resize(frame, resize, interpolation=cv.INTER_LINEAR)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    del frames[0]  # Removes first element, to do in preprocessing due to flow having size N - 1
    return np.array(frames) / 255.0  # Normalize in [0,1]


def scale(frame):
    # Analyze if this can help simplify
    # function
    # calculateAspectRatioFit(srcWidth, srcHeight, maxWidth, maxHeight)
    # {
    #
    #     var
    # ratio = Math.min(maxWidth / srcWidth, maxHeight / srcHeight);
    #
    # return {width: srcWidth * ratio, height: srcHeight * ratio};
    # }
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    scale_factor = min_dim / 256
    max_dim = max(y, x)

    new_max_dim = int(max_dim / scale_factor)

    if (y > x):
        return cv.resize(frame, (256, new_max_dim), interpolation=cv.INTER_LINEAR)
    else:
        return cv.resize(frame, (new_max_dim, 256), interpolation=cv.INTER_LINEAR)


### A more official implementation from
# https://github.com/deepmind/kinetics-i3d#sample-data-and-preprocessing
def load_video_rgb_custom(path, use_frames, max_frames=0, resize=(224, 224)):
    if not use_frames:
        frames = []
        cap = cv.VideoCapture(path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = apply_preprocess_rgb(frame, resize)
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        del frames[0]  # Removes first element, to do in preprocessing due to flow having size N - 1
        return np.array(frames)
    else:
        total_frames = [cv.imread(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Taking all frames. Ignoring max_frames
        frames_with_preprocess = [apply_preprocess_rgb(frame, resize) for frame in total_frames]

        del frames_with_preprocess[0]  # Removes first element, to do in preprocessing due to flow having size N - 1
        return np.array(frames_with_preprocess)


def apply_preprocess_rgb(frame, resize):
    frame = scale(frame)
    frame = (frame / 255) * 2 - 1  # Normalize in [-1,1]
    frame = crop_center_square(frame)
    frame = cv.resize(frame, resize, interpolation=cv.INTER_LINEAR)
    frame = frame[:, :, [2, 1, 0]]
    return frame


def load_video_flow_custom(path, use_frames, resize=(224, 224)):
    flow_frames = []

    if not use_frames:
        cap = cv.VideoCapture(path)

        _, first_raw_frame = cap.read()
        previous_frame = cv.cvtColor(first_raw_frame, cv.COLOR_BGR2GRAY)

        # Frame is composed as frame[WIDHT][HEIGHT] and return an RGB array
        # Return ret = true if it has successfully extracted the frame, along with the actual frame
        try:
            while True:
                ret, next_raw_frame = cap.read()
                if not ret:
                    break

                next_frame = cv.cvtColor(next_raw_frame, cv.COLOR_BGR2GRAY)

                curr_flow = apply_preprocess_optical_flow(next_frame, previous_frame, resize)

                flow_frames.append(curr_flow)

                previous_frame = next_frame
        finally:
            cap.release()

        return np.array(flow_frames)
    else:
        total_frames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        if len(total_frames) < 2:
            raise Exception(f"The video in path {path} should have at least two frames")

        previous_frame = cv.cvtColor(cv.imread(total_frames[0]), cv.COLOR_BGR2GRAY)

        for frame_path in total_frames[1:]:
            next_frame = cv.cvtColor(cv.imread(frame_path), cv.COLOR_BGR2GRAY)

            curr_flow = apply_preprocess_optical_flow(next_frame, previous_frame, resize)

            flow_frames.append(curr_flow)

            previous_frame = next_frame

        return np.array(flow_frames)


def apply_preprocess_optical_flow(next_frame, previous_frame, resize):
    # TLV-1
    # optical_flow_algorithm = cv.DualTVL1OpticalFlow_create()
    # curr_flow = optical_flow_algorithm.calc(previous_frame, next_frame, None)

    # Farneback
    curr_flow = cv.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # truncate [-20, 20]
    curr_flow[curr_flow > 20] = 20
    curr_flow[curr_flow < -20] = -20

    # scale to [-1, 1]
    max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
    curr_flow = curr_flow / max_val(curr_flow)

    # Final cropping and resize
    curr_flow = crop_center_square(curr_flow)
    curr_flow = cv.resize(curr_flow, resize, interpolation=cv.INTER_LINEAR)
    return curr_flow


# if __name__ == '__main__':
#     # 'Standard' implementation - RGB
#     numPyArrayOfInputVideo = load_video_rgb(input_video_path)
#
#     model_input = np.expand_dims(numPyArrayOfInputVideo, axis=0)  # Nel primo indice, c'è il batch_size
#
#     print(model_input.shape)
#
#     # Save .npy array
#     np.save(save_npy_rgb_path, model_input)

# if __name__ == '__main__':
#     # 'Modified' implementation - RGB
#     inputVideoRGB = load_video_rgb_custom(input_video_path)
#
#     model_input = np.expand_dims(inputVideoRGB, axis=0)  # Nel primo indice, c'è il batch_size
#
#     print(model_input.shape)
#
#     # Save .npy array
#     np.save(save_npy_rgb_path, model_input)


# def animate(video):
#     converted_video = np.clip(video * 255, 0, 255).astype(np.uint8)
#     imageio.mimsave('./animation.gif', converted_video, fps=30)
#     with open('./animation.gif', 'rb') as f:
#         display.display(display.Image(data=f.read(), height=300))


# if __name__ == '__main__':
#     numPyArrayOfInputVideo = np.load(save_npy_rgb_path)
#     animate(numPyArrayOfInputVideo)

if __name__ == '__main__':
    opt = parse_opts_prediction()

    input_video_or_frames_path = opt.input_video_or_frames_path
    save_npy_rgb_path = opt.save_npy_rgb_path
    save_npy_flow_path = opt.save_npy_flow_path
    type_prediction = opt.type_prediction
    use_frames = opt.use_frames

    # Implementation - Flow + RGB
    if not use_frames:
        for input_video in os.listdir(input_video_or_frames_path):
            complete_video_path = os.path.join(input_video_or_frames_path, input_video)

            output_flow_npy = os.path.join(save_npy_flow_path, input_video.split(".")[0])
            output_rgb_npy = os.path.join(save_npy_rgb_path, input_video.split(".")[0])

            print("Output flow npy array: {}.npy".format(output_flow_npy))
            print("Output rgb npy array: {}.npy".format(output_rgb_npy))

            inputVideoFlow = load_video_flow_custom(complete_video_path, use_frames)
            inputVideoRGB = load_video_rgb_custom(complete_video_path, use_frames)

            model_input_flow = np.expand_dims(inputVideoFlow, axis=0)  # Nel primo indice, c'è il batch_size
            model_input_rgb = np.expand_dims(inputVideoRGB, axis=0)  # Nel primo indice, c'è il batch_size

            # Save .npy array
            np.save(output_flow_npy, model_input_flow)
            np.save(output_rgb_npy, model_input_rgb)
    else:
        for target_class in os.listdir(input_video_or_frames_path):
            target_input_path = os.path.join(input_video_or_frames_path, target_class)
            for input_video in os.listdir(target_input_path):
                video_id_path = os.path.join(target_input_path, input_video)

                output_flow_npy = os.path.join(save_npy_flow_path, input_video)
                output_rgb_npy = os.path.join(save_npy_rgb_path, input_video)

                print("Output flow npy array: {}.npy".format(output_flow_npy))
                print("Output rgb npy array: {}.npy".format(output_rgb_npy))

                inputVideoRGB = load_video_rgb_custom(video_id_path, use_frames)
                inputVideoFlow = load_video_flow_custom(video_id_path, use_frames)

                model_input_flow = np.expand_dims(inputVideoFlow, axis=0)  # Nel primo indice, c'è il batch_size
                model_input_rgb = np.expand_dims(inputVideoRGB, axis=0)  # Nel primo indice, c'è il batch_size

                # Save .npy array
                np.save(output_flow_npy, model_input_flow)
                np.save(output_rgb_npy, model_input_rgb)

    print("Preprocessing completed.")
    # dirname = os.path.dirname(__file__)
    # scriptname = os.path.join(dirname, '../custom_evaluation.sh')

    # Calling main python script
    # Change here the arguments according on what you want

    # RGB + FLOW
    # subprocess.call(
    #     [scriptname, type_prediction.lower(), imagenet_pretrained, save_npy_rgb_path, save_npy_flow_path, use_frames])

    # Only RGB
    # subprocess.call([scriptname, "rgb", imagenet_pretrained, save_npy_rgb_path])
