import cv2 as cv
import imageio
import numpy as np
from IPython import display

# Useful variables
input_video_path = "/home/eugenio/Documenti/aim2.thesisactionrecognition/projects/action-detection/data/ActivityNet/Crawler/videos/volleyball_short.mp4"
save_npy_rgb_path = "/home/eugenio/Documenti/aim2.thesisactionrecognition/projects/kinetics-i3d/data/customVideoRGB"
save_npy_flow_path = "/home/eugenio/Documenti/aim2.thesisactionrecognition/projects/kinetics-i3d/data/customVideoFlow"


# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


# Adattato da https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/action_recognition_with_tf_hub.ipynb#scrollTo=USf0UvkYIlKo
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
    return np.array(frames) / 255.0  # Normalize in [0,1]


def scale(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    scale_factor = min_dim / 256
    max_dim = max(y, x)

    new_max_dim = max_dim / scale_factor

    if (y > x):
        return cv.resize(frame, (256, new_max_dim), interpolation=cv.INTER_LINEAR)
    else:
        return cv.resize(frame, (new_max_dim, 256), interpolation=cv.INTER_LINEAR)


# Implementazione più simile a ciò che dicono qui: https://github.com/deepmind/kinetics-i3d#sample-data-and-preprocessing
def load_video_rgb_custom(path, max_frames=0, resize=(224, 224)):
    cap = cv.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = scale(frame)
            frame = (frame / 255) * 2 - 1  # Normalize in [-1,1]
            frame = crop_center_square(frame)
            frame = cv.resize(frame, resize, interpolation=cv.INTER_LINEAR)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def load_video_flow_custom(path, resize=(224, 224)):
    cap = cv.VideoCapture(path)
    flow_frames = []

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

            flow_frames.append(curr_flow)

            previous_frame = next_frame
    finally:
        cap.release()

    return np.array(flow_frames)


if __name__ == '__main__':
    # 'Standard' implementation - RGB
    numPyArrayOfInputVideo = load_video_rgb(input_video_path)

    model_input = np.expand_dims(numPyArrayOfInputVideo, axis=0)  # Nel primo indice, c'è il batch_size

    print(model_input.shape)

    # Save .npy array
    np.save(save_npy_rgb_path, model_input)

if __name__ == '__main__':
    # 'Modified' implementation - RGB
    inputVideoRGB = load_video_rgb(input_video_path)

    model_input = np.expand_dims(numPyArrayOfInputVideo, axis=0)  # Nel primo indice, c'è il batch_size

    print(model_input.shape)

    # Save .npy array
    np.save(save_npy_rgb_path, model_input)


def animate(video):
    converted_video = np.clip(video * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_video, fps=30)
    with open('./animation.gif', 'rb') as f:
        display.display(display.Image(data=f.read(), height=300))


animate(numPyArrayOfInputVideo)

if __name__ == '__main__':
    # Implementation - Flow
    inputVideoFlow = load_video_flow_custom(input_video_path)

    model_input = np.expand_dims(inputVideoFlow, axis=0)  # Nel primo indice, c'è il batch_size

    print(model_input.shape)

    # Save .npy array
    np.save(save_npy_flow_path, model_input)
