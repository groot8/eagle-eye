import argparse
from tracker import avatar
import numpy as np

calibration_files = [
    np.array([[-0.211332, -0.405226, 70.781223], [-0.019746, -
                                                  1.564936, 226.377280], [-0.000025, -0.001961, 0.160791]]),
    np.array([[0.000745, 0.350335, -98.376103], [-0.164871, -
                                                 0.390422, 54.081423], [0.000021, -0.001668, 0.111075]])
]

points_colors = [
    (0,0,255),
    (0,255,255),
]


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    streams = []

    for (src, calibration_file, points_color) in zip(args['video'].split(','), calibration_files, points_colors):
        print(src)
        streams.append(avatar(args['prototxt'], args['model'],
                              src, None, args['confidence'], calibration_file, points_color))

    while True:
        for stream in streams:
            stream.forward()


main()
