import argparse
from edited_tracker import avatar
def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    stream = avatar(args['prototxt'],args['model'], args['video'], args['output'], args['confidence'])
    while True:
        stream.forward()
        

main()