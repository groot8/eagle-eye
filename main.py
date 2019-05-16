import argparse
from tracker import avatar,d_ps,set_ground_truth_file_path, showId, hideId, getIds, togglePause
import numpy as np

# calibration_files = [
#     np.array([[-1.6688907435,-6.9502305710,	940.69592392565],[1.1984806153,	-10.7495778320,	868.29873467315],[0.0004069210,	-0.0209324057,	0.42949125235]]),
#     np.array([[0.6174778372,	-0.4836875683,	147.00510919005],[0.5798503075,	3.8204849039,	-386.096405131],[0.0000000001,	0.0077222239,	-0.01593391935]]),
#     np.array([[-0.2717592338,1.0286363982,-17.6643219215],[-0.1373600672,-0.3326731339,161.0109069274],[0.0000600052,0.0030858398,-0.04195162855]]),
#     np.array([[-0.3286861858,0.1142963200,130.25528281945],[0.1809954834,-0.2059386455,125.0260427323],[0.0000693641,0.0040168154,-0.08284534995]])
# ]


calibration_files = [
    np.array([[0.176138,	0.647589,	-63.412272],[-0.180912,	0.622446,	-0.125533],[-0.000002,	0.001756,	0.102316]]),
    np.array([[0.177291,	0.004724,	31.224545],[0.169895,	0.661935,	-79.781865],[-0.000028,	0.001888,	0.054634]]),
    np.array([[-0.104843,      0.099275,  50.734500],[0.107082,   0.102216,  7.822562],[-0.000054,  0.001922,   -0.068053]]),
    np.array([[-0.142865,	0.553150,	-17.395045],[-0.125726,	0.039770,	75.937144],[-0.000011,	0.001780,	0.015675]])
]

# pos => from ground truth file
# 40, 99 => grid_width, drid_height
# 0, 38.48 => tv_origin_x, tv_origin_y
# 155, 381 => tv_width, tv_height
def grid_to_tv(pos, grid_width, grid_height, tv_origin_x, tv_origin_y, tv_width, tv_height):     
    tv_x = ( (pos % grid_width) + 0.5 ) * (tv_width / grid_width) + tv_origin_x
    tv_y = ( (pos / grid_width) + 0.5 ) * (tv_height / grid_height) + tv_origin_y
    return (tv_x, tv_y)

points_colors = [
    (0,0,255),
    (0,255,255),
    (255,255,255),
    (100,100,100)
]

ground_truth_file_path = 'output/cal_ground_truth_list.txt'

streams = []

def main(imshow):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--prototxt", required=True,
                    # help="path to Caffe 'deploy' prototxt file")
    # ap.add_argument("-m", "--model", required=True,
                    # help="path to Caffe pre-trained model")
    # ap.add_argument("-v", "--video", required=True,
                    # help="path to input video file")
    # ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    # help="minimum probability to filter weak detections")
    # args = vars(ap.parse_args())
    # args = {'video':'../GP_Data/terrace1-c0.avi,../GP_Data/terrace1-c1.avi'}
    args = {'video':'dataset/6p-c0.avi,dataset/6p-c1.avi,dataset/6p-c2.avi,dataset/6p-c3.avi'}

    for (src, calibration_file, points_color) in zip(args['video'].split(','), calibration_files, points_colors):
        streams.append(avatar(src, False, imshow, calibration_file, points_color))
    
    f= open(ground_truth_file_path,"w+")
    f.close()
    set_ground_truth_file_path(ground_truth_file_path)

    while True:
        for stream in streams:
            stream.forward()
        avatar.reset_list_points(imshow)
    
    # print(d_ps)

from flask import Flask, render_template, Response, request, jsonify
import cv2
from time import time, sleep

app = Flask(__name__)


def gen(si):
    while True:
        # Capture frame-by-frame
        try:
            yield streams[si].getFrame()
        except:
            sleep(1)
            print("[stream#"+si+"]pause")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle-pause')
def toggle_pause():
    togglePause()
    return '', 204

@app.route('/ids')
def listIds():
    return jsonify(getIds())

@app.route('/ids/show')
def show():
    showId(int(request.args.get('id')))
    return '', 204

@app.route('/ids/hide')
def hide():
    hideId(int(request.args.get('id')))
    return '', 204

@app.route('/run')
def run():
    main(eval(request.args.get('imshow')))
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(gen(int(request.args.get('si'))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


