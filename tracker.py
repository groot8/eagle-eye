# USAGE
# python multi_object_tracking_slow.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# import the necessary packages
from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2
import math

stream_num = 0


def intersectionOverUnion(bouns1, bouns2):
    (startX, startY, endX, endY) = (
        max(bouns1[0], bouns2[0]),
        max(bouns1[1], bouns2[1]),
        min(bouns1[2], bouns2[2]),
        min(bouns1[3], bouns2[3])
    )
    if startX > endX or startY > endY:
        return 0
    intersection = (endX - startX)*(endY - startY)
    area1 = (bouns1[2] - bouns1[0])*(bouns1[3] - bouns1[1])
    area2 = (bouns2[2] - bouns2[0])*(bouns2[3] - bouns2[1])
    if bouns1[0] >= bouns2[0] and bouns1[1] >= bouns2[1] and bouns1[2] <= bouns2[2] and bouns1[3] <= bouns2[3]:
        return 1
    if bouns2[0] >= bouns1[0] and bouns2[1] >= bouns1[1] and bouns2[2] <= bouns1[2] and bouns2[3] <= bouns1[3]:
        return 1
    return intersection / (area1 + area2 - intersection)


class avatar():
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    maxintersectionOverUnion = 0.2
    detection_interval = 1

    def __init__(self, prototxt, model, video, output, confidence, calibration_file, points_color):
        # load our serialized model from disk
        # print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        # initialize the video stream and output video writer
        # print("[INFO] starting video stream...")
        self.vs = cv2.VideoCapture(video)
        # start the frames per second throughput estimator
        self.fps = FPS().start()
        self.output = output
        self.confidence = confidence
        global stream_num
        stream_num += 1
        self.stream_num = stream_num
        # initialize the list of object trackers and corresponding class
        # labels
        self.trackers = []
        self.labels = []
        # intialize id
        self.id = 1
        self.frame = None
        # counter to reset trackers
        self.detection_counter = 0
        self.writer = None
        self.calibration_file = calibration_file
        self.points_color = points_color

    def __del__(self):
        # stop the timer and display FPS information
        self.fps.stop()
        #print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vs.release()

    def get_top_view_of_point(self, point):
        result = np.matmul(self.calibration_file/self.calibration_file[2][2], np.array([point[0], point[1], 0]))
        print(result)
        return (int(result[0]), int(result[1]))

    def get_top_view(self, frame):
        return cv2.warpPerspective(frame, self.calibration_file/self.calibration_file[2][2], (700, 500))

    def detect_people(self, img):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections

    def validate_trackers(self):
        index = 0
        for (t, l) in zip(self.trackers, self.labels):
            pos = t.get_position()
            # unpack the position object
            startX, startY = int(pos.left()), int(pos.top())
            endX, endY = int(pos.right()), int(pos.bottom())
            if self.search_and_match(index) != -1:
                self.trackers.remove(t)
                self.labels.remove(l)
            index += 1

    def search_and_match(self, tracker_index):
        pos = self.trackers[tracker_index].get_position()
        # unpack the position object
        startX, startY = int(pos.left()), int(pos.top())
        endX, endY = int(pos.right()), int(pos.bottom())
        boundries = (startX, startY, endX, endY)
        maxintersectionOverUnion = 0
        target = -1
        index = -1
        for t in self.trackers:
            index += 1
            if index <= tracker_index:
                continue
            pos = t.get_position()
            # unpack the position object
            startX, startY = int(pos.left()), int(pos.top())
            endX, endY = int(pos.right()), int(pos.bottom())
            temp = intersectionOverUnion(
                boundries, (startX, startY, endX, endY))
            if temp > maxintersectionOverUnion:
                maxintersectionOverUnion = temp
                target = index
        #print(minError)
        if maxintersectionOverUnion > self.maxintersectionOverUnion:
            return target
        else:
            return -1

    def forward(self):
        points = []
        # grab the next frame from the video file
        (grabbed, frame) = self.vs.read()
        intial_width = frame.shape[1]
        # # Create a blank 300x300 black image
        board = np.zeros(
            (frame.shape[0], frame.shape[1], frame.shape[2]), np.uint8)
        # # Fill board with red color(set each pixel to red)
        board[:] = (0, 0, 0)
        board = frame
        # increment detection counter and reset trackers if reach max
        current_FPS = self.vs.get(cv2.CAP_PROP_FPS)
        self.detection_counter += 1 / current_FPS
        # print(self.detection_counter)
        if self.detection_counter >= self.detection_interval:
            self.detection_counter = 0

        # check to see if we have reached the end of the video file
        if frame is None:
            return

        # resize the frame for faster processing and then convert the
        # frame from BGR to RGB ordering (dlib needs RGB ordering)
        frame = imutils.resize(frame, width=600)
        board = imutils.resize(board, width=600)
        # board = imutils.resize(board, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if self.output is not None and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(self.output, fourcc, 30,
                                          (frame.shape[1], frame.shape[0]), True)

        # if there are no object trackers we first need to detect objects
        # and then create a tracker for each object
        if self.detection_counter == 0:

            self.trackers = []
            self.labels = []
            detections = self.detect_people(frame)

            # loop over the detections
            (h, w) = frame.shape[:2]
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > self.confidence:
                    #print("===",confidence)
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])
                    label = self.CLASSES[idx] + str(self.id)

                    # if the class label is not a person, ignore it
                    if self.CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # if box match another one in range x continue
                    # center = (startX + (endX - startX) / 2, startY + (endY - startY) / 2)
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and start the correlation tracker
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    t.start_track(rgb, rect)

                    # update our set of trackers and corresponding class
                    # labels
                    self.labels.append(label)
                    self.trackers.append(t)

                    # grab the corresponding class label for the detection
                    # and draw the bounding box
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            self.validate_trackers()

        # otherwise, we've already performed detection so let's track
        # multiple objects
        else:
            # loop over each of the self.trackers
            for (t, l) in zip(self.trackers, self.labels):
                # update the tracker and grab the position of the tracked
                # object
                t.update(rgb)
                #print(str(t))
                pos = t.get_position()
                # unpack the position object
                startX, startY = int(pos.left()), int(pos.top())
                endX, endY = int(pos.right()), int(pos.bottom())
                # draw the bounding box from the correlation object tracker
                # if confidencex >= 0.5:
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                cv2.putText(frame, l, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                cv2.circle(frame, (int((startX+endX)/2), endY),
                           10, self.points_color, -1)
                points.append((int(((startX+endX)/2)*(intial_width/600)), int(endY * (intial_width/600))))

        # check to see if we should write the frame to disk
        if self.writer is not None:
            self.writer.write(frame)
        board = imutils.resize(board, width=intial_width)
        # run one of those examples
        # example 1
        # board = self.get_top_view(board)
        # for point in points:
        #     cv2.circle(board, self.get_top_view_of_point(point), 10, self.points_color, -1)
        # example 2
        for point in points:
            cv2.circle(board, point, 10, self.points_color, -1)
        board = self.get_top_view(board)
        
        # show the output frame
        cv2.imshow("Frame(cal)"+str(self.stream_num),
                   imutils.resize(board, width=600))
        # cv2.imshow("Frame"+str(self.stream_num), frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            return

        # update the FPS counter
        self.fps.update()
