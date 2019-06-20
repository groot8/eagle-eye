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
from  yolo3.test import *
from time import sleep
from lab import *

def set_ground_truth_file_path(path):
    global ground_truth_file_path
    ground_truth_file_path = path

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

ids = [] #this would not be overwirtten but updated
last_id = 0

def get_fbs():
    return Avatar.fbs

def togglePause():
    Avatar.pause = not Avatar.pause

def showId(i):
    global ids
    for id in ids:
        if id[1] == i:
            id[3] = False
            
def hideId(i):
    global ids
    for id in ids:
        if id[1] == i:
            id[3] = True

def getIds():
    global ids
    res = []
    for id in ids:
        res.append([id[1],id[3]])
    return res

class Avatar():
    # initialize the list of class labels MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    streams_count = 0

    pause = False

    fbs = 0

    d_points = []

    def learn():
        global shape # has shape of original frame
        board = np.zeros(shape, np.uint8)
        board[:] = (0, 0, 0)
        Person.updateIds(Avatar.d_points)
        Person.imDrawPersons(board)
        cv2.imshow("Board", board)
        Avatar.d_points = []

    def getFrame(self):
        return self.frames.pop(0)

    def __init__(self, video, save_output, calibration_file, points_color, start_time = 0, end_time = float("inf")):
        self.frames = []
        # load our serialized model from disk
        # print("[INFO] loading model...")
        # self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        # initialize the video stream and output video writer
        # print("[INFO] starting video stream...")
        self.vs = cv2.VideoCapture(video)
        # start the frames per second throughput estimator
        self.fps = FPS().start()
        self.save_output = save_output
        self.s_i = Avatar.streams_count # assigning the stream index
        Avatar.streams_count += 1
        # initialize the list of object trackers and corresponding class
        # labels
        self.trackers = []
        self.labels = []
        # intialize id
        self.frame = None
        # counter to reset trackers
        self.detection_counter = 0
        self.writer = None
        self.calibration_file = calibration_file
        self.points_color = points_color
        # start time and end time in sec
        Avatar.fbs = self.vs.get(cv2.CAP_PROP_FPS)
        self.frames_counter = 0 
        # to start stream from specific time
        while self.frames_counter / Avatar.fbs < start_time and start_time != 0:
            self.vs.read()
            self.frames_counter += 1
        self.max_frames_counter = end_time * Avatar.fbs

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
        # print(point)
        result = np.matmul(
            self.calibration_file, np.array([point[0], point[1], 1]))
        # result = cv2.perspectiveTransform(np.array([point[0], point[1], 1]), self.calibration_file/self.calibration_file[2][2])
        result = (result/result[2])
        # print(result)
        return (int(result[0]), int(result[1]))

    def get_top_view(self, frame):
        return cv2.warpPerspective(frame, self.calibration_file/self.calibration_file[2][2], (700, 560))
        # return cv2.warpPerspective(frame, self.calibration_file/self.calibration_file[2][2], (360, 288))

    def detect_people(self, img):
        detections = detect(img)
        # (h, w) = img.shape[:2]
        # blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)
        # self.net.setInput(blob)
        # detections = self.net.forward()
        return detections

    def validate_and_link_trackers(self):
        d_points = []
        index = 0
        for (t, l) in zip(self.trackers, self.labels):
            pos = t.get_position()
            # unpack the position object
            startX, startY = int(pos.left()), int(pos.top())
            endX, endY = int(pos.right()), int(pos.bottom())
            if self.search_and_match(index) != -1:
                self.trackers.remove(t)
                self.labels.remove(l)
                continue
            self.labels[index] = ""
            point = (int(((startX+endX)/2)*(self.intial_width/600)), int(endY * (self.intial_width/600)))
            pos = self.get_top_view_of_point(point)
            cluster = Cluster(DPoint(pos[0], pos[1],(0, 0, 0),0))
            cluster.s_i = 0
            d_points.append(cluster)
            index += 1


        for id in Person.persons_db:
            if Person.persons_db[id].damaged():
                continue 
            Person.persons_db[id].s_i = 1
            d_points.append(Person.persons_db[id])
        person_cluster_clusters = KMeans.predict(d_points, Config.person_max_displacement)
        

        for person_cluster_cluster in person_cluster_clusters:
            person = None
            cluster = None
            for person_or_cluster in person_cluster_cluster.d_points:
                if str(type(person_or_cluster)) == "<class 'lab.Person'>":
                    person = person_or_cluster
                else:
                    cluster = person_or_cluster
            # fires when an id finds a match cluster
            # probably id just made a small displacement
            if person is not None and cluster is not None:
                self.labels[d_points.index(cluster)] = "person " + str(person.id)
                
        # print(self.labels)

    def search_and_match(self, tracker_index):
        pos = self.trackers[tracker_index].get_position()
        # unpack the position object
        startX, startY = int(pos.left()), int(pos.top())
        endX, endY = int(pos.right()), int(pos.bottom())
        boundries = (startX, startY, endX, endY)
        intersection_over_union_ratio = 0
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
            if temp > intersection_over_union_ratio:
                intersection_over_union_ratio = temp
                target = index
        if intersection_over_union_ratio > Config.max_intersection_over_union:
            return target
        else:
            return -1

    def forward(self):

        if self.frames_counter > self.max_frames_counter:
            self.__del__()
            assert(False)

        if Avatar.pause:
            sleep(0.5)
            return 
        points = []

        # uncomment to print frame counter
        # print(self.frames_counter, self.max_frames_counter)

        self.frames_counter += 1

        # grab the next frame from the video file
        (grabbed, frame) = self.vs.read()
        intial_width = frame.shape[1]
        self.intial_width = intial_width
        # Create a blank 300x300 black image
        board = np.zeros(
            (frame.shape[0], frame.shape[1], frame.shape[2]), np.uint8)
        # Fill board with red color(set each pixel to red)
        board[:] = (0, 0, 0)
        board = frame
        # increment detection counter and reset trackers if reach max
        self.detection_counter += 1 / Avatar.fbs

        if self.detection_counter >= Config.detection_interval:
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
        if self.writer is None and self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter('output/'+str(self.s_i)+'.avi', fourcc, 30,
                                          (frame.shape[1], frame.shape[0]), True)

        # if there are no object trackers we first need to detect objects
        # and then create a tracker for each object
        if self.detection_counter == 0:
            self.trackers = []
            self.labels = []
            detections = self.detect_people(frame)

            # loop over the detections
            (h, w) = frame.shape[:2]
            for i in np.arange(0, len(detections[0])):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0][i]

                # filter out weak detections by requiring a minimum
                # confidence
                if True:
                    #print("===",confidence)
                    # extract the index of the class label from the
                    # detections list
                    # idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    # if self.CLASSES[idx] != "person":
                        # continue

                    

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    xx,yy,ww,hh = detections[1][i]#, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = int(xx), int(yy), int(xx+ww), int(yy+hh) 
                    #box.astype("int")
                    # if box match another one in range x continue
                    # center = (startX + (endX - startX) / 2, startY + (endY - startY) / 2)
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and start the correlation tracker
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    t.start_track(rgb, rect)

                    # update our set of trackers and corresponding class
                    # labels
                    self.labels.append("")
                    self.trackers.append(t)

            self.validate_and_link_trackers()
            # loop over each of the self.trackers
            for (t, l) in zip(self.trackers, self.labels):
                #print(str(t))
                pos = t.get_position()
                # unpack the position object
                startX, startY = int(pos.left()), int(pos.top())
                endX, endY = int(pos.right()), int(pos.bottom())
                if l != "":
                    # draw the bounding box from the correlation object tracker
                    # if confidencex >= 0.5:
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                    cv2.putText(frame, l, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    cv2.circle(frame, (int((startX+endX)/2), endY),
                            5, self.points_color, -1)
                points.append(
                    (int(((startX+endX)/2)*(intial_width/600)), int(endY * (intial_width/600))))
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
                if l != "":
                    # draw the bounding box from the correlation object tracker
                    # if confidencex >= 0.5:
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                (0, 255, 0), 2)
                    cv2.putText(frame, l, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    cv2.circle(frame, (int((startX+endX)/2), endY),
                            5, self.points_color, -1)
                points.append(
                    (int(((startX+endX)/2)*(intial_width/600)), int(endY * (intial_width/600))))

        # check to see if we should write the frame to disk
        if self.writer is not None:
            self.writer.write(frame)
        board = imutils.resize(board, width=intial_width)
        # run one of those examples
        # example 1
        board = self.get_top_view(board)
        global shape
        shape = board.shape
        for point in points:
            top_view = self.get_top_view_of_point(point)
            if top_view[0] >= 0 and top_view[1] >= 0:
                Avatar.d_points.append(DPoint(top_view[0], top_view[1], self.points_color, self.s_i, np.array([self.s_i])))
            cv2.circle(board, top_view, 5, self.points_color, -1)

        cv2.imshow("Frame"+str(self.s_i), frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            while (cv2.waitKey(1) & 0xFF) != ord("s"):
                pass
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            return
        self.frames.append((b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tostring() + b'\r\n'))
        # update the FPS counter
        self.fps.update()
