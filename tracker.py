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
stream_num = 0
frame_num = 0
d_ps = []

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

# point has the form (posX, posY)
def get_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2+(point1[1] - point2[1])**2)

# point has the form [(posX, posY), (r, g, b), stream_num, flag]
# this flag if true it indicates that this point has already been clustered so ignore it (True == occupied)
# however here we use the first 3 elements only
def find_nearst_point_from_spec_stream(point, list_points, stream_num):
    target = None
    error = float("inf")
    for cur_point in list_points:
        # if it has different stream num than the required or alredy occupied then continue
        if cur_point[2] != stream_num or cur_point[3]:
            continue
        distance = get_distance(point[0], cur_point[0])
        if distance < error:
            error = distance
            target = cur_point
    global maxError
    if error <= maxError:
        return target
    else:
        return None

# point has the form [(posX, posY), (r, g, b), stream_num, flag]
# however here we use the first element only
def find_nearst_point(point, list_points, maxError):
    target = None
    error = float("inf")
    for cur_point in list_points:
        distance = get_distance(point[0], cur_point[0])
        if distance < error:
            error = distance
            target = cur_point
    if error <= maxError:
        return target
    else:
        return None

# point has the form [(posX, posY), (r, g, b), stream_num, flag]
def make_clusters(list_points):
    clusters = []
    #last stream index    
    global stream_num #the last stream num found
    for point in list_points:
        # if point already in a cluster then continue
        if point[3]:
            continue
        # consider this point a cluster then loop try to find nearst points in other streams
        cluster = point
        # since we made a cluster of this point we should set flag to true to indicate that it is occupied 
        point[2] = True
        for i in range(stream_num + 1):
            # we dont want to find any nearst point in the same stream so we continue if the point has the same stream num
            if point[2] == i:
                continue
            # try to find nearst point in stream i
            target = find_nearst_point_from_spec_stream(cluster, list_points, i)
            # if found then update cluster point to be the intermediate point between org point and the match found
            if target is not None:
                # set flag in target point to true to indicate that it is occupied
                target[3] = True
                cluster = ((int((cluster[0][0]+target[0][0])/2), int((cluster[0][1]+target[0][1])/2)),(int((cluster[1][0]+target[1][0])/2),int((cluster[1][1]+target[1][1])/2),int((cluster[1][2]+target[1][2])/2)))
        # then append the cluster to the clusters array and continue processing the other points
        clusters.append(cluster)
    return clusters

shape = (0, 0, 0)

maxError =  100 #max distance between points to form a cluster
maxErrorForBoxToGetId = 20 #max distance between points to form a cluster
maxErrorForIds = 40 #max distance between old cluster and the new one
list_points = [] #this would be overwritten after each forward
ids = [] #this would not be overwirtten but updated
last_id = 0

pause = False

def get_fbs():
    global fbs
    return fbs
def togglePause():
    global pause
    pause = not pause

def validate_clusters(clusters):
    # ids has the form of [(posx, posy), id_num, life_time, hide]
    global ids
    global last_id
    for id in ids:
        id[2] -= 1
    # # for each cluster find the nearst id and if a match found within a certain threshold
    # # then update the position of that match with the current cluster position and continue
    # # if no match found then this is probalby a new id so append the cluster to the ids array
    conterato = 0
    clusters = clusters.copy()
    while len(clusters) > 0:

        conterato = conterato + 1
        if(conterato>200 ):
            break
        cluster = clusters[0]
        id = find_nearst_point(cluster, ids, maxErrorForIds)
        if id is None:
            last_id += 1
            ids.append([cluster[0],last_id, 3, False])
            clusters.remove(cluster)
        else:
            if(find_nearst_point(id, clusters, maxErrorForIds) != cluster):
                continue
            id[0] = cluster[0]
            id[2] = 3
            clusters.remove(cluster)
    for id in ids:
        if(id[2] <= 0):
            ids.remove(id)

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

class avatar():
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    maxintersectionOverUnion = 0.2
    detection_interval = 1

    def reset_list_points(imshow):
        global shape # has shape of original frame
        board = np.zeros(shape, np.uint8)
        board[:] = (0, 0, 0)
        global list_points
        # a point has the form of [(posX, posY), (r, g, b), stream_num, flag]
        for point in list_points:
            cv2.circle(board, point[0], 10, point[1], -1)
        # a cluster has the form of [(posx, posy, (r, g, b))]
        clusters = make_clusters(list_points)
        global frame_num
        print("#Frame : " + str(frame_num))
        frame_num += 1
        data = []
        for cluster in clusters:
            data.append(cluster[0])
            cv2.circle(board, cluster[0], 15, cluster[1], -1)
        if frame_num % 25 == 0:
            global ground_truth_file_path
            f= open(ground_truth_file_path,"a+")
            f.write(str(frame_num) + " " + str(data) + "\n")
            f.close()
            # d_ps.append(data)
        validate_clusters(clusters)
        # global ids
        # for id in ids:
        #     cv2.putText(board, str(id[1]), id[0],
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 10)
        list_points = []
        if imshow:
            cv2.imshow("Board",
                    imutils.resize(board, width=600))

    def getFrame(self):
        return self.frames.pop(0)

    def __init__(self, video, save_output, imshow, calibration_file, points_color):
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
        # self.confidence = confidence
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
        # flag to diplay cv2 imshow windoes or not
        self.imshow = imshow

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

    def validate_trackers(self):
        global ids
        list_ids = []
        for id in ids:
            list_ids.append(id)
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
            # # here you should use get_top_view_of_point with the point of bottom middle point on box to get position in top view  
            # # find the nearst id to that view and append that id with the label of this box
            label = "person"
            point = (int(((startX+endX)/2)*(self.intial_width/600)), int(endY * (self.intial_width/600)))
            id = find_nearst_point([self.get_top_view_of_point(point)],list_ids, 200)
            if id is not None:
                list_ids.remove(id)
                if id[3]:
                    self.labels[index] = ""
                else:
                    self.labels[index] = 'person '+ str(id[1])
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
        global pause
        if pause:
            sleep(0.5)
            return 
        points = []
        # grab the next frame from the video file
        (grabbed, frame) = self.vs.read()
        intial_width = frame.shape[1]
        self.intial_width = intial_width
        # # Create a blank 300x300 black image
        board = np.zeros(
            (frame.shape[0], frame.shape[1], frame.shape[2]), np.uint8)
        # # Fill board with red color(set each pixel to red)
        board[:] = (0, 0, 0)
        board = frame
        # increment detection counter and reset trackers if reach max
        current_FPS = self.vs.get(cv2.CAP_PROP_FPS)
        global fbs
        fbs = current_FPS
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
        if self.writer is None and self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter('output/'+str(self.stream_num)+'.avi', fourcc, 30,
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
                    label = "person"
                    self.labels.append(label)
                    self.trackers.append(t)

            self.validate_trackers()
            # loop over each of the self.trackers
            for (t, l) in zip(self.trackers, self.labels):
                #print(str(t))
                pos = t.get_position()
                # unpack the position object
                startX, startY = int(pos.left()), int(pos.top())
                endX, endY = int(pos.right()), int(pos.bottom())
                if l == "":
                    try:
                        sub_face = frame[startY:endY, startX:endX]
                        # Get input size
                        width, height, _ = sub_face.shape
                        # Desired "pixelated" size
                        w, h = (16, 16)
                        # Resize input to "pixelated" size
                        sub_face = cv2.resize(sub_face, (w, h), interpolation=cv2.INTER_LINEAR)
                        # Initialize output image
                        sub_face = cv2.resize(sub_face, (height, width), interpolation=cv2.INTER_NEAREST)
                        frame[startY:endY, startX:endX] = sub_face
                    except:
                        pass
                else:
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
                if l == "":
                    try:
                        sub_face = frame[startY:endY, startX:endX]
                        # Get input size
                        width, height, _ = sub_face.shape
                        # Desired "pixelated" size
                        w, h = (16, 16)
                        # Resize input to "pixelated" size
                        sub_face = cv2.resize(sub_face, (w, h), interpolation=cv2.INTER_LINEAR)
                        # Initialize output image
                        sub_face = cv2.resize(sub_face, (height, width), interpolation=cv2.INTER_NEAREST)
                        frame[startY:endY, startX:endX] = sub_face
                    except:
                        pass
                else:
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
            global list_points
            list_points.append([top_view, self.points_color, self.stream_num, False])
            cv2.circle(board, top_view, 5, self.points_color, -1)
        
        # example 2
        # for point in points:
        #     cv2.circle(board, point, 1, self.points_color, -1)
        # board = self.get_top_view(board)

        # show the output frame
        # cv2.imshow("Frame(cal)"+str(self.stream_num),
        #            imutils.resize(board, width=600))
        if self.imshow:
            cv2.imshow("Frame"+str(self.stream_num), frame)
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
