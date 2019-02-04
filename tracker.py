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
class avatar():
    # initialize the list of class labels MobileNet SSD was trained to
    # detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    	"sofa", "train", "tvmonitor"]
    # initialize the list of object trackers and corresponding class
    # labels	
    trackers = []
    labels = []
    lives = []
    # intialize id
    id = 1
    frame = None
    # counter to reset trackers
    detection_counter = 0
    writer = None
    
    def __init__(self, prototxt, model, video, output, confidence, livesCounter = 3):
        # load our serialized model from disk
        # print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.livesCounter = livesCounter
        # initialize the video stream and output video writer
        # print("[INFO] starting video stream...")
        self.vs = cv2.VideoCapture(video)
        # start the frames per second throughput estimator
        self.fps = FPS().start()
        self.output = output
        self.confidence = confidence

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

    def detect_people(self, img):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections


    def validate_trackers(self, frame):
            index = 0
            for (t, l, c) in zip(self.trackers, self.labels, self.lives):
                pos = t.get_position()
                startX, startY = int(pos.left()), int(pos.top())
                endX, endY= int(pos.right()), int(pos.bottom())
                h_orig, w_orig = frame.shape[:2]
                padding = 50
                startX = startX-padding if startX-padding>=0 else 0  
                startY = startY-padding if startY-padding>=0  else 0
                endX = endX+padding if endX+padding<=w_orig else w_orig 
                endY = endY+padding if endY+padding<=h_orig else h_orig  
                subimg = frame[startY:endY, startX:endX]
                detections = self.detect_people(subimg)
                confidence = detections[0,0,0,2]
                print("confidence",confidence)
                if confidence <= 0.3:
                    print('lives: ',self.lives)
                    if c  <= 0:
                        self.trackers.remove(t)
                        self.labels.remove(l)
                        self.lives.remove(c)
                        print('removed...', l)
                    else:
                        self.lives[index]-=(1-confidence)*1.2
                else:
                    self.lives[index] = self.lives[index]+(1-confidence) if self.lives[index]<3 else 3
                index += 1

    def search_and_match(self, point):
        minError = math.inf
        target = None
        for t in self.trackers:
            current = t.get_position().center()
            error = math.sqrt((point[0]-current.x)**2+(point[1]-current.y)**2)
            if error < minError:
                minError = error
                target = t
        #print(minError)
        if minError < 25:
            return target
        else:
            return None

    def forward(self):
        	# grab the next frame from the video file
        (grabbed, frame) = self.vs.read()
        # increment detection counter and reset trackers if reach max
        current_FPS = self.vs.get(cv2.CAP_PROP_FPS)
        self.detection_counter += 1 / current_FPS
        # print(self.detection_counter)
        if self.detection_counter >= 0.5:
            self.detection_counter = 0
        
        # check to see if we have reached the end of the video file
        if frame is None:
            return 

        # resize the frame for faster processing and then convert the
        # frame from BGR to RGB ordering (dlib needs RGB ordering)
        frame = imutils.resize(frame, width=600)
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
                
            self.validate_trackers(frame)
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
                    center = (startX + (endX - startX) / 2, startY + (endY - startY) / 2)
                    if self.search_and_match(center) is not None:
                        continue

                    self.id += 1

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and start the correlation tracker
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    t.start_track(rgb, rect)

                    # update our set of trackers and corresponding class
                    # labels
                    self.labels.append(label)
                    self.trackers.append(t)
                    self.lives.append(self.livesCounter)

                    # grab the corresponding class label for the detection
                    # and draw the bounding box
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

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
                endX, endY= int(pos.right()), int(pos.bottom())
                # draw the bounding box from the correlation object tracker
                # if confidencex >= 0.5:
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                cv2.putText(frame, l, (startX, startY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # check to see if we should write the frame to disk
        if self.writer is not None:
            self.writer.write(frame)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            return

        # update the FPS counter
        self.fps.update()

