# USAGE
# python multi_object_tracking_slow.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
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


def search_and_match(point):
	minError = math.inf
	target = None
	for t in trackers:
		current = t.get_position().center()
		error = math.sqrt((point[0]-current.x)**2+(point[1]-current.y)**2)
		if error < minError:
			minError = error
			target = t
	#print(minError)
	if minError < 50:
		return target
	else:
		return None

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
# print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and output video writer
# print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# initialize the list of object trackers and corresponding class
# labels	
trackers = []
labels = []

# start the frames per second throughput estimator
fps = FPS().start()

# intialize id
id = 1

# counter to reset trackers
detection_counter = 0

# loop over frames from the video file stream
while True:
	
	# grab the next frame from the video file
	(grabbed, frame) = vs.read()

	# increment detection counter and reset trackers if reach max
	current_FPS = vs.get(cv2.CAP_PROP_FPS)
	detection_counter += 1 / current_FPS
	# print(detection_counter)
	if detection_counter >= 0.5:
		detection_counter = 0
	
	# check to see if we have reached the end of the video file
	if frame is None:
		break

	# resize the frame for faster processing and then convert the
	# frame from BGR to RGB ordering (dlib needs RGB ordering)
	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# if there are no object trackers we first need to detect objects
	# and then create a tracker for each object
	if detection_counter == 0:
		# grab the frame dimensions and convert the frame to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

		# pass the blob through the network and obtain the detections
		# and predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				#print("===",confidence)
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx] + str(id)

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue


				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# if box match another one in range x continue
				center = (startX + (endX - startX) / 2, startY + (endY - startY) / 2)
				if search_and_match(center) is not None:
					continue

				id += 1

				# construct a dlib rectangle object from the bounding
				# box coordinates and start the correlation tracker
				t = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				t.start_track(rgb, rect)

				# update our set of trackers and corresponding class
				# labels
				labels.append(label)
				trackers.append(t)

				# grab the corresponding class label for the detection
				# and draw the bounding box
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# otherwise, we've already performed detection so let's track
	# multiple objects
	else:
		# loop over each of the trackers
		for (t, l) in zip(trackers, labels):
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			#print(str(t))
			pos = t.get_position()
			# unpack the position object
			startX, startY = int(pos.left()), int(pos.top())
			endX, endY= int(pos.right()), int(pos.bottom())
			print('sx', startX)
			print('sy', startY)
			print('ex', endX)
			print('ey', endY)
			h_orig, w_orig = frame.shape[:2]
			padding = 30
			xstartX = startX-padding if startX-padding>=0 else 0  
			xstartY = startY-padding if startY-padding>=0  else 0
			xendX = endX+padding if endX+padding<=w_orig else w_orig 
			xendY = endY+padding if endY+padding<=h_orig else h_orig  
			subimg = frame[xstartY:xendY, xstartX:xendX]
			(h, w) = subimg.shape[:2]
			blobx = cv2.dnn.blobFromImage(subimg, 0.007843, (w, h), 127.5)
			# pass the blob through the network and obtain the detections
			# and predictions
			net.setInput(blobx)
			detectionsx = net.forward()
			confidencex = detectionsx[0, 0, 0, 2]
			print("insha2allah",confidencex)
			#cv2.imshow('test',test)
			#with open("Output.txt",'w') as ofile:
			#	ofile.write(str(test))


			# draw the bounding box from the correlation object tracker
			if confidencex != 0:
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, l, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()