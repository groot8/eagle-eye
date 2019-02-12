import cv2
import os
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
    help="video name")
ap.add_argument("-d", "--dist", required=True,
    help="folder name")
args = vars(ap.parse_args())
os.system('mkdir '+args['dist'])
vidcap = cv2.VideoCapture(args['video'])
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(args['dist']+"/frame%d.jpg" % count, image)
  success,image = vidcap.read()
  count += 1
  print('frame ', count, 'extracted!')
print('Extracted ',count,'frame successfully!')
