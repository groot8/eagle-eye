import numpy as np
import cv2
from lab import *

data = [
    [
        DPoint(200,125,(0,255,0),1),
        DPoint(250,125,(255,0,0),2),
        DPoint(200,300,(0,255,0),1),
        DPoint(200,400,(255,0,0),2),
        DPoint(200,200,(255,0,0),2),
        DPoint(450,200,(0,0,255),3)
    ],[
        DPoint(170,125,(0,255,0),1)
    ],[
        DPoint(350,125,(0,255,0),1),
        DPoint(210,300,(255,0,0),2)
    ],[
        DPoint(350,125,(0,255,0),1)
    ],[
        DPoint(360,135,(0,255,0),1)
    ],[
        DPoint(370,145,(0,255,0),1),
        DPoint(220,300,(255,0,0),2)
    ],[
        DPoint(380,155,(0,255,0),1)
    ],[
        DPoint(390,165,(0,255,0),1)
    ]
]

board = np.zeros((500,500,3), np.uint8)

for d_points in data:
    board[:] = (255, 255, 255)
    Person.updateIds(d_points)
    Person.imDrawPersons(board)
    cv2.imshow("List of points", board)
    cv2.waitKey()