import cv2
import math
import numpy as np
import os
import time
from knn import *

# configuration
class Config:
    cluster_max_error = 100 # max distance between d_points to form a cluster
    person_max_displacement = 100 # max distance between clusters & persons to form a person_cluster_cluster
    person_life = 3 # number of times that a person won't appear in any person_cluster_cluster before damaged
    max_intersection_over_union = 0.2 # max ratio between trackers overlapping on each other
    detection_interval = 1 # interval between detections
    delta_momentum = 0.8 #momentum to update persons that doesn't have a match at some frame before damaged

# base class
class Point:
    
    def __init__(self, pos_x = 0, pos_y = 0, color = (0,0,0)):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.color = color
    
    def getPosition(self):
        return (self.pos_x, self.pos_y)

    @staticmethod
    def calcDist(point1, point2):
        return math.sqrt(
            (point1.pos_x - point2.pos_x)**2 +
            (point1.pos_y - point2.pos_y)**2
        )

    @staticmethod
    def imDrawPoints(img, points_list, p_size = 5):
        for point in points_list:
            cv2.circle(img, point.getPosition(), p_size, point.color, -1)

# dectection point from specific stream
class DPoint(Point):

    def __init__(self, pos_x, pos_y, color, s_i, detection_features = np.array([])):        
        # pos_x & pos_y in the top view
        # s_i stands for stream index
        super().__init__(pos_x, pos_y, color)
        self.s_i = s_i
        self.detection_features = detection_features

# group of d_points that represent a person
class Cluster(Point):

    def __init__(self, i_d_point = None):
        # i_d_point stands for initial point for a cluster
        self.d_points = []
        if i_d_point is not None:
            self.addDPoint(i_d_point)

    def replaceRoot(self, cluster):
        self.d_points = cluster.d_points
        self.update()

    def update(self):
        (pos_x, pos_y, b, g, r) = (0, 0, 0, 0, 0)
        length = len(self.d_points)
        for i in range(length):
            pos_x += self.d_points[i].pos_x
            pos_y += self.d_points[i].pos_y
            b += self.d_points[i].color[0]
            g += self.d_points[i].color[1]
            r += self.d_points[i].color[2]
        (pos_x, pos_y, b, g, r) = (int(pos_x/length), int(pos_y/length), int(b/length), int(g/length), int(r/length))
        (self.pos_x, self.pos_y, self.color) = (pos_x, pos_y, (b, g, r))

    def addDPoint(self, d_point):
        self.d_points.append(d_point)
        self.update()     

    def hasCommonS_i(self, cluster):
        s_i_set = set()
        for d_point in self.d_points:
            s_i_set.add(d_point.s_i)

        for d_point in cluster.d_points:
            s_i_set.add(d_point.s_i)
        return len(cluster.d_points) + len(self.d_points) > len(s_i_set)

    def imDraw(self, img, p_size, l_size):
        if(len(self.d_points) == 0):
            return
        # draw lines connecting ponits & centroid
        for d_point in self.d_points:
            cv2.line(img, d_point.getPosition(), self.getPosition(), self.color, l_size)
        # draw points forming cluster
        Point.imDrawPoints(img, self.d_points)
        # draw cluster centroid
        cv2.circle(img, self.getPosition(), p_size, self.color, -1)

    @staticmethod
    def imDrawClusters(img, clusters, p_size = 5, l_size = 1):
        for cluster in clusters:
            cluster.imDraw(img, p_size, l_size)

# kmeans clustering algorithm to cluster dPoints
class KMeans:

    @staticmethod
    def predict(d_points, max_dist = float("inf")):
        clusters = []
        # transform array of points to array of clasters
        for d_point in d_points:
            clusters.append(Cluster(d_point))
        # iterate and perform clustering until convergion
        while True:
            match_found = False
            length = len(clusters)
            (t_i, t_j) = (0, 0)
            dist = float("inf")
            for i in range(length):
                for j in range(length):
                    # we want to make sure that points from the same strem index 
                    # wont end up in the same cluster
                    if(i != j and not clusters[i].hasCommonS_i(clusters[j])):
                        temp = Point.calcDist(clusters[i], clusters[j])
                        if(temp < dist and temp <= max_dist):
                            match_found = True
                            dist = temp
                            (t_i, t_j) = (i, j)
            if match_found:
                bias = -1 if t_i < t_j else 1
                f_c = clusters.pop(t_i)
                s_c = clusters.pop(t_j + bias)
                cluster = Cluster()
                for d_point in f_c.d_points:
                    cluster.addDPoint(d_point)
                for d_point in s_c.d_points:
                    cluster.addDPoint(d_point)
                clusters.append(cluster)
            else:
                break
        return clusters

class Person(Cluster):
    count = 0
    persons_db = {}
    def __init__(self,cluster):
        self.replaceRoot(cluster)
        Person.count += 1
        self.id = Person.count
        Person.persons_db[self.id] = self
        self.life = Config.person_life
        self.f_prev_pos = None
        self.s_prev_pos = None

    def imDraw(self, img, p_size, l_size):
        cv2.putText(img, str(self.id), (self.pos_x - 5, self.pos_y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
        super().imDraw(img, p_size, l_size)

    def heal(self):
        self.life = Config.person_life

    def causeDamage(self):
        self.life -= 1

    def damaged(self):
        return self.life <= 0

    def stepForward(self):
        if self.s_prev_pos is None:
            self.s_prev_pos = [self.pos_x, self.pos_y]
        elif self.f_prev_pos is None:
            self.f_prev_pos = [self.pos_x, self.pos_y]
        else:
            d_x = self.f_prev_pos[0] - self.s_prev_pos[0]        
            d_y = self.f_prev_pos[1] - self.s_prev_pos[1]
            self.s_prev_pos = self.f_prev_pos        
            self.f_prev_pos = [
                int(self.s_prev_pos[0] + d_x*Config.delta_momentum), 
                int(self.s_prev_pos[1] + d_y*Config.delta_momentum)
            ]
            self.pos_x, self.pos_y = self.f_prev_pos[0], self.f_prev_pos[1]

    report_file_path = '/home/yousif-elmos/Desktop/college/eagle-eye/output/report.json'

    report_file_stream = None

    @staticmethod
    def saveReport():
        if Person.report_file_stream is None and Person.report_file_path is not None:
            Person.report_file_stream = open(Person.report_file_path, "w+")
        Person.report_file_stream.write('{')
        flag = True
        for id in Person.persons_db:
            if Person.persons_db[id].damaged():
                continue
            string = '"'+str(id)+'":['+str(Person.persons_db[id].pos_x)+','+str(Person.persons_db[id].pos_y)+']'
            if flag:
                flag = False
            else:
                string = ","+string
            Person.report_file_stream.write(string)            
        Person.report_file_stream.write('}\n')

    @staticmethod
    def imDrawPersons(img, p_size = 5, l_size = 1):
        for id in Person.persons_db:
            if Person.persons_db[id].damaged():
                continue
            Person.persons_db[id].imDraw(img, p_size, l_size)
        Person.saveReport()

    # this method is responsible for updating ids positions and decide
    # what clusters should make new persons and what should not
    @staticmethod
    def updateIds(d_points, accept_new_ids):
        # get current view clusters from d_points
        clusters = KMeans.predict(d_points, Config.cluster_max_error)
        # cluster the previous clusters with persons detected
        d_points = []
        for c in clusters:
            c.s_i = 0
            d_points.append(c)
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
            if person is not None and cluster is not None and len(cluster.d_points) >= 2:
                person.replaceRoot(cluster)
                person.heal()
            # fires when an id doesn't find a match            
            # probably that id disappeard from that frame
            elif cluster is None:
                person.stepForward()
                person.causeDamage()
            # first when a cluster doesn't find a match
            # probably someone new appeared
            elif len(cluster.d_points) >= 2 and accept_new_ids:
                features = []
                for d_point in cluster.d_points:
                    features.append(d_point.detection_features)
                id = Knn.predict(features)
                if id is None:
                    Person(cluster)
                else:
                    person = Person.persons_db[id]
                    person.replaceRoot(cluster)
                    person.heal()
        
        del person_cluster_clusters
    
    # this method is responsible for generating list_labels_features for the knn model
    @staticmethod
    def get_list_labels_features():
        list_labels_features = []
        for id in Person.persons_db:
            for d_point in Person.persons_db[id].d_points:
                list_labels_features.append([id, d_point.detection_features])
        return list_labels_features