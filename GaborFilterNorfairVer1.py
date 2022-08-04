import cv2
import numpy as np

from norfair import Detection, Tracker, Video, draw_tracked_objects
from scipy.io import loadmat


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


#returns true if the point (xCoord, yCoord) is within an exclusion zone. Exclusion zones are defined as 20 by 20 squares centered on the point listed inside exclusionZones
def isWithinExclusionZone(exclusionZones, yCoord, xCoord, GaborValue):
    for exclusionZone in exclusionZones:
        
        if abs(exclusionZone[0] - yCoord) <= 5 and abs(exclusionZone[1] - xCoord) <= 15:
            if GaborValue > exclusionZone[2]:
                exclusionZone[0] = yCoord
                exclusionZone[1] = xCoord
                exclusionZone[2] = GaborValue
                return False
            else:
                return True

    exclusionZones.append([i,j,GaborValue])
    return False
            

def getGaborFilterValue(GaborFilter, past7Frames, centerR, centerC):
    accumulate = 0
    for i in range(9):
        for j in range(9):
            for k in range(7):
                accumulate += GaborFilter[i][j][k] * past7Frames[k][i-4+centerR,j-4+centerC]
    return accumulate

# Gabor filter
GaborFilter = loadmat('filter_2_4.mat')
gaborThreshold = 225

# Norfair
video = Video(input_path="./space_invaders_shortened.mp4")
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=20)

numFramesTraversed = 0
frameHistory = []
for frame in video:
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if numFramesTraversed < 7:
        frameHistory.append(gray)
        numFramesTraversed += 1
        continue
    for i in range(6):
        frameHistory[6-i] = frameHistory[6-i-1]
    frameHistory[0] = gray
    h, w = gray.shape
    detections = []
    exclusionZones = [] #array of points that blocks things from being "double counted"
    for i in range(int(h * 0.2),int( h*0.7)):
        for j in range(int(w * 0.6), int(w-4)):
            gaborValue = getGaborFilterValue(GaborFilter['tmp997'], frameHistory, i, j)
            #print("\ngaborValue: " + str(gaborValue) + "\n")
            if gaborValue > gaborThreshold:
                isWithinExclusionZone(exclusionZones, i, j, gaborValue)
                
    for exclusionZone in exclusionZones:
        i = exclusionZone[0]
        j = exclusionZone[1]
        detections.append(Detection(points = np.array([[j,i]]), scores = np.array([1])))
        print("\npoint to add to detections: " + str(i) + ", " + str(j))

    print("\nlength of detections: " + str(len(detections)) +"\n")
    tracked_objects = tracker.update(detections=detections)
    draw_tracked_objects(frame, tracked_objects)
    video.write(frame)
    
