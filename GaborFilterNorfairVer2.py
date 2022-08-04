import cv2
import numpy as np

from norfair import Detection, Tracker, Video, draw_tracked_objects
from scipy.io import loadmat


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

groupSizeWidth = 15
groupSizeHeight = 15

#returns -1 if not within other group, else returns group number
def isWithinOtherGroup(mat, row, col):
    h, w, depth = mat.shape
    lowerBoundRow = row - groupSizeHeight
    upperBoundRow = row + groupSizeHeight + 1
    lowerBoundCol = col - groupSizeWidth
    upperBoundCol = col + groupSizeWidth + 1
    if lowerBoundRow < 4:
        lowerBoundRow = 4
    if upperBoundRow > h-4:
        upperBoundRow = h-4
    if lowerBoundCol < 4:
        lowerBoundCol = 4
    if upperBoundCol > w-4:
        upperBoundCol = w-4
    for i in range(int(lowerBoundRow), int(upperBoundRow)):
        for j in range(int(lowerBoundCol), int(upperBoundCol)):
            if mat[i][j][0] > gaborThreshold and mat[i][j][1] != -1:
                return mat[i][j][1]
    return -1
            


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
video = Video(input_path="./reconstructedevents.avi")
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=20)

numFramesTraversed = 0
grayHistory = []
frameHistory = []
for frame in video:
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if numFramesTraversed < 7:
        grayHistory.append(gray)
        frameHistory.append(frame)
        numFramesTraversed += 1
        continue
    for i in range(6):
        grayHistory[6-i] = grayHistory[6-i-1]
        frameHistory[6-i] = frameHistory[6-i-1]
    grayHistory[0] = gray
    frameHistory[0] = frame
    h, w = gray.shape
    detections = []
    groups = []
    mat = np.zeros((h,w,2), dtype = "int16")
    #set all group numbers to -1 initially
    for i in range(h):
        for j in range(w):
            mat[i][j][1] = -1

    for i in range(int(4),int( h-4)):
        for j in range(int(4), int(w-4)):
            gaborValue = getGaborFilterValue(GaborFilter['tmp997'], grayHistory, i, j)
            mat[i][j][0] = gaborValue
                
    #assign "groups" to all of the Mats
    nextGroupNumber = 0
    groupMaxes = []
    for i in range(int(4),int(h-4)):
        for j in range(int(4), int(w-4)):
            if mat[i][j][0] > gaborThreshold:
                groupNum = isWithinOtherGroup(mat,i,j)
                if groupNum == -1:
                    mat[i][j][1] = nextGroupNumber
                    groupMaxes.append([i,j])
                    nextGroupNumber += 1
                else:
                    mat[i][j][1] = groupNum
                    groupMax = groupMaxes[groupNum]
                    if mat[i][j][0] > mat[groupMax[0]][groupMax[1]][0]:
                        groupMaxes[groupNum] = [i,j]

    for i in range(len(groupMaxes)):
        detections.append(Detection(points = np.array([groupMaxes[i][1],groupMaxes[i][0]]), scores = np.array([1])))
        print("\npoint to add to detections: " + str(groupMaxes[i][1]) + ", " + str(groupMaxes[i][0]))

    print("\nlength of detections: " + str(len(detections)) +"\n")
    tracked_objects = tracker.update(detections=detections)
    draw_tracked_objects(frameHistory[3], tracked_objects)
    video.write(frameHistory[3])
    
