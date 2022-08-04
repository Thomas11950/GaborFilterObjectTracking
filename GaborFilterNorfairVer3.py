import cv2
import numpy as np

from norfair import Detection, Tracker, Video, draw_tracked_objects
from scipy.io import loadmat

class GaborFilter:
    def __init__(self, GaborFilterFilepath, gaborThreshold, groupSizeHeight, groupSizeWidth):
        self.GaborFilter = loadmat(GaborFilterFilepath)['tmp997']
        self.groupSizeHeight = groupSizeHeight
        self.groupSizeWidth = groupSizeWidth
        self.gaborThreshold = gaborThreshold
    def isWithinOtherGroup(self, mat, row, col):
        h, w, depth = mat.shape
        lowerBoundRow = row - self.groupSizeHeight
        upperBoundRow = row + self.groupSizeHeight + 1
        lowerBoundCol = col - self.groupSizeWidth
        upperBoundCol = col + self.groupSizeWidth + 1
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
                if mat[i][j][0] > self.gaborThreshold and mat[i][j][1] != -1:
                    return mat[i][j][1]
        return -1

    def getGaborFilterValue(self, past7Frames, centerR, centerC):
        accumulate = 0
        for i in range(9):
            for j in range(9):
                for k in range(7):
                    accumulate += self.GaborFilter[i][j][k] * past7Frames[k][i-4+centerR][j-4+centerC]
        return accumulate
    
    def getGroups(self, grayHistory):
        h = len(grayHistory[0])
        w = len(grayHistory[0][0])
        mat = np.zeros((h,w,2), dtype = "int32")
        for i in range(h):
            for j in range(w):
                mat[i][j][1] = -1

        for i in range(int(4),int(h-4)):
            for j in range(int(4), int(w-4)):
                gaborValue = self.getGaborFilterValue(grayHistory, i, j)
                mat[i][j][0] = gaborValue
                
    
        #assign "groups" to all of the Mats
        nextGroupNumber = 0
        groupMinRow = []
        groupMinCol = []
        groupMaxRow = []
        groupMaxCol = []
        maxGabor = []
        for i in range(int(4),int(h-4)):
            for j in range(int(4), int(w-4)):
                if mat[i][j][0] > self.gaborThreshold:
                    groupNum = self.isWithinOtherGroup(mat,i,j)
                    if groupNum == -1:
                        mat[i][j][1] = nextGroupNumber
                        groupMinRow.append(i)
                        groupMinCol.append(j)
                        groupMaxRow.append(i)
                        groupMaxCol.append(j)
                        maxGabor.append(mat[i][j][0])
                        nextGroupNumber += 1
                    else:
                        mat[i][j][1] = groupNum
                        
                        if i < groupMinRow[groupNum]:
                            groupMinRow[groupNum] = i
                        if i > groupMaxRow[groupNum]:
                            groupMaxRow[groupNum] = i
                        if j < groupMinCol[groupNum]:
                            groupMinCol[groupNum] = j
                        if j > groupMaxCol[groupNum]:
                            groupMaxCol[groupNum] = j
                        if mat[i][j][0] > maxGabor[groupNum]:
                            maxGabor[groupNum] = mat[i][j][0]
        groups = []
        for i in range(nextGroupNumber):
            groups.append([[(groupMinRow[i]+groupMaxRow[i])/2, (groupMinCol[i]+groupMaxCol[i])/2],[groupMinRow[i], groupMinCol[i], groupMaxRow[i], groupMaxCol[i]],maxGabor[i]])
        return groups

class GaborFilters:
    global numGaborSpeeds
    global numGaborAngles
    numGaborSpeeds = 2
    numGaborAngles = 8
    def __init__(self, gaborThreshold, groupSizeHeight, groupSizeWidth):
        self.gaborFilters = []
        for i in range(numGaborSpeeds):
            listToAdd = []
            for j in range(numGaborAngles):
                listToAdd.append(GaborFilter("Filters/filter_"+str(i+2)+"_"+str(j+1)+".mat", gaborThreshold,groupSizeHeight, groupSizeWidth))
            self.gaborFilters.append(listToAdd)
    def getGroupsFromDifferentFilters(self,grayHistory):
        groupList = []
        for i in range(numGaborSpeeds):
            listToAdd = []
            for j in range(numGaborAngles):
                groupsToAdd = self.gaborFilters[i][j].getGroups(grayHistory)
                for group in groupsToAdd:
                    group.append([i,j])
                listToAdd.append(groupsToAdd)
                print("\nfinished convolving with filter " + str(j) + "\n")
            groupList.append(listToAdd)
        return groupList

    class SetOfGroups:
        def __init__(self, groups):
            self.groups = groups
        def determineIfTwoGroupsOverlap(self,group1,group2):
            group1RowMin = group1[1][0]
            group1RowMax = group1[1][2]
            group1ColMin = group1[1][1]
            group1ColMax = group1[1][3]
            group2RowMin = group2[1][0]
            group2RowMax = group2[1][2]
            group2ColMin = group2[1][1]
            group2ColMax = group2[1][3]
            print("entering determineIfTwoGroupsOverlap function\n")
            print("group1RowMin: " + str(group1RowMin) + ", group1RowMax: " + str(group1RowMax) + ", group1ColMin: " + str(group1ColMin) + ", group1ColMax: " + str(group1ColMax) + ", group2RowMin: " + str(group2RowMin) + ", group2RowMax: " + str(group2RowMax) + ", group2ColMin: " + str(group2ColMin) + ", group2ColMax: " + str(group2ColMax) + "\n")
            toReturn = not ((group1ColMax < group2ColMin or group1ColMin > group2ColMax) or (group1RowMin > group2RowMax or group1RowMax < group2RowMin))
            print("toReturn: " + str(toReturn) + "\n")
            return toReturn
        def addGroupIfOverlapWithExisting(self, groupToAdd):
            for existingGroup in self.groups:
                if self.determineIfTwoGroupsOverlap(existingGroup, groupToAdd):
                    self.groups.append(groupToAdd)
                    print("in the addGroupIfOverlapWithExisting function, the following group is added: " + str(groupToAdd) + "\n")
                    print("the existing groups in the overlaps are: " + str(self.groups))
                    return True
            return False
        def determineGroupPosition(self):
            maxGabor = 0
            posOfMaxGabor = -1
            for i in range(len(self.groups)):
                if self.groups[i][2] > maxGabor:
                    maxGabor = self.groups[i][2]
                    posOfMaxGabor = i
            return self.groups[posOfMaxGabor]

    def matchGroupsAndDeterminePeakFilter(self,grayHistory):
        
        groupsFromDifferentFilters = self.getGroupsFromDifferentFilters(grayHistory)
        combinedGroups = []
        for i in range(numGaborSpeeds):
            for j in range(numGaborAngles):
                for groupToAdd in groupsFromDifferentFilters[i][j]:
                    addedIntoExistingSet = False
                    for alreadyAddedSetOfGroups in combinedGroups:
                        if alreadyAddedSetOfGroups.addGroupIfOverlapWithExisting(groupToAdd):
                            addedIntoExistingSet = True
                            break
                    if not addedIntoExistingSet:
                        combinedGroups.append(self.SetOfGroups([groupToAdd]))
        combinedGroupsOnlyPeakGabor = []
        for combinedGroup in combinedGroups:
            combinedGroupsOnlyPeakGabor.append(combinedGroup.determineGroupPosition())
        return combinedGroupsOnlyPeakGabor


    

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

groupSizeWidth = int(input("group size width: "))
groupSizeHeight = int(input("group size height: "))
gaborThreshold = int(input("Gabor Threshold: "))

# Norfair
input_video_path = input("input video path: ")
output_video_path = input("output video path: ")
video = Video(input_path=input_video_path, output_path = output_video_path)
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=20)

numFramesTraversed = 0
grayHistory = []
frameHistory = []

log = open(output_video_path + "log.txt", 'w')
trackedObjectsOut = open(output_video_path + "trackedObjectsOut.txt", 'w')

gaborFilters = GaborFilters(gaborThreshold,groupSizeHeight,groupSizeWidth)
for frame in video:
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    grayMat = np.zeros((h,w), dtype = "int16")
    for i in range(h):
        for j in range(w):
            grayMat[i][j] = gray[i,j] - 128

    if numFramesTraversed < 7:
        grayHistory.append(grayMat)
        frameHistory.append(frame)
        numFramesTraversed += 1
        continue
    else:
        numFramesTraversed += 1
    for i in range(6):
        grayHistory[6-i] = grayHistory[6-i-1]
        frameHistory[6-i] = frameHistory[6-i-1]
    grayHistory[0] = grayMat
    frameHistory[0] = frame
    groupsToTrack = gaborFilters.matchGroupsAndDeterminePeakFilter(grayHistory)
    detections = []
    log.write("currently on frame: " + str(numFramesTraversed)+"\n")
    for group in groupsToTrack:
        detections.append(Detection(points = np.array([group[0][0],group[0][1]]), scores = np.array([1])))
        print("\npoint to add to detections: " + str(group[0][0]) + ", " + str(group[0][1]))
        frameHistory[3] = cv2.arrowedLine(frameHistory[3], (int(group[0][1]), int(h-group[0][0])),(int(group[0][1] + 60 *(group[3][0]-0.5)* np.cos((group[3][1]-1) * 0.3926991 - np.pi/2)), int(h-group[0][0] - 60 *(group[3][0]-0.5)* np.sin((group[3][1]-1) * 0.3926991 - np.pi/2))),(255,0,0),2)
        frameHistory[3] = cv2.putText(frameHistory[3], str(group[3][0]) + ", " + str(group[3][1]), (int(group[0][1]),int( h-group[0][0])), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2, cv2.LINE_AA)
        log.write("group: " + str(group) + "\n")
                        
    print("\nlength of detections: " + str(len(detections)) +"\n")
    tracked_objects = tracker.update(detections=detections)
    for tracked_object in tracked_objects:
        trackedObjectsOut.write(str(numFramesTraversed-3)+","+str(tracked_object.estimate)+","+str(tracked_object.id)+","+str(tracked_object.last_detection)+","+str(tracked_object.last_distance)+","+str(tracked_object.age)+","+str(tracked_object.live_points)+","+str(tracked_object.initializing_id)+"\n")
    draw_tracked_objects(frameHistory[3], tracked_objects)
    video.write(frameHistory[3])

log.close()
trackedObjectsOut.close()