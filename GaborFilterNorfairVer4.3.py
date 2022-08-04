#Implements frame skipping for tracking objects of different speeds, but the number of alternations can be intervaled
from binascii import a2b_base64
import cv2
import numpy as np

from norfair import Detection, Tracker, Video, draw_tracked_objects
from scipy.io import loadmat

class GaborFilter:
    def __init__(self, GaborFilterFilepath, gaborThreshold, groupSizeHeight, groupSizeWidth, gaborBaseVector):
        self.gaborBaseVector = gaborBaseVector
        GaborFilterRaw = loadmat(GaborFilterFilepath)['tmp997']
        self.GaborFilter = self.changeFilterDimensions(GaborFilterRaw)
        self.groupSizeHeight = groupSizeHeight
        self.groupSizeWidth = groupSizeWidth
        self.gaborThreshold = gaborThreshold
    def changeFilterDimensions(self, GaborFilterRaw):
        #we want to change the gabor filter to have time as first dimension, height as second, width as third. Currently, the gabor filter has height, width, time
        out = np.zeros((7, 9, 9), "float")
        for i in range(7):
            for j in range(9):
                for k in range(9):
                    out[i][j][k] = GaborFilterRaw[j][k][i]
        return out

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
        partOfFrameHistoryToMultiply = past7Frames[0:7, centerR-4:centerR+5, centerC-4:centerC+5]
        return np.sum(np.multiply(partOfFrameHistoryToMultiply, self.GaborFilter))
    
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
        groupSize = []
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
                        groupSize.append(1)
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
                        groupSize[groupNum] += 1
        groups = []
        for i in range(nextGroupNumber): 
            if groupSize[i] > 1:
                groups.append([[(groupMinRow[i]+groupMaxRow[i])/2, (groupMinCol[i]+groupMaxCol[i])/2],[groupMinRow[i], groupMinCol[i], groupMaxRow[i], groupMaxCol[i]],[maxGabor[i]*self.gaborBaseVector[0], maxGabor[i]*self.gaborBaseVector[1]]])
        return groups

class GaborFilters:
    global numGaborSpeeds
    global numGaborAngles
    numGaborSpeeds = 4;
    numGaborAngles = 8;
    def __init__(self, gaborThreshold, groupSizeHeight, groupSizeWidth, baseVectorSize, numAlternationsMin, numAlternationsMax, alternationInterval):
        self.gaborFilters = []
        for i in range(numGaborSpeeds):
            listToAdd = []
            for j in range(numGaborAngles):
                if i == 0:
                    movementMagnitude = 2
                elif i == 1:
                    movementMagnitude = 1
                elif i == 2:
                    movementMagnitude = -1
                else:
                    movementMagnitude = -2
                listToAdd.append(GaborFilter("Filters/filter_"+str(i+1)+"_"+str(j+1)+".mat", gaborThreshold,groupSizeHeight, groupSizeWidth, [-baseVectorSize * movementMagnitude * np.sin(j * 0.3926991 - np.pi/2),baseVectorSize * movementMagnitude * np.cos(j * 0.3926991 - np.pi/2)]))
            self.gaborFilters.append(listToAdd)
        self.groupSizeHeight = groupSizeHeight
        self.groupSizeWidth = groupSizeWidth
        self.numAlternationsMin = numAlternationsMin
        self.numAlternationsMax = numAlternationsMax
        self.alternationInterval = alternationInterval
    def getGroupsFromDifferentFilters(self,grayHistory):
        groupList = []
        for i in range(numGaborSpeeds):
            for j in range(numGaborAngles):
                for k in range(self.numAlternationsMin,self.numAlternationsMax+1, self.alternationInterval):

                    groupsToAdd = self.gaborFilters[i][j].getGroups(grayHistory[int(len(grayHistory)/2)-3*(int(k)):int(len(grayHistory)/2)+3*(int(k))+1:int(k)])
                    for group in groupsToAdd:
                        group[2][0] *= (k)
                        group[2][1] *= (k)
                    groupList.append(groupsToAdd)
                    print("\nfinished convolving with filter " + str(i+1) + "_" + str(j+1) + "_" + str(k) + "\n")
        return groupList

    class SetOfGroups:
        def __init__(self, groups, groupSizeHeight, groupSizeWidth):
            self.groups = groups
            self.groupSizeWidth = groupSizeWidth
            self.groupSizeHeight = groupSizeHeight
            boundingGroup = self.getBoundingGroup()
            self.minRow = boundingGroup[0]
            self.maxRow = boundingGroup[1]
            self.minCol = boundingGroup[2]
            self.maxCol = boundingGroup[3]
        def getBoundingGroup(self):
            minRow = self.groups[0][1][0]
            maxRow = self.groups[0][1][2]
            minCol = self.groups[0][1][1]
            maxCol = self.groups[0][1][3]
            for group in self.groups:
                if group[1][0] < minRow:
                    minRow = group[1][0]
                if group[1][2] > maxRow:
                    maxRow = group[1][2]
                if group[1][1] < minCol:
                    minCol = group[1][1]
                if group[1][3] > maxCol:
                    maxCol = group[1][3]
            return [minRow, maxRow, minCol, maxCol]


        def getGroups(self):
            return self.groups
        def determineIfGroupOverlaps(self, groupToAdd):
            group1RowMin = self.minRow
            group1RowMax = self.maxRow
            group1Row = (group1RowMin+group1RowMax)/2
            group1ColMin = self.minCol
            group1ColMax = self.maxCol
            group1Col = (group1ColMin+group1ColMax)/2
            group2RowMin = groupToAdd[1][0]
            group2RowMax = groupToAdd[1][2]
            group2Row = (group2RowMin+group2RowMax)/2
            group2ColMin = groupToAdd[1][1]
            group2ColMax = groupToAdd[1][3]
            group2Col = (group2ColMin+group2ColMax)/2
            toReturn = (np.abs(group1Row-group2Row) < self.groupSizeHeight and np.abs(group1Col-group2Col) < self.groupSizeWidth) or not ((group1ColMax < group2ColMin or group1ColMin > group2ColMax) or (group1RowMin > group2RowMax or group1RowMax < group2RowMin))
            return toReturn
        def addGroupIfOverlapWithExisting(self, groupToAdd):
            if self.determineIfGroupOverlaps(groupToAdd):
                print("in the addGroupIfOverlapWithExisting function, the following group is added: " + str(groupToAdd) + "\n")
                print("the existing groups in the overlaps are: " + str(self.groups))
                self.groups.append(groupToAdd)
                if groupToAdd[1][0] < self.minRow:
                    self.minRow = groupToAdd[1][0]
                if groupToAdd[1][2] > self.maxRow:
                    self.maxRow = groupToAdd[1][2]
                if groupToAdd[1][1] < self.minCol:
                    self.minCol = groupToAdd[1][1]
                if groupToAdd[1][3] > self.maxCol:
                    self.maxCol = groupToAdd[1][3]
                return True
            return False
        def groupOverlapsWithExisting(self, groupToCheck):
            return self.determineIfGroupOverlaps(groupToCheck)
        def addGroup(self, groupToAdd):
            self.groups.append(groupToAdd)
        def determineGroupPositionAndMovementVector(self):
            totalRow = 0
            totalCol = 0
            totalRowVector = 0
            totalColVector = 0
            for group in self.groups:
                totalRow += group[0][0]
                totalCol += group[0][1]
                totalRowVector += group[2][0]
                totalColVector += group[2][1]
            return [[totalRow/len(self.groups), totalCol/len(self.groups)],[totalRowVector/len(self.groups),totalColVector/len(self.groups)]]
        def addAnotherSetOfGroups(self, setOfGroupsToAdd):
            for group in setOfGroupsToAdd.getGroups():
                self.addGroup(group)

    def matchGroupsAndDeterminePeakFilter(self,grayHistory):
        
        groupsFromDifferentFilters = self.getGroupsFromDifferentFilters(grayHistory)
        combinedGroups = []
        for i in range(numGaborSpeeds*numGaborAngles*int((self.numAlternationsMax-self.numAlternationsMin)/self.alternationInterval+1)):
            for groupToAdd in groupsFromDifferentFilters[i]:
                addedIntoExistingSet = False
                indexOfSetThatTheGroupToAdWasAddedInto = 0
                iterator = 0
                while iterator < len(combinedGroups):
                    if not addedIntoExistingSet and combinedGroups[iterator].addGroupIfOverlapWithExisting(groupToAdd):
                        addedIntoExistingSet = True
                        indexOfSetThatTheGroupToAdWasAddedInto = iterator
                        iterator+=1
                    elif addedIntoExistingSet and combinedGroups[iterator].groupOverlapsWithExisting(groupToAdd):
                        combinedGroups[indexOfSetThatTheGroupToAdWasAddedInto].addAnotherSetOfGroups(combinedGroups[iterator])
                        del combinedGroups[iterator]
                    else:
                        iterator+=1
                if not addedIntoExistingSet:
                    combinedGroups.append(self.SetOfGroups([groupToAdd], self.groupSizeHeight, self.groupSizeWidth))
                
        combinedGroupsOnlyPeakGabor = []
        for combinedGroup in combinedGroups:
            combinedGroupsOnlyPeakGabor.append(combinedGroup.determineGroupPositionAndMovementVector())
        return combinedGroupsOnlyPeakGabor


    

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

groupSizeWidth = int(input("group size width: "))
groupSizeHeight = int(input("group size height: "))
gaborThreshold = int(input("Gabor Threshold: "))
numAlternationsMin = int(input("Num Alternations Min: "))
numAlternationsMax = int(input("Num Alternations Max: "))
alternationInterval = int(input("Alternation Interval: "))
frameHistorySizeNeeded = numAlternationsMax*6+1
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

gaborFilters = GaborFilters(gaborThreshold,groupSizeHeight,groupSizeWidth, 3, numAlternationsMin, numAlternationsMax, alternationInterval)
for frame in video:
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    grayMat = np.zeros((h,w), dtype = "int16")
    for i in range(h):
        for j in range(w):
            grayMat[i][j] = gray[i,j] - 128

    if numFramesTraversed < frameHistorySizeNeeded:
        grayHistory.append(grayMat)
        frameHistory.append(frame)
        numFramesTraversed += 1
        continue
    else:
        numFramesTraversed += 1
    for i in range(frameHistorySizeNeeded-1):
        grayHistory[frameHistorySizeNeeded-1-i] = grayHistory[frameHistorySizeNeeded-1-i-1]
        frameHistory[frameHistorySizeNeeded-1-i] = frameHistory[frameHistorySizeNeeded-1-i-1]
    grayHistory[0] = grayMat
    frameHistory[0] = frame
    groupsToTrack = gaborFilters.matchGroupsAndDeterminePeakFilter(np.array(grayHistory))
    detections = []
    log.write("currently on frame: " + str(numFramesTraversed)+"\n")
    for group in groupsToTrack:
        detections.append(Detection(points = np.array([group[0][1],group[0][0]]), scores = np.array([1])))
        print("\npoint to add to detections: " + str(group[0][0]) + ", " + str(group[0][1]))
        frameHistory[int(frameHistorySizeNeeded/2)] = cv2.arrowedLine(frameHistory[int(frameHistorySizeNeeded/2)], (int(group[0][1]), int(group[0][0])),(int(group[0][1]+group[1][1]), int(group[0][0]+group[1][0])),(255,0,0),2)
        log.write("group: " + str(group) + "\n")
                        
    print("\nlength of detections: " + str(len(detections)) +"\n")
    tracked_objects = tracker.update(detections=detections)
    for tracked_object in tracked_objects:
        trackedObjectsOut.write(str(numFramesTraversed-3)+","+str(tracked_object.estimate)+","+str(tracked_object.id)+","+str(tracked_object.last_detection)+","+str(tracked_object.last_distance)+","+str(tracked_object.age)+","+str(tracked_object.live_points)+","+str(tracked_object.initializing_id)+"\n")
    draw_tracked_objects(frameHistory[int(frameHistorySizeNeeded/2)], tracked_objects)
    video.write(frameHistory[int(frameHistorySizeNeeded/2)])

log.close()
trackedObjectsOut.close()