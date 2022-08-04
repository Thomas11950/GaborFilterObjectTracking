#ROI code
from binascii import a2b_base64
from pickle import TRUE
import cv2
import numpy as np

from norfair import Detection, Tracker, Video, draw_tracked_objects
from scipy.io import loadmat

#DBSCAN
#https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


from multiprocessing import Process as Processs, Queue, Pipe

class GaborFilter:
    global minPts
    minPts = 4
    def __init__(self, GaborFilterFilepath, gaborThreshold, gaborBaseVector, filterID):
        self.gaborBaseVector = gaborBaseVector
        GaborFilterRaw = loadmat(GaborFilterFilepath)['tmp997']
        self.GaborFilter = self.changeFilterDimensions(GaborFilterRaw)
        self.gaborThreshold = gaborThreshold
        self.GaborFilterFilepath = GaborFilterFilepath
        self.filterID = filterID
    def changeFilterDimensions(self, GaborFilterRaw):
        #we want to change the gabor filter to have time as first dimension, height as second, width as third. Currently, the gabor filter has height, width, time
        out = np.zeros((7, 9, 9), "float")
        for i in range(7):
            for j in range(9):
                for k in range(9):
                    out[i][j][k] = GaborFilterRaw[j][k][i]
        return out
    

    def getGaborFilterValue(self, past7Frames, centerR, centerC):
        partOfFrameHistoryToMultiply = past7Frames[0:7, centerR-4:centerR+5, centerC-4:centerC+5]
        return np.sum(np.multiply(partOfFrameHistoryToMultiply, self.GaborFilter))
    def stageGetGroups(self, grayHistory):
        self.grayHistory = grayHistory
    def getGroups(self):
        grayHistory = self.grayHistory
        h = len(grayHistory[0])
        w = len(grayHistory[0][0])
        pointsExceedingGaborThreshold = []
        gaborValuesOfPointsExceedingGaborThreshold = []
                
        for i in range(int(4),int(h-4)):
            for j in range(int(4), int(w-4)):
                gaborValue = self.getGaborFilterValue(grayHistory, i, j)
                if gaborValue > self.gaborThreshold:
                    pointsExceedingGaborThreshold.append([i,j])
                    gaborValuesOfPointsExceedingGaborThreshold.append(gaborValue)

        #Using k-distances, determine the optimal value of epsilon
        if(len(gaborValuesOfPointsExceedingGaborThreshold) < minPts):
            return []
        df=pd.DataFrame(np.array(pointsExceedingGaborThreshold))
        neighbors = NearestNeighbors(n_neighbors=minPts)
        neighbors_fit = neighbors.fit(df)
        distances, indices = neighbors_fit.kneighbors(df)
        distances = np.sort(distances[:,minPts-1], axis=0)
        #plt.plot(distances)
        #plt.show()
        i = np.arange(len(distances))
        knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
        if knee.knee == None:
            return []
        eps = distances[knee.knee] * 2.5
        print("eps: " + str(eps) + "\n")

        #assign "groups" to all of the Mats using DBSCAN
        dbscan_cluster = DBSCAN(eps=eps, min_samples=minPts)
        dbscan_cluster.fit(pointsExceedingGaborThreshold)
        groupIDs = dbscan_cluster.labels_
        
        #plt.scatter(np.array(pointsExceedingGaborThreshold)[:, 0], np.array(pointsExceedingGaborThreshold)[:, 1], c=groupIDs)

        numClusters=len(set(groupIDs))-(1 if -1 in groupIDs else 0)
        groupMaxRows = np.zeros((numClusters),dtype = "int16")
        groupMinRows = np.full((numClusters),h, dtype = "int16")
        groupMaxCols = np.zeros((numClusters),dtype = "int16")
        groupMinCols = np.full((numClusters),w,dtype = "int16")
        maxGabors = np.zeros((numClusters),dtype = "int16")
        i = 0
        while i < len(pointsExceedingGaborThreshold):
            cluster = groupIDs[i]
            if not cluster == -1:
                if gaborValuesOfPointsExceedingGaborThreshold[i] > maxGabors[cluster]:
                    maxGabors[cluster] = gaborValuesOfPointsExceedingGaborThreshold[i]
                if pointsExceedingGaborThreshold[i][0] < groupMinRows[cluster]:
                    groupMinRows[cluster] = pointsExceedingGaborThreshold[i][0]
                if pointsExceedingGaborThreshold[i][0] > groupMaxRows[cluster]:
                    groupMaxRows[cluster] = pointsExceedingGaborThreshold[i][0]
                if pointsExceedingGaborThreshold[i][1] < groupMinCols[cluster]:
                    groupMinCols[cluster] = pointsExceedingGaborThreshold[i][1]
                if pointsExceedingGaborThreshold[i][1] > groupMaxCols[cluster]:
                    groupMaxCols[cluster] = pointsExceedingGaborThreshold[i][1]
            i+=1
        groups = []
        for i in range(numClusters): 
            groups.append([[(groupMinRows[i]+groupMaxRows[i])/2, (groupMinCols[i]+groupMaxCols[i])/2],[groupMinRows[i], groupMinCols[i], groupMaxRows[i], groupMaxCols[i]],[maxGabors[i]*self.gaborBaseVector[0], maxGabors[i]*self.gaborBaseVector[1]], self.filterID])
            print("group " + str(i) + ": " + str(groups[i]))
        return groups
    
#given a list of gabor filters, return their groups
def returnGroupsFromGaborFilters(gaborFilters, groupsQueue, childConn):
    #sending 0 means not done
    childConn.send(0)
    for gaborFilter in gaborFilters:
        print("convolving a filter: " + gaborFilter.GaborFilterFilepath)
        childConn.send(0)
        groupsQueue.put(gaborFilter.getGroups())
    while True:
        childConn.send(1)

class GaborFilters:
    global numGaborSpeeds
    global numGaborAngles
    numGaborSpeeds = 4;
    numGaborAngles = 8;
    def __init__(self, gaborThreshold, baseVectorSize, numAlternationsMin, numAlternationsMax, alternationInterval, numCoresAvailable, objectExpansionBuffer):
        self.gaborFilters = []
        counter = 0
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
                subListToAdd = []
                for k in range(numAlternationsMin,numAlternationsMax+1, alternationInterval):
                    subListToAdd.append(GaborFilter("Filters/filter_"+str(i+1)+"_"+str(j+1)+".mat", gaborThreshold, [-baseVectorSize * k * movementMagnitude * np.sin(j * 0.3926991 - np.pi/2),baseVectorSize * k* movementMagnitude * np.cos(j * 0.3926991 - np.pi/2)], counter))
                    counter += 1
                listToAdd.append(subListToAdd)
            self.gaborFilters.append(listToAdd)
        self.numAlternationsMin = numAlternationsMin
        self.numAlternationsMax = numAlternationsMax
        self.alternationInterval = alternationInterval
        self.numCoresAvailable = numCoresAvailable
        self.nextFrameROIs = []
        self.firstTimeFindingGroup = True
        self.objectExpansionBuffer = objectExpansionBuffer
    def getGroupsFromDifferentFilters(self,grayHistory):
        ProcessList = []
        groupList = []

        gaborFilters = []
        groupsQueue = Queue()
        for i in range(numGaborSpeeds):
            for j in range(numGaborAngles):
                for k in range(int((self.numAlternationsMax-self.numAlternationsMin)/self.alternationInterval+1)):
                    numAlternations = k * self.alternationInterval + self.numAlternationsMin
                    self.gaborFilters[i][j][k].stageGetGroups(grayHistory[int(len(grayHistory)/2)-3*(int(numAlternations)):int(len(grayHistory)/2)+3*(int(numAlternations))+1:int(numAlternations)])
                    gaborFilters.append(self.gaborFilters[i][j][k])
        numCoresPerFilter = len(gaborFilters) / self.numCoresAvailable
        parentConns = []
        childConns = []
        for i in range(self.numCoresAvailable):
            parentConnToAdd, childConnToAdd = Pipe()
            parentConns.append(parentConnToAdd)
            childConns.append(childConnToAdd)
            ProcessList.append(Processs(target=returnGroupsFromGaborFilters,args=(gaborFilters[int(i*numCoresPerFilter):int((i+1)*numCoresPerFilter):1], groupsQueue, childConnToAdd)))
        for Process in ProcessList:
            Process.start()
        while True:
            #print("main thread is running\n")
            allProcessesDone = True
            i = 0
            for parentConn in parentConns:
                #print("stuck on parentConn " + str(i) + "\n")
                receivedValue = parentConn.recv()
                #print("received value: " + str(receivedValue) + "\n")
                if not receivedValue == 1:
                    allProcessesDone = False
                i+=1
                #print("abt to start on parentConn " + str(i)+"\n")
            #print("broke out of for loop\n")
            if allProcessesDone:
                #print("all processes done\n")
                break
            
            #print("Queue Empty: " + str(groupsQueue.empty())+"\n")
            while not groupsQueue.empty():
                groupList.append(groupsQueue.get())
                #print("Queue size: " + str(groupsQueue.qsize()) + "\n")
        
        while not groupsQueue.empty() or not len(groupList) == numGaborSpeeds*numGaborAngles*int((self.numAlternationsMax-self.numAlternationsMin)/self.alternationInterval+1):
            groupList.append(groupsQueue.get())
            #print("Queue size: " + str(groupsQueue.qsize) + "\n")
        for Process in ProcessList:
            Process.terminate()
        totalGroupWidth = 0
        totalGroupHeight = 0
        totalGroupNum = 0
        #print("groupList: " + str(groupList) + "\n")
        for subGroupList in groupList:
            for group in subGroupList:
                totalGroupWidth += group[1][3] - group[1][1]
                totalGroupHeight += group[1][2] - group[1][0]
                totalGroupNum += 1
        if totalGroupNum == 0:
            return [groupList,0,0]
        else:
            return [groupList, totalGroupHeight/totalGroupNum, totalGroupWidth/totalGroupNum]

    class SetOfGroups:
        def __init__(self, groups, groupSizeHeight, groupSizeWidth, numAlternations):
            self.groups = groups
            self.groupSizeWidth = groupSizeWidth
            self.groupSizeHeight = groupSizeHeight
            boundingGroup = self.getBoundingGroup()
            self.minRow = boundingGroup[0]
            self.maxRow = boundingGroup[1]
            self.minCol = boundingGroup[2]
            self.maxCol = boundingGroup[3]
            self.numAlternations = numAlternations
        def shiftGroupPosition(self, rowShift, colShift):
            for group in self.groups:
                group[0][0] += rowShift
                group[0][1] += colShift
                group[1][0] += rowShift
                group[1][1] += colShift
                group[1][2] += rowShift
                group[1][3] += colShift
            self.minRow += rowShift
            self.maxRow += rowShift
            self.minCol += colShift
            self.maxCol += colShift
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
                #print("in the addGroupIfOverlapWithExisting function, the following group is added: " + str(groupToAdd) + "\n")
                #print("the existing groups in the overlaps are: " + str(self.groups))
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
            totalRowVector = 0
            totalColVector = 0
            #for this set of groups, create an array where index is filter ID and its stored number is the number of groups from that gabor filter ID
            numGroupsWithGaborIDs = np.zeros((numGaborSpeeds*numGaborAngles*self.numAlternations), dtype = "uint8")
            for group in self.groups:
                numGroupsWithGaborIDs[group[3]] += 1
            totalFiltersRepresented = 0
            for numGroupsWithGaborID in numGroupsWithGaborIDs:
                if numGroupsWithGaborID >= 1:
                    totalFiltersRepresented += 1
            for group in self.groups:
                totalRowVector += group[2][0] / numGroupsWithGaborIDs[group[3]]
                totalColVector += group[2][1] / numGroupsWithGaborIDs[group[3]]
            return [[self.minRow, self.minCol, self.maxRow, self.maxCol],[totalRowVector/totalFiltersRepresented,totalColVector/totalFiltersRepresented]]
        def addAnotherSetOfGroups(self, setOfGroupsToAdd):
            for group in setOfGroupsToAdd.getGroups():
                self.addGroup(group)

    def matchGroupsAndDeterminePeakFilter(self,grayHistory):
        height = len(grayHistory[0])
        width = len(grayHistory[0][0])
        if self.firstTimeFindingGroup:
            self.nextFrameROIs.clear()
            self.nextFrameROIs.append([0, 0, height, width])
            self.firstTimeFindingGroup = False
        superCombinedGroups = []
        print("ROIs to convolve: "+str(self.nextFrameROIs)+"\n")
        for nextFrameROI in self.nextFrameROIs:
            grayHistoryToInsertAsParam = grayHistory[0:len(grayHistory), int(nextFrameROI[0]):int(nextFrameROI[2]), int(nextFrameROI[1]):int(nextFrameROI[3])]
            groupsFromDifferentFiltersData = self.getGroupsFromDifferentFilters(grayHistoryToInsertAsParam)
            groupsFromDifferentFilters = groupsFromDifferentFiltersData[0]
            groupSizeHeight = groupsFromDifferentFiltersData[1]
            groupSizeWidth = groupsFromDifferentFiltersData[2]
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
                        combinedGroups.append(self.SetOfGroups([groupToAdd], groupSizeHeight, groupSizeWidth, int((self.numAlternationsMax-self.numAlternationsMin)/self.alternationInterval+1)))
            for combinedGroup in combinedGroups:
                combinedGroup.shiftGroupPosition(nextFrameROI[0],nextFrameROI[1])
                superCombinedGroups.append(combinedGroup)
                
        combinedGroupsOnlyPeakGabor = []
        self.nextFrameROIs = []
        for combinedGroup in superCombinedGroups:
            combinedGroupOnlyPeakGabor = combinedGroup.determineGroupPositionAndMovementVector()
            combinedGroupsOnlyPeakGabor.append(combinedGroupOnlyPeakGabor)
            combinedGroupOnlyPeakGaborHeight = combinedGroupOnlyPeakGabor[0][2] - combinedGroupOnlyPeakGabor[0][0]
            combinedGroupOnlyPeakGaborWidth = combinedGroupOnlyPeakGabor[0][3] - combinedGroupOnlyPeakGabor[0][1]
            nextFrameROI = [combinedGroupOnlyPeakGabor[0][0]+combinedGroupOnlyPeakGabor[1][0]-combinedGroupOnlyPeakGaborHeight*self.objectExpansionBuffer-20,combinedGroupOnlyPeakGabor[0][1]+combinedGroupOnlyPeakGabor[1][1]-combinedGroupOnlyPeakGaborWidth*self.objectExpansionBuffer-20,combinedGroupOnlyPeakGabor[0][2]+combinedGroupOnlyPeakGabor[1][0]+combinedGroupOnlyPeakGaborHeight*self.objectExpansionBuffer+20,combinedGroupOnlyPeakGabor[0][3]+combinedGroupOnlyPeakGabor[1][1]+combinedGroupOnlyPeakGaborWidth*self.objectExpansionBuffer+20]
            if nextFrameROI[0] < 0:
                nextFrameROI[0] = 0
            if nextFrameROI[1] > height:
                nextFrameROI[1] = height
            if nextFrameROI[2] < 0:
                nextFrameROI[2] = 0
            if nextFrameROI[3] > width:
                nextFrameROI[3] = width
            addedToExistingROI = False
            for existingNextFrameROI in self.nextFrameROIs:
                if not ((nextFrameROI[3] < existingNextFrameROI[1] or nextFrameROI[1] > existingNextFrameROI[3]) or (nextFrameROI[0] > existingNextFrameROI[2] or nextFrameROI[2] < existingNextFrameROI[0])):
                    addedToExistingROI = True
                    existingNextFrameROI[2] = max(existingNextFrameROI[2],nextFrameROI[2])
                    existingNextFrameROI[3] = max(existingNextFrameROI[3],nextFrameROI[3])
                    existingNextFrameROI[0] = min(existingNextFrameROI[0],nextFrameROI[0])
                    existingNextFrameROI[1] = min(existingNextFrameROI[1],nextFrameROI[1])
            if not addedToExistingROI:
                self.nextFrameROIs.append(nextFrameROI)
            print("final group: " + str(combinedGroupOnlyPeakGabor) + "\n")

        return combinedGroupsOnlyPeakGabor
    def getNextFrameROIs(self):
        return self.nextFrameROIs


    

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

gaborThreshold = int(input("Gabor Threshold: "))
numAlternationsMin = int(input("Num Alternations Min: "))
numAlternationsMax = int(input("Num Alternations Max: "))
alternationInterval = int(input("Alternation Interval: "))
frameHistorySizeNeeded = numAlternationsMax*6+1
numCores = int(input("Num Cores Available: "))
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

gaborFilters = GaborFilters(gaborThreshold, 0.05, numAlternationsMin, numAlternationsMax, alternationInterval, numCores, 0.4)
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
        groupCol = (group[0][3] + group[0][1])/2
        groupRow = (group[0][2] + group[0][0])/2
        detections.append(Detection(points = np.array([groupCol,groupRow]), scores = np.array([1])))
        print("\npoint to add to detections: " + str(groupRow) + ", " + str(groupCol))
        frameHistory[int(frameHistorySizeNeeded/2)] = cv2.arrowedLine(frameHistory[int(frameHistorySizeNeeded/2)], (int(groupCol), int(groupRow)),(int(groupCol+group[1][1]), int(groupRow+group[1][0])),(255,0,0),2)
        frameHistory[int(frameHistorySizeNeeded/2)] = cv2.rectangle(frameHistory[int(frameHistorySizeNeeded/2)], (int(group[0][1]), int(group[0][0])),(int(group[0][3]),int(group[0][2])),(0,255,122),2)
        log.write("group: " + str(group) + "\n")
    for nextFrameROI in gaborFilters.getNextFrameROIs():
        frameHistory[int(frameHistorySizeNeeded/2)] = cv2.rectangle(frameHistory[int(frameHistorySizeNeeded/2)], (int(nextFrameROI[1]),int(nextFrameROI[0])), (int(nextFrameROI[3]),int(nextFrameROI[2])),(255,255,255),2)
    print("\nlength of detections: " + str(len(detections)) +"\n")
    tracked_objects = tracker.update(detections=detections)
    for tracked_object in tracked_objects:
        trackedObjectsOut.write(str(numFramesTraversed-3)+","+str(tracked_object.estimate)+","+str(tracked_object.id)+","+str(tracked_object.last_detection)+","+str(tracked_object.last_distance)+","+str(tracked_object.age)+","+str(tracked_object.live_points)+","+str(tracked_object.initializing_id)+"\n")
    draw_tracked_objects(frameHistory[int(frameHistorySizeNeeded/2)], tracked_objects)
    video.write(frameHistory[int(frameHistorySizeNeeded/2)])

log.close()
trackedObjectsOut.close()