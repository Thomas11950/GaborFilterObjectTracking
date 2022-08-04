#Remember objects from object history, edit constants, fix sorting
from binascii import a2b_base64
from ctypes import Union
from pickle import TRUE
from readline import append_history_file
import cv2
import numpy as np
#Norfair
from norfair import Detection, Tracker, Video, draw_tracked_objects
from scipy.io import loadmat
import math
#DBSCAN
#https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/
import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import v_measure_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

from multiprocessing import Process as Processs, Queue, Pipe

import networkx as nx
import community
import warnings
warnings.filterwarnings("ignore")
#STSC
from functools import reduce
from scipy.optimize import minimize

from itertools import groupby
from scipy.linalg import eigh, inv, sqrtm


def reformat_result(cluster_labels, n):
    zipped_data = zip(cluster_labels, range(n))
    zipped_data = sorted(zipped_data, key=lambda x: x[0])
    grouped_feature_id = [[j[1] for j in i[1]] for i in groupby(zipped_data, lambda x: x[0])]
    return grouped_feature_id


def affinity_to_lap_to_eig(affinity):
    tril = np.tril(affinity, k=-1)
    a = tril + tril.T
    d = np.diag(a.sum(axis=0))
    dd = inv(sqrtm(d))
    l = dd.dot(a).dot(dd)
    w, v = eigh(l)
    return w, v


def get_min_max(w, min_n_cluster, max_n_cluster):
    if min_n_cluster is None:
        min_n_cluster = 2
    if max_n_cluster is None:
        max_n_cluster = np.sum(w > 0)
        if max_n_cluster < 2:
            max_n_cluster = 2
    if min_n_cluster > max_n_cluster:
        raise ValueError('min_n_cluster should be smaller than max_n_cluster')
    return min_n_cluster, max_n_cluster


def generate_Givens_rotation(i, j, theta, size):
    g = np.eye(size)
    c = np.cos(theta)
    s = np.sin(theta)
    g[i, i] = c
    g[j, j] = c
    if i > j:
        g[j, i] = -s
        g[i, j] = s
    elif i < j:
        g[j, i] = s
        g[i, j] = -s
    else:
        raise ValueError('i and j must be different')
    return g


def generate_Givens_rotation_gradient(i, j, theta, size):
    g = np.zeros((size, size))
    c = np.cos(theta)
    s = np.sin(theta)
    g[i, i] = -s
    g[j, j] = -s
    if i > j:
        g[j, i] = -c
        g[i, j] = c
    elif i < j:
        g[j, i] = c
        g[i, j] = -c
    else:
        raise ValueError('i and j must be different')
    return g


def generate_U_list(ij_list, theta_list, size):
    return [generate_Givens_rotation(ij[0], ij[1], theta, size)
            for ij, theta in zip(ij_list, theta_list)]


def generate_V_list(ij_list, theta_list, size):
    return [generate_Givens_rotation_gradient(ij[0], ij[1], theta, size)
            for ij, theta in zip(ij_list, theta_list)]


def get_U_ab(a, b, U_list, K):
    I = np.eye(U_list[0].shape[0])
    if a == b:
        if a < K and a != 0:
            return U_list[a]
        else:
            return I
    elif a > b:
        return I
    else:
        return reduce(np.dot, U_list[a:b], I)


def get_A_matrix(X, U_list, V_list, k, K):
    Ul = get_U_ab(0, k, U_list, K)
    V = V_list[k]
    Ur = get_U_ab(k + 1, K, U_list, K)
    return X.dot(Ul).dot(V).dot(Ur)


def get_rotation_matrix(X, C):
    ij_list = [(i, j) for i in range(C) for j in range(C) if i < j]
    K = len(ij_list)

    def cost_and_grad(theta_list):
        U_list = generate_U_list(ij_list, theta_list, C)
        V_list = generate_V_list(ij_list, theta_list, C)
        R = reduce(np.dot, U_list, np.eye(C))
        Z = X.dot(R)
        mi = np.argmax(Z, axis=1)
        M = np.choose(mi, Z.T).reshape(-1, 1)
        cost = np.sum((Z / M) ** 2)
        grad = np.zeros(K)
        for k in range(K):
            A = get_A_matrix(X, U_list, V_list, k, K)
            tmp = (Z / (M ** 2)) * A
            tmp -= ((Z ** 2) / (M ** 3)) * (np.choose(mi, A.T).reshape(-1, 1))
            tmp = 2 * np.sum(tmp)
            grad[k] = tmp

        return cost, grad

    theta_list_init = np.array([0.0] * int(C * (C - 1) / 2))
    opt = minimize(cost_and_grad,
                   x0=theta_list_init,
                   method='CG',
                   jac=True,
                   options={'disp': False})
    return opt.fun, reduce(np.dot, generate_U_list(ij_list, opt.x, C), np.eye(C))

def self_tuning_spectral_clustering(affinity, get_rotation_matrix, min_n_cluster=None, max_n_cluster=None):
    w, v = affinity_to_lap_to_eig(affinity)
    print("reached checkpoint 2.1")
    min_n_cluster, max_n_cluster = get_min_max(w, min_n_cluster, max_n_cluster)
    re = []
    firstLoop = True
    minCost = 0
    nClustersWithMinCost = 0
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r = get_rotation_matrix(x, c)
        re.append((cost, x.dot(r)))
        print('n_cluster: %d \t cost: %f' % (c, cost))
        if firstLoop:
            minCost = cost
            nClustersWithMinCost = c
            firstLoop = False
        else:
            if cost < minCost or abs(cost-minCost)/minCost < 0.01:
                minCost = cost
                nClustersWithMinCost = c
    return nClustersWithMinCost

def self_tuning_spectral_clustering_np(affinity, min_n_cluster=None, max_n_cluster=None):
    return self_tuning_spectral_clustering(affinity, get_rotation_matrix, min_n_cluster, max_n_cluster)

#Ciortan Madalina

from scipy.spatial.distance import pdist, squareform
def getAffinityMatrix(coordinates, k = 7):
    """
    Calculate affinity matrix based on input coordinates matrix and the numeber
    of nearest neighbours.
    
    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates)) 
    
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix*15)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


#helper fxns
def getOverlapArea(rect1, rect2):
    overlap = [0,0,0,0]
    overlap[0] = max(rect1[0],rect2[0])
    overlap[1] = max(rect1[1],rect2[1])
    overlap[2] = min(rect1[2],rect2[2])
    overlap[3] = min(rect1[3],rect2[3])
    overlapHeight = overlap[2] - overlap[0]
    if overlapHeight < 0:
        overlapHeight = 0
    overlapWidth = overlap[3] - overlap[1]
    if overlapWidth < 0:
        overlapWidth = 0
    overlapArea = overlapHeight * overlapWidth
    return overlapArea

def getRectangularOverlap(rect1, rect2):
    overlapArea = getOverlapArea(rect1, rect2)
    union = (rect1[2] - rect1[0]) * (rect1[3]-rect1[1]) + (rect2[2] - rect2[0]) * (rect2[3]-rect2[1]) - overlapArea
    return overlapArea/union

def getRectangularSimilarity(rect1, rect2):
    overlapArea = getOverlapArea(rect1, rect2)
    areaRect1 = (rect1[2] - rect1[0]) * (rect1[3]-rect1[1])
    areaRect2 = (rect2[2] - rect2[0]) * (rect2[3]-rect2[1])
    return overlapArea/min(areaRect1, areaRect2)

#helper fxns silhouette
def calculate_silhouette_value(X, labels, pointToCalculate, labelOfPointToCalculate):
    aCount = 0
    aTotal = 0
    bCount = 0
    bTotal = 0
    for i in range(len(X)):
        if not (X[i][0] == pointToCalculate[0] and X[i][1] == pointToCalculate[1]):
            dist = math.hypot(pointToCalculate[0]-X[i][0],pointToCalculate[1]-X[i][1])
            if labelOfPointToCalculate == labels[i]:
                aTotal += dist
                aCount += 1
            else:
                bTotal += dist
                bCount += 1
    aMean = aTotal/aCount
    bMean = bTotal/bCount
    print("aMean: " + str(aMean) + ", bMean: " + str(bMean)+"\n")
    return (bMean-aMean)/max(bMean,aMean)

def calculate_silhouette_samples(X, labels):
    toReturn = []
    for i in range(len(X)):
        toReturn.append(calculate_silhouette_value(X, labels, X[i], labels[i]))
    #toReturn = calculate_silhouette_value(X, labels, X, labels)
    return np.array(toReturn)

def calculate_silhouette_coeff(X, labels, numGroups):
    groupTotals = np.zeros((numGroups),dtype="float")
    groupCounts = np.zeros((numGroups),dtype="uint32")
    silhouette_values = silhouette_samples(X=np.array(X), labels=labels)
    for i in range(len(labels)):
        sqrtOfSilhouette = abs(silhouette_values[i])**(1.0/12)
        silhouette_value_to_add = sqrtOfSilhouette
        if silhouette_values[i] < 0:
            silhouette_value_to_add *= -1
        groupTotals[labels[i]]+=silhouette_value_to_add
        groupCounts[labels[i]]+=1
    groupAverages = groupTotals/groupCounts
    #print(str(groupAverages)+"\n")
    return np.mean(groupAverages)

class GaborFilter:
    global minPts
    minPts = 4
    global clusteringMethod
    clusteringMethod = "DBSCAN"
    global minKMeansSweep, maxKMeansSweep
    minKMeansSweep = 2
    maxKMeansSweep = 10
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

        groupIDs = []
        if clusteringMethod == "DBSCAN":
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
            eps = distances[knee.knee] * 4
            #print("eps: " + str(eps) + "\n")

            #assign "groups" to all of the Mats using DBSCAN
            dbscan_cluster = DBSCAN(eps=eps, min_samples=minPts)
            dbscan_cluster.fit(pointsExceedingGaborThreshold)
            groupIDs = dbscan_cluster.labels_
        
            #plt.scatter(np.array(pointsExceedingGaborThreshold)[:, 0], np.array(pointsExceedingGaborThreshold)[:, 1], c=groupIDs)
        elif clusteringMethod == "SPECTRAL":
            affinity_matrix = getAffinityMatrix(pointsExceedingGaborThreshold,k = int(len(pointsExceedingGaborThreshold)-1))
            n_clusters = self_tuning_spectral_clustering_np(affinity_matrix, max_n_cluster = 7)
            print("n_clusters: " + str(n_clusters)+"\n")
            spectral_cluster = SpectralClustering(n_clusters=n_clusters,affinity = 'precomputed')
            spectral_cluster.fit_predict(X=affinity_matrix)
            groupIDs = spectral_cluster.labels_
            
            plt.scatter(np.array(pointsExceedingGaborThreshold)[:, 0], np.array(pointsExceedingGaborThreshold)[:, 1], c=groupIDs)
            plt.show()
        elif clusteringMethod == "KMEANS_SILHOUETTE":
            maxSilhouetteCoeff = 0
            groupingWithMaxSilhouetteCoeff = []
            for i in range(minKMeansSweep, maxKMeansSweep+1):
                KMeans_cluster = KMeans(n_clusters=i, init = 'k-means++')
                KMeans_cluster.fit_predict(X=pointsExceedingGaborThreshold)
                groupIDsToSweep = KMeans_cluster.labels_
                silhouetteCoeff = calculate_silhouette_coeff(X = pointsExceedingGaborThreshold, labels = groupIDsToSweep, numGroups = i)
                #print("k-value: " + str(i) +", silhouette score: " + str(silhouetteCoeff) + "\n")
                if silhouetteCoeff > maxSilhouetteCoeff:
                    maxSilhouetteCoeff = silhouetteCoeff
                    groupingWithMaxSilhouetteCoeff = groupIDsToSweep
            print("max silhouette coeff: " + str(maxSilhouetteCoeff) + "\n")
            if maxSilhouetteCoeff > 0.9:
                groupIDs = groupingWithMaxSilhouetteCoeff
            else:
                groupIDs = np.zeros((len(pointsExceedingGaborThreshold)), dtype = "uint32")
            plt.scatter(np.array(pointsExceedingGaborThreshold)[:, 0], np.array(pointsExceedingGaborThreshold)[:, 1], c=groupIDs)
            plt.show()
        else:
            distortions = []
            K = range(1,10)
            for k in K:
                kmeanModel = KMeans(n_clusters=k)
                kmeanModel.fit(pointsExceedingGaborThreshold)
                distortions.append(kmeanModel.inertia_)
            plt.plot(K, distortions, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Distortion')
            plt.title('The Elbow Method showing the optimal k')
            plt.show()
            groupIDs = np.full((len(pointsExceedingGaborThreshold)),0,dtype = "uint32")


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
            #print("group " + str(i) + ": " + str(groups[i]))
        return groups
    
#given a list of gabor filters, return their groups
def returnGroupsFromGaborFilters(gaborFilters, groupsQueue, childConn):
    #sending 0 means not done
    childConn.send(0)
    for gaborFilter in gaborFilters:
        childConn.send(0)
        groupsQueue.put(gaborFilter.getGroups())
    while True:
        childConn.send(1)

#Return the area of a set of groups
def getSetOfGroupArea(setOfGroups):
    return setOfGroups.determineGroupArea()
class GaborFilters:
    global numGaborSpeeds
    global numGaborAngles
    global sizeOfObjectHistory
    global checkEntireFrameInterval
    numGaborSpeeds = 4
    numGaborAngles = 8
    sizeOfObjectHistory = 100
    checkEntireFrameInterval = 20
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
        self.objectHistoriesForGivenID = []
        self.usedIDs = []
        self.countMatchGroupsAndDeterminePeakFilter = 1
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
            self.groupSetID = -1
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
            if groupToAdd[1][0] < self.minRow:
                self.minRow = groupToAdd[1][0]
            if groupToAdd[1][2] > self.maxRow:
                self.maxRow = groupToAdd[1][2]
            if groupToAdd[1][1] < self.minCol:
                self.minCol = groupToAdd[1][1]
            if groupToAdd[1][3] > self.maxCol:
                self.maxCol = groupToAdd[1][3]
        def determineGroupBoundingBox(self):
            return [self.minRow, self.minCol, self.maxRow, self.maxCol]
        def determineGroupPosition(self):
            return [(self.minRow+self.maxRow)/2,(self.minCol+self.maxCol)/2]
        def determineGroupSize(self):
            return [self.maxRow - self.minRow, self.maxCol - self.minCol]
        def determineGroupArea(self):
            return (self.maxRow - self.minRow) * (self.maxCol - self.minCol)
        def determineGroupBoundingBoxAndMovementVectorAndID(self):
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
            return [[self.minRow, self.minCol, self.maxRow, self.maxCol],[totalRowVector/totalFiltersRepresented,totalColVector/totalFiltersRepresented],self.groupSetID]
        def addAnotherSetOfGroups(self, setOfGroupsToAdd):
            for group in setOfGroupsToAdd.getGroups():
                self.addGroup(group)
        def setGroupSetID(self, ID):
            self.groupSetID = ID
        def getGroupSetID(self):
            return self.groupSetID
        #returns an array of the groups (other than itself) the group is split into
        def splitGroups(self):
            #construct the networkX graph
            #start with the nodes
            G = nx.Graph()
            groupsArr = np.array(self.groups)
            for i in range(len(groupsArr)):
                G.add_node(i)
            for i in range(len(groupsArr)):
                for j in range(i+1,len(groupsArr)):
                    rectangularSimilarity = getRectangularSimilarity(groupsArr[i][1],groupsArr[j][1])
                    if(rectangularSimilarity > 0):
                        G.add_edge(i,j,weight=rectangularSimilarity)
            
            #calculate the degree of each node
            nodeDegrees = np.zeros((len(groupsArr)), dtype = "float")
            for i in range(len(groupsArr)):
                nodeDegrees[i] = G.degree(nbunch = i, weight = "weight")
            """
            xArr = np.linspace(0,len(nodeDegrees),len(nodeDegrees))
            nodeDegreesEdited = np.append(np.sort(nodeDegrees,axis=0),[])
            print("node degrees: " + str(nodeDegreesEdited))
            plt.scatter(xArr, nodeDegreesEdited)
            polyfit_Coeffs = np.polyfit(xArr, nodeDegreesEdited, 3)
            plt.plot(xArr, polyfit_Coeffs[0]*xArr**3 + polyfit_Coeffs[1]*xArr**2 + polyfit_Coeffs[2]*xArr**1+polyfit_Coeffs[3])
            plt.plot(xArr, np.full((len(xArr)), len(self.groups), dtype = "float"))
            plt.plot(xArr, np.full((len(xArr)), len(self.groups)*0.6, dtype = "float"))
            
            plt.plot(xArr, np.full((len(xArr)),nodeDegreesEdited[int(len(xArr)*0.9)], dtype = "float"))
            
            plt.plot(xArr, np.full((len(xArr)),nodeDegreesEdited[int(len(xArr)*0.9)]/2, dtype = "float"))
            plt.show()
            """
            #calculate whether 90% of all degree values fall below 65% of the number of groups in the array. If so, remove all groups that have degree values above 80% of the number of groups in the array
            nodeDegreesSorted = np.sort(nodeDegrees,axis=0)
            sixtyPercentOfGroupNum = len(self.groups)*0.65
            nodeDegrees90thPercentile = nodeDegreesSorted[int(len(nodeDegreesSorted)*0.9)]
            if nodeDegrees90thPercentile < sixtyPercentOfGroupNum:
                for i in range(len(nodeDegrees)):
                    if nodeDegrees[i] > len(self.groups)*0.8:
                        G.remove_node(i)

            #calculate communities
            
            communities = community.best_partition(G, resolution = 0.001)
            print("communities: " + str(communities) + "\n")
            #calculate the number of communities
            maxCommunity = 0
            for i in communities:
                if communities[i] > maxCommunity:
                    maxCommunity = communities[i]
            numCommunities = maxCommunity+1
            #calculate commmunity sizes and a list that contains lists of the groups in the community, with the index of the list corresponding to the community number
            communitySizes = np.zeros((numCommunities), dtype = "uint32")
            groupsInCommunities = []
            for i in range(numCommunities):
                groupsInCommunities.append([])
            for i in communities:
                communitySizes[communities[i]]+=1
                groupsInCommunities[communities[i]].append(i)
            #calculate size and community number of the largest community
            maxCommunitySize = 0
            maxCommunitySizeIndex = 0
            for i in range(len(communitySizes)):
                if communitySizes[i] > maxCommunitySize:
                    maxCommunitySize = communitySizes[i]
                    maxCommunitySizeIndex = i
            #fold communities that are less than 50% of the size of the largest community into the community that has the fewest cuts
            communitiesToRemove = []
            communitiesToKeep = []
            for i in range(len(communitySizes)):
                if communitySizes[i] < 0.5 * maxCommunitySize:
                    communitiesToRemove.append(i)
                else:
                    communitiesToKeep.append(i)
                    
            newSetsOfGroups = {}
            numberOfGroupsDeletedFromSelf = 0
            for i in range(len(groupsArr)):
                if i in communities:
                    if not communities[i] == maxCommunitySizeIndex:
                        if communities[i] in newSetsOfGroups:
                            newSetsOfGroups[communities[i]].addGroup(groupsArr[i])
                        else:
                            newSetsOfGroups[communities[i]] = self.__class__([groupsArr[i]], self.groupSizeHeight, self.groupSizeWidth, self.numAlternations)
                        self.groups.pop(i-numberOfGroupsDeletedFromSelf)
                        numberOfGroupsDeletedFromSelf += 1
                else:
                    self.groups.pop(i-numberOfGroupsDeletedFromSelf)
                    numberOfGroupsDeletedFromSelf += 1
            
            for communityToRemove in communitiesToRemove:
                communityToKeepWithLowestCutSize = 0
                lowestCutSize = 0
                firstLoop = True
                for communityToKeep in communitiesToKeep:
                    cut_Size = nx.cut_size(G, groupsInCommunities[communityToKeep], groupsInCommunities[communityToRemove], weight = "weight")
                    if firstLoop:
                        communityToKeepWithLowestCutSize = communityToKeep
                        lowestCutSize = cut_Size
                        firstLoop = False
                    else:
                        if cut_Size < lowestCutSize:
                            communityToKeepWithLowestCutSize = communityToKeep
                            lowestCutSize = cut_Size
                if communityToKeepWithLowestCutSize == maxCommunitySizeIndex:
                    self.addAnotherSetOfGroups(newSetsOfGroups[communityToRemove])
                else:
                    newSetsOfGroups[communityToKeepWithLowestCutSize].addAnotherSetOfGroups(newSetsOfGroups[communityToRemove])
                newSetsOfGroups.pop(communityToRemove)

            boundingGroup = self.getBoundingGroup()
            self.minRow = boundingGroup[0]
            self.maxRow = boundingGroup[1]
            self.minCol = boundingGroup[2]
            self.maxCol = boundingGroup[3]

            
            return newSetsOfGroups.values()

    class ROI:
        def __init__(self, ROIposition, containedIDs, predictedBoundingBoxesForContainedIDs):
            self.ROIposition = ROIposition
            self.containedIDs = containedIDs
            self.predictedBoundingBoxesForContainedIDs = predictedBoundingBoxesForContainedIDs
            self.alreadyRetrievedIDs = []
        def __str__(self):
            return str(self.ROIposition)
        def addObjectToROI(self, ID, boundingBoxForContainedID):
            self.containedIDs.append(ID)
            self.predictedBoundingBoxesForContainedIDs.append(boundingBoxForContainedID)
        def getPos(self):
            return self.ROIposition
        def getContainedIDs(self):
            return self.containedIDs
        def getPredictedBoundingBoxesForContainedIDs(self):
            return self.predictedBoundingBoxesForContainedIDs
        def getIDs(self, groups):
            #create matrix that matches the overlap between every single input group and every single group inside the ROI
            dictGroupIndicesToID = {}
            if len(groups) == 0 or len(self.predictedBoundingBoxesForContainedIDs) == 0:
                return dictGroupIndicesToID
            overlapValues = np.zeros((len(groups), len(self.predictedBoundingBoxesForContainedIDs)),dtype = "float")
            for i in range(len(groups)):
                for j in range(len(self.predictedBoundingBoxesForContainedIDs)):
                    overlapValues[i][j] = getRectangularOverlap(groups[i].determineGroupBoundingBox(), self.predictedBoundingBoxesForContainedIDs[j])
            print("overlap values: " + str(overlapValues)+"\n")
            while not np.amax(overlapValues) == 0:
                rowOfMax, colOfMax = np.unravel_index(overlapValues.argmax(), overlapValues.shape)
                dictGroupIndicesToID[rowOfMax] = self.containedIDs[colOfMax]
                for i in range(len(self.predictedBoundingBoxesForContainedIDs)):
                    overlapValues[rowOfMax][i] = 0
                for i in range(len(groups)):
                    overlapValues[i][colOfMax] = 0
            return dictGroupIndicesToID
            
        def getID(self, group):
            maxOverlap = 0
            maxDistIndex = -1
            i = 0
            while i < len(self.predictedBoundingBoxesForContainedIDs):
                overlap = getRectangularOverlap(self.predictedBoundingBoxesForContainedIDs[i], group.determineGroupBoundingBox())
                if overlap > 0.5 and overlap > maxOverlap and not self.containedIDs[i] in self.alreadyRetrievedIDs:
                    maxOverlap = overlap
                    maxDistIndex = i
                i+=1
            if maxDistIndex == -1:
                return -1
            else:
                self.alreadyRetrievedIDs.append(self.containedIDs[maxDistIndex])
                return self.containedIDs[maxDistIndex]
        def resetAlreadyRetrievedIDs(self):
            self.alreadyRetrievedIDs = []
        def normalizeROIToHeightAndWidth(self, height, width):
            if self.ROIposition[0] < 0:
                self.ROIposition[0] = 0
            if self.ROIposition[1] > height:
                self.ROIposition[1] = height
            if self.ROIposition[2] < 0:
                self.ROIposition[2] = 0
            if self.ROIposition[3] > width:
                self.ROIposition[3] = width
        def checkOtherROIOverlap(self, otherROI):
            return not ((self.ROIposition[3] < otherROI.getPos()[1] or self.ROIposition[1] > otherROI.getPos()[3]) or (self.ROIposition[0] > otherROI.getPos()[2] or self.ROIposition[2] < otherROI.getPos()[0]))
        def combineOtherROI(self, otherROI):
            self.ROIposition[2] = max(self.ROIposition[2],otherROI.getPos()[2])
            self.ROIposition[3] = max(self.ROIposition[3],otherROI.getPos()[3])
            self.ROIposition[0] = min(self.ROIposition[0],otherROI.getPos()[0])
            self.ROIposition[1] = min(self.ROIposition[1],otherROI.getPos()[1])
            for otherROIContainedID in otherROI.getContainedIDs():
                self.containedIDs.append(otherROIContainedID)
            for otherROIPredictedBoundingBoxesForContainedIDs in otherROI.getPredictedBoundingBoxesForContainedIDs():
                self.predictedBoundingBoxesForContainedIDs.append(otherROIPredictedBoundingBoxesForContainedIDs)

    class ObjectHistory:
        def __init__(self, ID, initialObjectSize, initialObjectPosition, initialVector):
            self.ID = ID
            self.sizeHistory = np.full((sizeOfObjectHistory,2),initialObjectSize, dtype = "float")
            self.positionHistory = np.full((sizeOfObjectHistory,4),initialObjectPosition,dtype = "float")
            self.vectorHistory = np.full((sizeOfObjectHistory,2),initialVector,dtype = "float")
            self.numFramesWithoutBeingSeen = 0
        def getID(self):
            return self.ID
        def addToSizeAndPositionAndVectorHistory(self, sizeToAdd, positionToAdd, vectorToAdd):
            totalSizeSimilarity = 0
            for i in range(sizeOfObjectHistory):
                totalSizeSimilarity += (sizeToAdd[0]-self.sizeHistory[i][0])/self.sizeHistory[i][0] + (sizeToAdd[1]-self.sizeHistory[i][1])/self.sizeHistory[i][1]
            avgSizeSimilarity = totalSizeSimilarity/sizeOfObjectHistory
            if abs(avgSizeSimilarity) < 0.7:
                return

            for i in range(sizeOfObjectHistory-1):
                self.sizeHistory[sizeOfObjectHistory-1-i] = self.sizeHistory[sizeOfObjectHistory-2-i]
                self.positionHistory[sizeOfObjectHistory-1-i] = self.positionHistory[sizeOfObjectHistory-2-i]
                self.vectorHistory[sizeOfObjectHistory-1-i] = self.vectorHistory[sizeOfObjectHistory-2-i]
            self.sizeHistory[0] = sizeToAdd
            self.positionHistory[0] = positionToAdd
            self.vectorHistory[0] = vectorToAdd
        def getAvgSizeAcrossHistory(self):
            sumHeight = 0
            sumWidth = 0
            for size in self.sizeHistory:
                sumHeight += size[0]
                sumWidth += size[1]
            return np.array([sumHeight/sizeOfObjectHistory, sumWidth/sizeOfObjectHistory])
        def getAvgPositionAcrossHistory(self):
            sumRow = 0
            sumCol = 0
            for position in self.positionHistory:
                sumRow += position[0]
                sumCol += position[1]
            return np.array([sumRow/sizeOfObjectHistory,sumCol/sizeOfObjectHistory])
        def incrementNumFramesWithoutBeingSeen(self):
            self.numFramesWithoutBeingSeen+=1
        def resetNumFramesWithoutBeingSeen(self):
            self.numFramesWithoutBeingSeen = 0
        def predictedPositionAfterNotBeingSeen(self):
            return np.array([self.positionHistory[0][0]+self.vectorHistory[0][0]*self.numFramesWithoutBeingSeen, self.positionHistory[0][1]+self.vectorHistory[0][1]*self.numFramesWithoutBeingSeen, self.positionHistory[0][2]+self.vectorHistory[0][0]*self.numFramesWithoutBeingSeen, self.positionHistory[0][3]+self.vectorHistory[0][1]*self.numFramesWithoutBeingSeen])
        
    def matchGroupsAndDeterminePeakFilter(self,grayHistory):
        height = len(grayHistory[0])
        width = len(grayHistory[0][0])
        if self.firstTimeFindingGroup:
            self.nextFrameROIs.clear()
            self.nextFrameROIs.append(self.ROI([0, 0, height, width],[],[]))
            self.firstTimeFindingGroup = False
        superCombinedGroups = []
        print("ROIs to convolve: ")
        for nextFrameROI in self.nextFrameROIs:
            print(str(nextFrameROI)+", ")
        print("\n")
        for nextFrameROI in self.nextFrameROIs:
            for ID in nextFrameROI.getContainedIDs():
                if not ID in self.usedIDs:
                    self.usedIDs.append(ID)
        for nextFrameROI in self.nextFrameROIs:
            nextFrameROIPos = nextFrameROI.getPos()
            grayHistoryToInsertAsParam = grayHistory[0:len(grayHistory), int(nextFrameROIPos[0]):int(nextFrameROIPos[2]), int(nextFrameROIPos[1]):int(nextFrameROIPos[3])]
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
            combinedGroupsInitialLen = len(combinedGroups)
            counter = 0
            for combinedGroup in combinedGroups:
                if not counter < combinedGroupsInitialLen:
                    break
                newGroups = combinedGroup.splitGroups()
                for newGroup in newGroups:
                    combinedGroups.append(newGroup)
                counter+=1
            combinedGroups.sort(key = getSetOfGroupArea, reverse = True)
            print("group areas: ")
            for combinedGroup in combinedGroups:
                print(str(combinedGroup.determineGroupArea()) + ", ")
            print("\n")
            """
            for combinedGroup in combinedGroups:
                combinedGroup.shiftGroupPosition(nextFrameROIPos[0],nextFrameROIPos[1])
                groupSetID = nextFrameROI.getID(combinedGroup)
                if groupSetID == -1:
                    objectFoundFromHistory = False
                    for objectHistoryForGivenID in self.objectHistoriesForGivenID:
                        if getRectangularOverlap(objectHistoryForGivenID.predictedPositionAfterNotBeingSeen(), combinedGroup.determineGroupBoundingBox()) > 0.5:
                            objectFoundFromHistory = True
                            groupSetID = objectHistoryForGivenID.getID()
                            break
                    if not objectFoundFromHistory:
                        j = 0
                        while True:
                            if not j in self.usedIDs:
                                groupSetID = j
                                self.usedIDs.append(j)
                                break
                            j+=1
                combinedGroup.setGroupSetID(groupSetID)
                superCombinedGroups.append(combinedGroup)
            nextFrameROI.resetAlreadyRetrievedIDs()
            """
            for combinedGroup in combinedGroups:
                combinedGroup.shiftGroupPosition(nextFrameROIPos[0],nextFrameROIPos[1])
            groupSetIDs = nextFrameROI.getIDs(combinedGroups)
            print("groupSetIDs: "+str(groupSetIDs)+"\n")
            for groupSetIndex in groupSetIDs:
                combinedGroups[groupSetIndex].setGroupSetID(groupSetIDs[groupSetIndex])
            for i in range(len(combinedGroups)):
                if not i in groupSetIDs:
                    objectFoundFromHistory = False
                    for objectHistoryForGivenID in self.objectHistoriesForGivenID:
                        if getRectangularOverlap(objectHistoryForGivenID.predictedPositionAfterNotBeingSeen(), combinedGroups[i].determineGroupBoundingBox()) > 0.5:
                            objectFoundFromHistory = True
                            groupSetID = objectHistoryForGivenID.getID()
                            break
                    if not objectFoundFromHistory:
                        j = 0
                        while True:
                            if not j in self.usedIDs:
                                groupSetID = j
                                self.usedIDs.append(j)
                                break
                            j+=1
                    combinedGroups[i].setGroupSetID(groupSetID)
            for combinedGroup in combinedGroups:
                superCombinedGroups.append(combinedGroup)
            
        for objectHistoryForGivenID in self.objectHistoriesForGivenID:
            objectInHistorySeen = False
            for combinedGroup in superCombinedGroups:
                if objectHistoryForGivenID.getID() == combinedGroup.getGroupSetID():
                    objectInHistorySeen = True
            if not objectInHistorySeen:
                objectHistoryForGivenID.incrementNumFramesWithoutBeingSeen()
            else:
                objectHistoryForGivenID.resetNumFramesWithoutBeingSeen()

        combinedGroupsOnlyPeakGabor = []
        combinedGroupsAllGroups = []
        self.nextFrameROIs = []
        if self.countMatchGroupsAndDeterminePeakFilter % checkEntireFrameInterval == 0:
            self.nextFrameROIs.append(self.ROI([0, 0, height, width],[],[]))
            for combinedGroup in superCombinedGroups:
                combinedGroupOnlyPeakGabor = combinedGroup.determineGroupBoundingBoxAndMovementVectorAndID()
                combinedGroupsOnlyPeakGabor.append(combinedGroupOnlyPeakGabor)
                combinedGroupsAllGroups.append(combinedGroup.getGroups())
                combinedGroupPeakGaborNextPredictedBoundingBoxPosition = [combinedGroupOnlyPeakGabor[0][0]+combinedGroupOnlyPeakGabor[1][0],combinedGroupOnlyPeakGabor[0][1]+combinedGroupOnlyPeakGabor[1][1],combinedGroupOnlyPeakGabor[0][2]+combinedGroupOnlyPeakGabor[1][0],combinedGroupOnlyPeakGabor[0][3]+combinedGroupOnlyPeakGabor[1][1]]
                sizeIsSignificantDeviationFromHistoricalSize = False
                objectHistoryAvgSize = []
                for objectHistory in self.objectHistoriesForGivenID:
                    if objectHistory.getID() == combinedGroup.getGroupSetID():
                        objectHistoryAvgSize = objectHistory.getAvgSizeAcrossHistory()
                        if abs((combinedGroupOnlyPeakGabor[0][0]-objectHistoryAvgSize[0])/objectHistoryAvgSize[0] + (combinedGroupOnlyPeakGabor[0][1]-objectHistoryAvgSize[1])/objectHistoryAvgSize[1])/2 < 0.7:
                            sizeIsSignificantDeviationFromHistoricalSize = True
                        break
                
                combinedGroupPeakGaborNextPredictedBoundingBoxPosition = [combinedGroupOnlyPeakGabor[0][0]+combinedGroupOnlyPeakGabor[1][0],combinedGroupOnlyPeakGabor[0][1]+combinedGroupOnlyPeakGabor[1][1],combinedGroupOnlyPeakGabor[0][2]+combinedGroupOnlyPeakGabor[1][0],combinedGroupOnlyPeakGabor[0][3]+combinedGroupOnlyPeakGabor[1][1]]
                if sizeIsSignificantDeviationFromHistoricalSize:
                    combinedGroupPeakGaborNextPredictedBoundingBoxPosition = [(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[0]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[2])/2-objectHistoryAvgSize[0]/2,(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[1]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[3])/2-objectHistoryAvgSize[1]/2,(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[0]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[2])/2+objectHistoryAvgSize[0]/2,(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[1]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[3])/2+objectHistoryAvgSize[1]/2]

                self.nextFrameROIs[0].addObjectToROI(combinedGroupOnlyPeakGabor[2],combinedGroupPeakGaborNextPredictedBoundingBoxPosition)
                print("final group: " + str(combinedGroupOnlyPeakGabor) + "\n")

        else:
            for combinedGroup in superCombinedGroups:
                combinedGroupOnlyPeakGabor = combinedGroup.determineGroupBoundingBoxAndMovementVectorAndID()
                combinedGroupsOnlyPeakGabor.append(combinedGroupOnlyPeakGabor)
                combinedGroupsAllGroups.append(combinedGroup.getGroups())

                maxObjectSizeInHistory = 0
                objectFound = False
                for objectHistoryForGivenID in self.objectHistoriesForGivenID:
                    if objectHistoryForGivenID.getID() == combinedGroup.getGroupSetID():
                        objectFound = True
                        objectHistoryForGivenID.addToSizeAndPositionAndVectorHistory(np.array(combinedGroup.determineGroupSize()),np.array(combinedGroupOnlyPeakGabor[0]),np.array(combinedGroupOnlyPeakGabor[1]))
                        maxObjectSizeInHistory = objectHistoryForGivenID.getAvgSizeAcrossHistory()
                if not objectFound:
                    self.objectHistoriesForGivenID.append(self.ObjectHistory(combinedGroup.getGroupSetID(),np.array(combinedGroup.determineGroupSize()),np.array(combinedGroupOnlyPeakGabor[0]),np.array(combinedGroupOnlyPeakGabor[1])))
                    maxObjectSizeInHistory = np.array(combinedGroup.determineGroupSize())

                sizeIsSignificantDeviationFromHistoricalSize = False
                for objectHistory in self.objectHistoriesForGivenID:
                    if objectHistory.getID() == combinedGroup.getGroupSetID():
                        objectHistoryAvgSize = objectHistory.getAvgSizeAcrossHistory()
                        if abs((combinedGroupOnlyPeakGabor[0][0]-objectHistoryAvgSize[0])/objectHistoryAvgSize[0] + (combinedGroupOnlyPeakGabor[0][1]-objectHistoryAvgSize[1])/objectHistoryAvgSize[1])/2 < 0.7:
                            sizeIsSignificantDeviationFromHistoricalSize = True
                        break
                
                combinedGroupPeakGaborNextPredictedBoundingBoxPosition = [combinedGroupOnlyPeakGabor[0][0]+combinedGroupOnlyPeakGabor[1][0],combinedGroupOnlyPeakGabor[0][1]+combinedGroupOnlyPeakGabor[1][1],combinedGroupOnlyPeakGabor[0][2]+combinedGroupOnlyPeakGabor[1][0],combinedGroupOnlyPeakGabor[0][3]+combinedGroupOnlyPeakGabor[1][1]]
                if sizeIsSignificantDeviationFromHistoricalSize:
                    combinedGroupPeakGaborNextPredictedBoundingBoxPosition = [(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[0]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[2])/2-objectHistoryAvgSize[0]/2,(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[1]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[3])/2-objectHistoryAvgSize[1]/2,(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[0]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[2])/2+objectHistoryAvgSize[0]/2,(combinedGroupPeakGaborNextPredictedBoundingBoxPosition[1]+combinedGroupPeakGaborNextPredictedBoundingBoxPosition[3])/2+objectHistoryAvgSize[1]/2]

                nextFrameROI = self.ROI([combinedGroupOnlyPeakGabor[0][0]+combinedGroupOnlyPeakGabor[1][0]-maxObjectSizeInHistory[0]*self.objectExpansionBuffer,combinedGroupOnlyPeakGabor[0][1]+combinedGroupOnlyPeakGabor[1][1]-maxObjectSizeInHistory[1]*self.objectExpansionBuffer,combinedGroupOnlyPeakGabor[0][2]+combinedGroupOnlyPeakGabor[1][0]+maxObjectSizeInHistory[0]*self.objectExpansionBuffer,combinedGroupOnlyPeakGabor[0][3]+combinedGroupOnlyPeakGabor[1][1]+maxObjectSizeInHistory[1]*self.objectExpansionBuffer],[combinedGroup.getGroupSetID()], [combinedGroupPeakGaborNextPredictedBoundingBoxPosition])
                if sizeIsSignificantDeviationFromHistoricalSize:
                    nextFrameROI = self.ROI([combinedGroupOnlyPeakGabor[0][0]+combinedGroupOnlyPeakGabor[1][0]-maxObjectSizeInHistory[0]*self.objectExpansionBuffer,combinedGroupOnlyPeakGabor[0][1]+combinedGroupOnlyPeakGabor[1][1]-maxObjectSizeInHistory[1]*self.objectExpansionBuffer,combinedGroupOnlyPeakGabor[0][2]+combinedGroupOnlyPeakGabor[1][0]+maxObjectSizeInHistory[0]*self.objectExpansionBuffer,combinedGroupOnlyPeakGabor[0][3]+combinedGroupOnlyPeakGabor[1][1]+maxObjectSizeInHistory[1]*self.objectExpansionBuffer],[combinedGroup.getGroupSetID()], [combinedGroupPeakGaborNextPredictedBoundingBoxPosition])
                
                nextFrameROI.normalizeROIToHeightAndWidth(height, width)
                addedToExistingROI = False
                for existingNextFrameROI in self.nextFrameROIs:
                    if existingNextFrameROI.checkOtherROIOverlap(nextFrameROI):
                        addedToExistingROI = True
                        existingNextFrameROI.combineOtherROI(nextFrameROI)
                if not addedToExistingROI:
                    self.nextFrameROIs.append(nextFrameROI)
                print("final group: " + str(combinedGroupOnlyPeakGabor) + "\n")
        q = 0
        while q < len(self.nextFrameROIs):
            nextFrameROIPos = self.nextFrameROIs[q].getPos()
            if int(nextFrameROIPos[0]) == int(nextFrameROIPos[2]) or int(nextFrameROIPos[1]) == int(nextFrameROIPos[3]):
                self.nextFrameROIs.pop(q)
            else:
                q+=1

        #final check for overlapping ROIs
        i = 0
        while i < len(self.nextFrameROIs):
            j = i+1
            while j < len(self.nextFrameROIs):
                if self.nextFrameROIs[i].checkOtherROIOverlap(self.nextFrameROIs[j]):
                    self.nextFrameROIs[i].combineOtherROI(self.nextFrameROIs[j])
                    self.nextFrameROIs.pop(j)
                else:
                    j+=1
            i+=1

        self.countMatchGroupsAndDeterminePeakFilter+=1
        return (combinedGroupsOnlyPeakGabor, combinedGroupsAllGroups)
    def getNextFrameROIs(self):
        toReturn = []
        for nextFrameROI in self.nextFrameROIs:
            toReturn.append(nextFrameROI.getPos())
        return toReturn


    

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

gaborFilters = GaborFilters(gaborThreshold, 0.005, numAlternationsMin, numAlternationsMax, alternationInterval, numCores, 0.4)
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
    groupsToTrack, allGroups = gaborFilters.matchGroupsAndDeterminePeakFilter(np.array(grayHistory))
    detections = []
    log.write("currently on frame: " + str(numFramesTraversed)+"\n")
    for superGroup in allGroups:
        for group in superGroup:
            frameHistory[int(frameHistorySizeNeeded/2)] = cv2.rectangle(frameHistory[int(frameHistorySizeNeeded/2)], (int(group[1][1]), int(group[1][0])),(int(group[1][3]),int(group[1][2])),(122,122,255),1)
    for group in groupsToTrack:
        groupCol = (group[0][3] + group[0][1])/2
        groupRow = (group[0][2] + group[0][0])/2
        detections.append(Detection(points = np.array([groupCol,groupRow]), scores = np.array([1])))
        print("\npoint to add to detections: " + str(groupRow) + ", " + str(groupCol))
        frameHistory[int(frameHistorySizeNeeded/2)] = cv2.arrowedLine(frameHistory[int(frameHistorySizeNeeded/2)], (int(groupCol), int(groupRow)),(int(groupCol+group[1][1]), int(groupRow+group[1][0])),(255,0,0),2)
        #frameHistory[int(frameHistorySizeNeeded/2)] = cv2.rectangle(frameHistory[int(frameHistorySizeNeeded/2)], (int(group[0][1]), int(group[0][0])),(int(group[0][3]),int(group[0][2])),(0,255,122),2)
        frameHistory[int(frameHistorySizeNeeded/2)] = cv2.putText(frameHistory[int(frameHistorySizeNeeded/2)], str(group[2]), (int(groupCol),int(groupRow)),cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2, cv2.LINE_AA)
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