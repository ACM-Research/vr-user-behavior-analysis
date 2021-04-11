import math
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import cv2

# Height and Width given in angles (degrees)
VIEWPORT_WIDTH = 114
VIEWPORT_HEIGHT = 57

def scalingVoteFunctionLinear(a, b):
    scaleArr = np.zeros((2*b+1, 2*a+1))
    for x in range(-a, a + 1):
        for y in range(-b, b + 1):
            # theta = 0
            if x == 0 and y == 0:
                scaleArr[b][a] = (a+b)/2
            else:
                theta = math.atan2(y, x)
                ellipseRadius = (a * b) / math.sqrt((a**2) * math.sin(theta) ** 2 + (b ** 2) * math.cos(theta) ** 2)
                radius = math.sqrt(x ** 2 + y ** 2)
                dist = ((a+b)/2) * ((ellipseRadius - radius) / ellipseRadius) #Max value of linear function
                if dist < 0:
                    scaleArr[y + b][x + a] = 0
                else:
                    scaleArr[y + b][x + a] = dist
    return scaleArr

def scalingVoteFunctionSqrt(a, b):
    linearArr =  scalingVoteFunctionLinear(a, b)
    scaleArr = np.zeros((2*b+1, 2*a+1))
    for x in range(-a, a + 1):
        for y in range(-b, b + 1):
            scaleArr[y+b][x+a] = math.sqrt(linearArr[y+b][x+a])
    return scaleArr

def scalingVoteFunctionSemiCrcl(a, b):
    linearArr =  scalingVoteFunctionLinear(a, b)
    scaleArr = np.zeros((2*b+1, 2*a+1))
    for x in range(-a, a + 1):
        for y in range(-b, b + 1):
            scaleArr[y+b][x+a] = (a+b / 2) - math.sqrt((a+b / 2) ** 2 - linearArr[y+b][x+a] ** 2)
    return scaleArr

def scalingVoteFunctionUniform(a, b):
    linearArr =  scalingVoteFunctionLinear(a, b)
    scaleArr = np.zeros((2*b+1, 2*a+1))
    for x in range(-a, a + 1):
        for y in range(-b, b + 1):
            if linearArr[y+b][x+a] > 0.1:
                scaleArr[y+b][x+a] = 1
    return scaleArr

def scalingVoteFunctionSquared(a, b):
    linearArr =  scalingVoteFunctionLinear(a, b)
    scaleArr = np.zeros((2*b+1, 2*a+1))
    for x in range(-a, a + 1):
        for y in range(-b, b + 1):
            scaleArr[y+b][x+a] = linearArr[y+b][x+a] ** 2
    return scaleArr

class HeatMap:

    def __init__(self, dParser):
        self.rows = dParser.rows
        self.cols = dParser.cols
        self.data = dParser

    def generateHeatMapTestingArrs(self, testFrames, functions):
        heatMapArrays = []
        semiHorizontalAxis = int(VIEWPORT_WIDTH * (self.cols / 360) / 2)
        semiVerticalAxis = int(VIEWPORT_HEIGHT * (self.rows / 180) / 2)
        dirname = os.getcwd()
        for frame in testFrames:
            imgName = f'frame{frame}.jpg'
            for scalingFunction in functions:
                fileName = f'{dirname}/testFunc/heatMaps/{scalingFunction}/{imgName}'
                if scalingFunction == 'semiCrcl':
                    scaleArr = scalingVoteFunctionSemiCrcl(semiHorizontalAxis, semiVerticalAxis)
                elif scalingFunction == 'sqrt':
                    scaleArr = scalingVoteFunctionSqrt(semiHorizontalAxis, semiVerticalAxis)
                elif scalingFunction == 'uniform':
                    scaleArr = scalingVoteFunctionUniform(semiHorizontalAxis, semiVerticalAxis)
                elif scalingFunction == 'linear':
                    scaleArr = scalingVoteFunctionLinear(semiHorizontalAxis, semiVerticalAxis)
                elif scalingFunction == 'squared':
                    scaleArr = scalingVoteFunctionSquared(semiHorizontalAxis, semiVerticalAxis)
                self.data.convertusertraces(frame)
                heatMapArr = np.zeros((self.rows, self.cols))
                for usertrace in self.data.usertraces:
                    center = usertrace[2]
                    centerX, centerY = center
                    for x in range(-semiHorizontalAxis, semiHorizontalAxis + 1):
                        for y in range(-semiVerticalAxis, semiVerticalAxis + 1):
                            if centerX + x >= 0 and centerY + y >= 0 and centerX + x < self.cols and centerY + y < self.rows:
                                heatMapArr[y + centerY][x + centerX] += scaleArr[y + semiVerticalAxis][x + semiHorizontalAxis]
                            elif centerX + x < 0 and centerY + y >= 0 and centerY + y < self.rows:
                                xPos = self.cols + centerX + x
                                heatMapArr[y + centerY][xPos] += scaleArr[y + semiVerticalAxis][x + semiHorizontalAxis]
                            elif centerX + x >= self.cols and centerY + y >= 0 and centerY + y < self.rows:
                                xPos = centerX + x - self.cols
                                heatMapArr[y + centerY][xPos] += scaleArr[y + semiVerticalAxis][x + semiHorizontalAxis]
                heatMapArrays.append((heatMapArr, fileName, frame))
        return heatMapArrays

    # function to create an array of all necessary heat map frames; allows for scalable max look value
    def generateHeatMapArrs(self, scalingFunction = 'semiCrcl'):
        heatMapArrays = []
        semiHorizontalAxis = int(VIEWPORT_WIDTH * (self.cols / 360) / 2)
        semiVerticalAxis = int(VIEWPORT_HEIGHT * (self.rows / 180) / 2)
        if scalingFunction == 'semiCrcl':
            scaleArr = scalingVoteFunctionSemiCrcl(semiHorizontalAxis, semiVerticalAxis)
        elif scalingFunction == 'sqrt':
            scaleArr = scalingVoteFunctionSqrt(semiHorizontalAxis, semiVerticalAxis)
        elif scalingFunction == 'uniform':
            scaleArr = scalingVoteFunctionUniform(semiHorizontalAxis, semiVerticalAxis)
        elif scalingFunction == 'linear':
            scaleArr = scalingVoteFunctionLinear(semiHorizontalAxis, semiVerticalAxis)
        elif scalingFunction == 'squared':
            scaleArr = scalingVoteFunctionSquared(semiHorizontalAxis, semiVerticalAxis)
        else:
            print("Unknown scaling function given, using semi-circle instead")
            scaleArr = scalingVoteFunctionSemiCrcl(semiHorizontalAxis, semiVerticalAxis)
        for frame in self.data.frameList():
            self.data.convertusertraces(frame)
            heatMapArr = np.zeros((self.rows, self.cols))
            for usertrace in self.data.usertraces:
                center = usertrace[2]
                centerX, centerY = center
                for x in range(-semiHorizontalAxis, semiHorizontalAxis + 1):
                    for y in range(-semiVerticalAxis, semiVerticalAxis + 1):
                        if centerX + x >= 0 and centerY + y >= 0 and centerX + x < self.cols and centerY + y < self.rows:
                            heatMapArr[y + centerY][x + centerX] += scaleArr[y + semiVerticalAxis][x + semiHorizontalAxis]
                        elif centerX + x < 0 and centerY + y >= 0 and centerY + y < self.rows:
                            xPos = self.cols + centerX + x
                            heatMapArr[y + centerY][xPos] += scaleArr[y + semiVerticalAxis][x + semiHorizontalAxis]
                        elif centerX + x >= self.cols and centerY + y >= 0 and centerY + y < self.rows:
                            xPos = centerX + x - self.cols
                            heatMapArr[y + centerY][xPos] += scaleArr[y + semiVerticalAxis][x + semiHorizontalAxis]
            heatMapArrays.append((heatMapArr, 'heatMaps', frame))
        return heatMapArrays

        #Takes approximately 30 seconds on video 23 or 24
    def createHeatMapVideo(self, fps, videoName = 'heatmapVideo.avi', videoOverlay = False, scalingFunction = 'semiCrcl'):
        print("Creating {} video".format(videoName))
        heatMapArrays = self.generateHeatMapArrs(scalingFunction)
        if videoOverlay:
            out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, self.data.imagesize)
        else:
            # print(heatMapArrays[0])
            plt.figure(figsize=(16,8),dpi=100)
            ax = sb.heatmap(heatMapArrays[0][0], vmin=0, cbar=False)
            
            ax.axis('off')
            plt.savefig('heatmap.jpg', bbox_inches="tight", pad_inches=0)
            plt.close()
            img = cv2.imread('heatmap.jpg')
            height, width, layers = img.shape
            size = (width, height) #Size of heatmap image
            out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        numofframes = len(self.data.frameList())
        progress = 0
        frameId = 1
        for map in heatMapArrays:
            plt.figure(figsize=(16,8),dpi=100)
            ax = sb.heatmap(map[0], vmin=0, cbar=False)
            ax.axis('off')
            
            plt.savefig('heatmap.jpg', bbox_inches="tight", pad_inches=0)
            plt.close()
            heatMap = cv2.imread('heatmap.jpg')
            if videoOverlay:
                resizedHeatMap = cv2.resize(heatMap, self.data.imagesize)
                frameImg = self.data.getFrame(frameId)
                greyImg = cv2.cvtColor(frameImg, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('grey.jpg', greyImg)
                greyImg = cv2.imread('grey.jpg') 
                fullImg = cv2.addWeighted(resizedHeatMap, 0.9, greyImg, 0.8, 0)
                out.write(fullImg)
                frameId += 30
            else:
                out.write(heatMap)
            progress += (100 / numofframes)
            print("Progress " + str(int(progress)) + "%", end='\r', flush=True)
        out.release()
        print()
