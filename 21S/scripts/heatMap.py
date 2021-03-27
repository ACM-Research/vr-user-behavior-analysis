import math
import os
from typing import Tuple, List, Dict, Callable

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sb
from pathlib import Path

import cv2
import math

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

class DataParser:

    def __init__(self, basedir, videoId, rows, cols):
        self.rows = rows
        self.cols = cols
        self.usertracepath = f"{basedir}/Data/UserTracesByVideo/{videoId}/"
        self.frameimgs = f"{basedir}/Data/VideosData/Videos/SourceFrames/{videoId}/"
        with Image.open(f"{self.frameimgs}/frame1.jpg") as im:
            self.imagesize = (im.size[0], im.size[1])
        self.importusertraces()
        self.testFrames = [121, 271, 691, 811, 1111, 1351, 1681]

    def frameList(self):
        framenums = []
        index = 1
        frames = [frame for frame in os.listdir(self.frameimgs)]
        for frame in frames:
            framenums.append(index)
            index += 30
        return framenums

    @staticmethod
    def convvec2angl(vector):
        phi = math.degrees(math.asin(vector[1]))
        theta = math.degrees(math.atan2(vector[0], vector[2]))
        return theta, phi

    #TODO: Create function for determining how accurate the scaling function is (compare to predefined 2d array of values)

    
        

    def getFrame(self, id):
        frameName = 'frame{}.jpg'.format(id)
        img = cv2.imread(os.path.join(self.frameimgs, frameName))
        return img

    def importusertraces(self):
        """Note that this parser is very simple in nature and doesn't really *need*
        a separate class."""
        self.all_user_traces = []
        user_folders = [trace for trace in os.listdir(self.usertracepath)]
        for user in user_folders:
            # Test for exclusion.
            # Or we would, if it weren't now done at runtime.
            # if QuestionnaireParser is not None:
            #     if not sample_exclusion_fxn(user[: user.find('.csv')], self.quesparser):
            #         continue
            userid = user[: user.find('.csv')]
            trace_data = pd.read_csv(f"{self.usertracepath}/{user}")
            trace_rows = trace_data.values
            self.all_user_traces.append((trace_rows, userid))
        
        
    def convertusertraces(self, frame):
        # draw user trace points
        self.usertraces = []
        # frameList = self.frameList()
        colwidth = self.imagesize[0] / self.cols # this can be changed
        rowheight = self.imagesize[1] / self.rows
        for trace_rows, userid in self.all_user_traces:
            
            trace_row = trace_rows[frame - 1]  # Be careful about indexing!!
            arr = [trace_row[5], trace_row[6], trace_row[7]]
            x, y = self.convvec2angl(arr)
            x = ((x+180)/360) * self.imagesize[0]
            y = ((90-y)/180) * self.imagesize[1]
            x_index = int(x / colwidth)
            y_index = int(y / rowheight)
            self.usertraces.append((userid, frame, (x_index, y_index)))
                
    # function to create an array of all necessary heat map frames; allows for scalable max look value
    def generateHeatMapArrs(self, scalingFunction = 'semiCrcl'):
        heatMapArrays = []
        semiHorizontalAxis = int(114 * (self.cols / 360))
        semiVerticalAxis = int(57 * (self.rows / 180))
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
        for frame in self.frameList():
            self.convertusertraces(frame)
            heatMapArr = np.zeros((self.rows, self.cols))
            for usertrace in self.usertraces:
                center = usertrace[2]
                centerX, centerY = center
                for x in range(-semiHorizontalAxis, semiHorizontalAxis + 1):
                    for y in range(-semiVerticalAxis, semiVerticalAxis + 1):
                        if centerX + x >= 0 and centerY + y >= 0 and centerX + x < self.cols and centerY + y < self.rows:
                            heatMapArr[y + centerY][x + centerX] += scaleArr[y + b][x + a]
            heatMapArrays.append(heatMapArr)
        return heatMapArrays

    # NOTE: deprecated function that creates a single heatmap img; max look value not scalable
    def generateTestMaps(self, testFrames):
        fig, ax = plt.subplots(figsize=(12,6))
        ax.axis('off')
        dirname = os.getcwd()
        fileNames = []
        heatMapArrays = []
        semiHorizontalAxis = int(114 * (self.cols / 360) / 2)
        semiVerticalAxis = int(57 * (self.rows / 180) / 2)
        progress = 0
        functions = ['semiCrcl', 'sqrt', 'uniform', 'linear', 'squared']
        numofframes = len(testFrames) * len(functions)
        for func in functions:
            Path(f'{dirname}/testFunc/{func}').mkdir(parents=True, exist_ok=True)
        radius = 5
        for frame in testFrames:
            imgName = f'frame{frame}.jpg'
            for scalingFunction in functions:
                fileName = f'{dirname}/testFunc/{scalingFunction}/{imgName}'
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
                self.convertusertraces(frame)
                heatMapArr = np.zeros((self.rows, self.cols))
                for usertrace in self.usertraces:
                    center = usertrace[2]
                    centerX, centerY = center
                    for x in range(-semiHorizontalAxis, semiHorizontalAxis + 1):
                        for y in range(-semiVerticalAxis, semiVerticalAxis + 1):
                            if centerX + x >= 0 and centerY + y >= 0 and centerX + x < self.cols and centerY + y < self.rows:
                                heatMapArr[y + centerY][x + centerX] += scaleArr[y + semiVerticalAxis][x + semiHorizontalAxis]
                heatMapArrays.append((heatMapArr, fileName))
        
        for Map in heatMapArrays:
            plt.figure(figsize=(16,8),dpi=100)
            ax = sb.heatmap(Map[0], vmin=0, cbar=False)
            ax.axis('off')
            # ax.set_aspect(.5)
            plt.savefig(Map[1], bbox_inches="tight", pad_inches=0)
            plt.close()
            progress += (100 / numofframes)
            print("Progress " + str(int(progress)) + "%", end='\r', flush=True)
        
    #Takes approximately 30 seconds on video 23 or 24
    def createHeatMapVideo(self, fps, videoName = 'heatmapVideo.avi', videoOverlay = False, scalingFunction = 'semiCrcl'):
        print("Creating {} video".format(videoName))
        heatMapArrays = self.generateHeatMapArrs(scalingFunction)
        max_looks = max([np.amax(arr) for arr in heatMapArrays])
        if videoOverlay:
            out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, self.imagesize)
        else:
            ax = sb.heatmap(heatMapArrays[0], vmin=0, vmax=max_looks, cbar=False)
            ax.axis('off')
            plt.savefig('heatmap.jpg', bbox_inches="tight", pad_inches=0)
            plt.close()
            img = cv2.imread('heatmap.jpg')
            height, width, layers = img.shape
            size = (width, height) #Size of heatmap image
            out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        numofframes = len(self.frameList())
        progress = 0
        frameId = 1
        for map in heatMapArrays:
            ax = sb.heatmap(map, vmin=0, vmax=max_looks, cbar=False)
            ax.axis('off')
            plt.savefig('heatmap.jpg', bbox_inches="tight", pad_inches=0)
            plt.close()
            heatMap = cv2.imread('heatmap.jpg')
            if videoOverlay:
                resizedHeatMap = cv2.resize(heatMap, self.imagesize)
                frameImg = self.getFrame(frameId)
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

#Will need to split up source videos into frames once ML is used

def main():
    filepath = os.getcwd()
    data = DataParser(filepath, videoId=23, rows=100, cols=200)
    #data.createHeatMapVideo(fps=2)
    data.createHeatMapVideo(fps=2, videoName = 'heatMapVideoWithOverlapLinear.avi', videoOverlay=True, scalingFunction='linear')

def testScaling():
    filepath = os.getcwd()
    data = DataParser(filepath, videoId=23, rows=100, cols=200)
    data.generateTestMaps(data.testFrames)
    
    #data.createHeatMapVideo(fps=2)

if __name__ == "__main__":
    # main()
    testScaling()
    #filepath = os.getcwd()
    #data = DataParser(filepath, videoId=23, rows=50, cols=100)
    #data.generateTestMaps(data.testFrames)
    #scaleArr = data.scalingVoteFunctionSemiCrcl(10, 5)
    #for a in scaleArr:
     #   for b in a:
      #      print(round(b,2), end=' ')
       # print()