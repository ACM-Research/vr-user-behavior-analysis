from dataParser import DataParser
from heatMap import HeatMap
from deresolutionizer import Deresolutionizer as DeRes
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path

import cv2



class VRConverter:

    def __init__(self, videoId, rows, cols, heatThreshold = .25):
        self.data = DataParser(os.getcwd(), videoId, rows, cols)
        self.heat = HeatMap(self.data)
        self.res = DeRes(self.data, heatThreshold)
        

    def renderMapImgs(self, mapArrs, videoOverlay=False):
        numofframes = len(mapArrs)
        progress = 0
        for Map in mapArrs:
            plt.figure(figsize=(16,8),dpi=100)
            ax = sb.heatmap(Map[0], vmin=0, cbar=False)
            ax.axis('off')
            # ax.set_aspect(.5)
            plt.savefig(Map[1], bbox_inches="tight", pad_inches=0)
            if videoOverlay:
                heatMap = cv2.imread(Map[1])
                resizedHeatMap = cv2.resize(heatMap, self.data.imagesize)
                frameImg = self.data.getFrame(Map[2])
                greyImg = cv2.cvtColor(frameImg, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('grey.jpg', greyImg)
                greyImg = cv2.imread('grey.jpg') 
                fullImg = cv2.addWeighted(resizedHeatMap, 0.95, greyImg, 0.55, 0)
                cv2.imwrite(Map[1], fullImg)
                
            plt.close()
            progress += (100 / numofframes)
            
            print("Progress " + str(int(progress)) + "%", end='\r', flush=True)

    def generateTestMaps(self, testFrames, videoOverlay=False):
        
        fig, ax = plt.subplots(figsize=(12,6))
        ax.axis('off')
        dirname = os.getcwd()
        Path(f'{dirname}/testFunc/splitImgs').mkdir(parents=True, exist_ok=True)
        functions = ['semiCrcl', 'sqrt', 'uniform', 'linear', 'squared']
        
        for func in functions:
            Path(f'{dirname}/testFunc/heatMaps/{func}').mkdir(parents=True, exist_ok=True)
            Path(f'{dirname}/testFunc/resMaps/{func}').mkdir(parents=True, exist_ok=True)
            Path(f'{dirname}/testFunc/compressedImgs/{func}').mkdir(parents=True, exist_ok=True)

        heatMapArrays = self.heat.generateHeatMapTestingArrs(testFrames, functions)

        resMapArrs = self.res.generateResMapArrs(heatMapArrays)
        
        print("Creating Resolution Maps...")
        self.renderMapImgs(resMapArrs, videoOverlay)

        progress = 0
        numofimgs = len(resMapArrs)
        print("Creating compressed images...")
        for Map in resMapArrs:
            fullImg = self.res.compressImg(Map)
            fullImg.save(Map[1].replace('resMaps', 'compressedImgs'), quality=95)
            progress += (100 / numofimgs)
            print("Progress " + str(int(progress)) + "%", end='\r', flush=True)
        
        print("Creating Heat Maps...")
        self.renderMapImgs(heatMapArrays, videoOverlay)

    def makeVideo(self, videoType, name, scalingFunction, fps=2, videoOverlay=False):
        if videoType == 'heat':
            self.heat.createHeatMapVideo(fps=fps, videoName=name, videoOverlay=videoOverlay, scalingFunction=scalingFunction)
        elif videoType == 'compress':
            self.res.createCompressedVideo(fps=fps, heatMapArrays=self.heat.generateHeatMapArrs(scalingFunction), videoName=name, scalingFunction=scalingFunction)
        elif videoType == 'control':
            self.data.createControlVideo(fps=fps, videoName=name)

    def getStats(self, scalingFunction):
        heatMaps = self.heat.generateHeatMapArrs(scalingFunction)
        resMaps = self.res.generateResMapArrs(heatMaps)
        userExpPerFrame, userExpPerUser = self.res.generateUserExpStats(resMaps, scalingFunction)
        storagePerFrame = self.res.generateStorageStats(resMaps)
        return userExpPerFrame, userExpPerUser, storagePerFrame

    def generateHeatMapCSVs(self):
        self.heat.generateHeatMapCSVs()

def main():

    converter = VRConverter(videoId=23, rows=50, cols=100, heatThreshold=.25)
    TEST_FRAMES = [121, 271, 691, 811, 1111, 1351, 1681]
    # converter.makeVideo('compress', 'CompressedSemi23New.avi', 'semiCrcl')
    userExpPerFrame, userExpPerUser, storagePerFrame = converter.getStats('semiCrcl')
    for frame in storagePerFrame:
        print(f'{frame}: {storagePerFrame[frame] * 100}%')
    # converter.generateTestMaps(TEST_FRAMES)
    # converter.generateHeatMapCSVs()

if __name__ == "__main__":
    main()
    # testScaling()
    