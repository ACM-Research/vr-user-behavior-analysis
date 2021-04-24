from dataParser import DataParser
from heatMap import HeatMap
from deresolutionizer import Deresolutionizer as DeRes
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sb
from pathlib import Path
import pandas as pd

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



def animate(i):
    frame = data.iloc[:int(i+1)]
    frame = pd.melt(frame, ignore_index=False, var_name='Type', value_name='Ratio')
    p = sb.lineplot(x=frame.index, y='Ratio', data=frame, hue='Type')
    plt.legend(['User Experience', 'Storage'])
    pass

def main():
    fig = plt.figure(figsize=(12, 6))
    sb.set_theme()
    plt.ylim(0, 1)
    converter = VRConverter(videoId=23, rows=50, cols=100, heatThreshold=.25)
    TEST_FRAMES = [121, 271, 691, 811, 1111, 1351, 1681]
    # converter.makeVideo('compress', 'CompressedSemi23New.avi', 'semiCrcl')
    userExpPerFrame, userExpPerUser, storagePerFrame  = converter.getStats('linear')
    # for frame in userExpPerFrame:
    #     print(f'{frame}: {userExpPerFrame[frame] * 100}%')
    frames = list(userExpPerFrame.keys())
    plt.xlim(frames[0], frames[-1])
    userValues = list(userExpPerFrame.values())
    storageValues = list(storagePerFrame.values())
    dataArray = np.array([userValues, storageValues])
    dataArray = np.transpose(dataArray)
    # print(dataArray)
    global data
    data = pd.DataFrame(dataArray, index=frames, columns=['User Experience', 'Storage'])
    
    # print(pd.melt(data, ignore_index=False, var_name='Type', value_name='Ratio'))
    # ax = sb.lineplot(data=data)
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), repeat=True)
    
    # ani.save()
    
    plt.show()
    # plt.plot(frames, values)
    # plt.show()
    # converter.generateTestMaps(TEST_FRAMES)
    # converter.generateHeatMapCSVs()

if __name__ == "__main__":
    main()
    # testScaling()
    