import math
import os
from typing import Tuple, List, Dict, Callable

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sb

import cv2
import math

class DataParser:

    def __init__(self, basedir, videoId, rows, cols):
        self.rows = rows
        self.cols = cols
        self.usertracepath = f"{basedir}/Data/UserTracesByVideo/{videoId}/"
        self.frameimgs = f"{basedir}/Data/VideosData/Videos/SourceFrames/{videoId}/"
        with Image.open(f"{self.frameimgs}/frame1.jpg") as im:
            self.imagesize = (im.size[0], im.size[1])
        self.importusertraces()

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

    @staticmethod
    def scalingVoteFunction(radius):
        scaleArr = np.zeros((2*radius+1, 2*radius+1))
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                distFromCircle = (radius * radius) - ((x*x) + (y*y))
                if distFromCircle < 0:
                    scaleArr[x + radius][y + radius] = 0
                else:
                    scaleArr[x + radius][y + radius] = math.sqrt(distFromCircle)
        return scaleArr

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
    def generateHeatMapArrs(self):
        heatMapArrays = []
        for frame in self.frameList():
            self.convertusertraces(frame)
            radius = 5
            scaleArr = self.scalingVoteFunction(radius)
            heatMapArr = np.zeros((self.rows, self.cols))
            for usertrace in self.usertraces:
                center = usertrace[2]
                centerX, centerY = center
                for x in range(-radius, radius + 1):
                    for y in range(-radius, radius + 1):
                        if centerX + x >= 0 and centerY + y >= 0 and centerX + x < self.cols and centerY + y < self.rows:
                            heatMapArr[y + centerY][x + centerX] += scaleArr[y + radius][x + radius]
            heatMapArrays.append(heatMapArr)
        return heatMapArrays

    # NOTE: deprecated function that creates a single heatmap img; max look value not scalable
    # def generateHeatMap(self, frameId):
    #     fig, ax = plt.subplots(figsize=(12,6))
    #     ax.axis('off')
    #     self.convertusertraces(frameId)
    #     heatMapArr = np.zeros((self.rows, self.cols))
    #     for usertrace in self.usertraces:
    #         indices = usertrace[2]
    #         heatMapArr[indices[1]][indices[0]] += 1
    #     sb.heatmap(heatMapArr, vmin=0, vmax=15)
    #     plt.savefig('heatmap.jpg')
    #     plt.close()
        
    #Takes approximately 30 seconds on video 23 or 24
    def createHeatMapVideo(self, fps, videoName = 'heatmapVideo.avi', videoOverlay = False):
        print("Creating {} video".format(videoName))
        heatMapArrays = self.generateHeatMapArrs()
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
                fullImg = cv2.addWeighted(resizedHeatMap, 3, greyImg, 1, 0)
                out.write(fullImg)
                frameId += 30
            else:
                out.write(heatMap)
            progress += (100 / numofframes)
            print("Progress " + str(int(progress)) + "%", end='\r', flush=True)
        out.release()
        print()

def main():
    filepath = os.getcwd()
    data = DataParser(filepath, videoId=23, rows=50, cols=100)
    data.createHeatMapVideo(fps=2)
    #data.createHeatMapVideo(fps=2, videoName = 'heatMapVideoWithOverlap.avi', videoOverlay=True)

if __name__ == "__main__":
    main()