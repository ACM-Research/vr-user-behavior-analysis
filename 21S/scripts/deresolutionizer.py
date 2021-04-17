import os

import math
import numpy as np
from PIL import Image, ImageDraw

import cv2

# Height and Width given in angles (degrees)
VIEWPORT_WIDTH = 114
VIEWPORT_HEIGHT = 57

class Deresolutionizer:

    def __init__(self, dParser):
        self.traces = dParser.convertTracesForAllUsers()
        self.rows = dParser.rows
        self.cols = dParser.cols
        self.imagesize = dParser.imagesize
        self.frameimgs = dParser.frameimgs
        self.framelist = dParser.frameList()
        
    def generateUserExpStats(self, resMaps):
        semiHorizontalAxis = int(VIEWPORT_WIDTH * (self.cols / 360) / 2)
        semiVerticalAxis = int(VIEWPORT_HEIGHT * (self.rows / 180) / 2)
        avgPerFrame = {frame: 0 for frame in self.framelist}
        avgPerUser = {}
        for user in self.traces:
            avgPerUser[user] = 0
            for trace in self.traces[user]:
                centerX, centerY = trace[1]
                frameIndex = int(trace[0] / 30)
                resMapArr = resMaps[frameIndex][0]
                userFrameTotal = 0
                area = 0
                for x in range(-semiHorizontalAxis, semiHorizontalAxis + 1):
                    for y in range(-semiVerticalAxis, semiVerticalAxis + 1):
                        theta = math.atan2(y, x)
                        a = semiHorizontalAxis
                        b = semiVerticalAxis
                        ellipseRadius = (a * b) / math.sqrt((a**2) * math.sin(theta) ** 2 + (b ** 2) * math.cos(theta) ** 2)
                        radius = math.sqrt(x ** 2 + y ** 2)
                        if radius < ellipseRadius:
                            area += 1
                            # if position is within frame
                            if centerX + x >= 0 and centerY + y >= 0 and centerX + x < self.cols and centerY + y < self.rows:
                                userFrameTotal += resMapArr[y + centerY][x + centerX] + 1
                            # if position is too far to the left
                            elif centerX + x < 0 and centerY + y >= 0 and centerY + y < self.rows:
                                xPos = self.cols + centerX + x
                                userFrameTotal += resMapArr[y + centerY][xPos] + 1
                            # too far to the right
                            elif centerX + x >= self.cols and centerY + y >= 0 and centerY + y < self.rows:
                                xPos = centerX + x - self.cols
                                userFrameTotal += resMapArr[y + centerY][xPos] + 1
                userFrameAvg = userFrameTotal / area
                avgPerFrame[trace[0]] += userFrameAvg
                avgPerUser[user] += userFrameAvg
            avgPerUser[user] = avgPerUser[user] / len(self.framelist)
        for frame in avgPerFrame:
            avgPerFrame[frame] /= len(self.traces)
        avg1 = sum(avgPerFrame.values()) / len(self.framelist)
        avg2 = sum(avgPerUser.values()) / len(self.traces)
        print(avg1)
        print(avg2)
                
            
                



    def generateResMapArrs(self, heatMapArrs):
        resMapArrs = []
        for Map in heatMapArrs:
            heatMap = Map[0]
            resMapArray = np.zeros((self.rows, self.cols))
            max_look = np.amax(heatMap)
            r = 0
            for row in heatMap:
                c = 0
                for col in row:
                    val = -1 if int(col)==0 else int(col * 5 / max_look)  # val = -1 if no one is looking, else 0-4
                    if val == 5:
                        val = 4
                    resMapArray[r][c] = val
                    c += 1
                r +=1
            resMap = (resMapArray, Map[1].replace('heatMaps', 'resMaps'), Map[2])
            resMapArrs.append(resMap)
        return resMapArrs

    def compressImg(self, Map):
        # print("splitting...")
        self.splitImage(Map[2], Map[0])
        # print("complete")
        fullImgArr = np.zeros_like(self.splitImgs[0])
        fullImg = Image.fromarray(fullImgArr)
        for imgArr in self.splitImgs:
            img = Image.fromarray(imgArr)
            fullImg.paste(img, (0, 0), img)
        fullImg = fullImg.convert('RGB')
        return fullImg
        
    def splitImage(self, frame, resMap):
        self.splitImgs = []
        unitWidth = self.imagesize[0] / self.cols
        unitHeight = self.imagesize[1] / self.rows
        frameName = 'frame{}.jpg'.format(frame)
        
        blackImg = Image.new('RGBA', self.imagesize, color = 0)
        imgArray = np.asarray(blackImg)
        self.splitImgs.append(imgArray)
        for i in range(0, 5):
            # print("calculations")
            frameImg = Image.open(os.path.join(self.frameimgs, frameName))
            reduction = 4-i
            reductionFactor = (2**reduction)
            frameImg = frameImg.resize((int(self.imagesize[0] / reductionFactor), int(self.imagesize[1] / reductionFactor)))
            frameImg = frameImg.resize(self.imagesize)
            mask = Image.new('L', frameImg.size, color = 255)
            draw = ImageDraw.Draw(mask)
            # transp_height = 0
            
            y_i = 0
            for r in resMap:
                transp_width = 0
                x_i = 0
                for c in r:
                    
                    if c != i:
                        transp_width += unitWidth 
                    else:
                        transp_area = (x_i, y_i, x_i + transp_width - 1, y_i + unitHeight - 1)
                        # print(transp_area)
                        if transp_width != 0:
                            draw.rectangle(transp_area, fill = 0)
                        x_i += transp_width + unitWidth
                        transp_width = 0 
                transp_area = (x_i, y_i, x_i + transp_width - 1, y_i + unitHeight - 1)
                # print(transp_area)
                if transp_width != 0:
                    draw.rectangle(transp_area, fill = 0)
                y_i += unitHeight
            # draw.rectangle((100, 10, 300, 190), fill = 0)
            # print("calcs complete")
            frameImg.putalpha(mask)
            # print('saving')
            # dirname = os.getcwd()
            # im = frameImg.convert('RGB')
            imgArray = np.asarray(frameImg)
            self.splitImgs.append(imgArray)
            # print(imgArray)
            # frameImg.save(f'{dirname}/testFunc/splitImgs/frame{frame}_res_{i}.png', quality=1)
            # im.save(f'frame{frame}_res_{i}.jpg', quality=1)
            
            # print('complete')

    def createCompressedVideo(self, fps, heatMapArrays, videoName = 'compressedVideo.avi', scalingFunction = 'semiCrcl'):
        # heatMapArrays = self.generateHeatMapArrs(scalingFunction)
        resMapArrays = self.generateResMapArrs(heatMapArrays)
        out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, self.imagesize)
        numofframes = len(resMapArrays)
        progress = 0
        print(f'Creating {videoName} video')
        for Map in resMapArrays:
            fullImg = self.compressImg(Map)
            fullImg.save('temp.jpg', quality=95)
            fullImg = cv2.imread('temp.jpg')
            out.write(fullImg)
            os.remove('temp.jpg')
            progress += (100 / numofframes)
            print("Progress " + str(int(progress)) + "%", end='\r', flush=True)

        out.release()
        # self.renderMapImgs([resMapArrays[len(resMapArrays) - 1]])
        print()