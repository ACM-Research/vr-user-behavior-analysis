import os

import numpy as np
from PIL import Image, ImageDraw

import cv2


class Deresolutionizer:

    def __init__(self, dParser):
        self.rows = dParser.rows
        self.cols = dParser.cols
        self.imagesize = dParser.imagesize
        self.frameimgs = dParser.frameimgs


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
                    val = int(col * 5 / max_look)
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
        
        for i in range(0, 5):
            # print("calculations")
            frameImg = Image.open(os.path.join(self.frameimgs, frameName))
            reduction = 4-i
            reductionFactor = (2**reduction) if i != 0 else (4**reduction)
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
                        transp_area = (x_i, y_i, x_i + transp_width, y_i + unitHeight)
                        # print(transp_area)
                        if transp_width != 0:
                            draw.rectangle(transp_area, fill = 0)
                        x_i += transp_width + unitWidth
                        transp_width = 0 
                transp_area = (x_i, y_i, x_i + transp_width, y_i + unitHeight)
                # print(transp_area)
                if transp_width != 0:
                    draw.rectangle(transp_area, fill = 0)
                y_i += unitHeight
            # draw.rectangle((100, 10, 300, 190), fill = 0)
            # print("calcs complete")
            frameImg.putalpha(mask)
            # print('saving')
            # im = frameImg.convert('RGB')
            imgArray = np.asarray(frameImg)
            self.splitImgs.append(imgArray)
            # print(imgArray)
            # frameImg.save(f'{dirname}/testFunc/splitImgs/frame{frame}_res_{i}.png')
            # im.save(f'frame{frame}_res_{i}.jpg')
            
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