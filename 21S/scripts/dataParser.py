import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
import seaborn as sb
from pathlib import Path

import cv2


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
        self.videoId = videoId
        
    def createControlImages(self):
        controlPath = f'{os.getcwd()}/21S/data/control/{self.videoId}'
        if not os.path.isdir(controlPath):
            frames = [f'{self.frameimgs}/frame{frame}.jpg' for frame in self.frameList()]
            Path(controlPath).mkdir(777, parents=True, exist_ok=True)
            for frame in frames:
                imgPath = f'{frame}'
                img = Image.open(frame)
                img.save(imgPath, quality=95)


    def frameList(self):
        framenums = []
        index = 1
        frames = [int(frame[5:-4]) for frame in os.listdir(self.frameimgs)]
        frames.sort()
        for frame in frames:
            framenums.append(frame)
            
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
    
    def convertTracesForAllUsers(self):
        traces_per_user = {}
        for user in self.all_user_traces:
            traces_per_user[user[1]] = self.convertTracesPerUser(user)
        return traces_per_user


    def convertTracesPerUser(self, user):
        traces = []
        colwidth = self.imagesize[0] / self.cols # this can be changed
        rowheight = self.imagesize[1] / self.rows
        trace_rows, userid = user
        for frame in self.frameList():
            trace_row = trace_rows[frame - 1]  # Be careful about indexing!!
            arr = [trace_row[5], trace_row[6], trace_row[7]]
            x, y = self.convvec2angl(arr)
            x = ((x+180)/360) * self.imagesize[0]
            y = ((90-y)/180) * self.imagesize[1]
            x_index = int(x / colwidth)
            y_index = int(y / rowheight)
            traces.append((frame, (x_index, y_index)))
        return traces

    def createControlVideo(self, fps, videoName='Control.avi'):
        frames = [f'{self.frameimgs}/frame{frame}.jpg' for frame in self.frameList()]
        # print(frames)
        out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'DIVX'), fps, self.imagesize)
        print(f'Creating {videoName} video')
        numofframes = len(frames)
        progress = 0
        for frame in frames:
            img = cv2.imread(frame)
            out.write(img)
            progress += (100 / numofframes)
            print("Progress " + str(int(progress)) + "%", end='\r', flush=True)
        out.release()
        print()

    




#Will need to split up source videos into frames once ML is used