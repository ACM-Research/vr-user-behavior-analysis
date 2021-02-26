import math
import os
from typing import Tuple, List, Dict, Callable

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import seaborn as sb

class DataParser:

    def __init__(self, basedir):
        self.usertracepath = f"{basedir}/Data/UserTracesByVideo/23/"
        self.frameimgs = f"{basedir}/Data/VideosData/Videos/SourceFrames/23/"
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
        colwidth = self.imagesize[0] / 6 # this can be changed
        rowheight = self.imagesize[1] / 3
        for trace_rows, userid in self.all_user_traces:
            
            trace_row = trace_rows[frame - 1]  # Be careful about indexing!!
            arr = [trace_row[5], trace_row[6], trace_row[7]]
            x, y = self.convvec2angl(arr)
            x = ((x+180)/360) * self.imagesize[0]
            y = ((90-y)/180) * self.imagesize[1]
            print(f"{x}, {y}")
            x_index = int(x / colwidth)
            y_index = int(y / rowheight)
            self.usertraces.append((userid, frame, (x_index, y_index)))
                
    def generateHeatMap(self):
        fig, ax = plt.subplots(figsize=(12,6))
        ax.axis('off')
        frame = 1291
        self.convertusertraces(frame)
        print(self.usertraces)
        heatMapArr = np.zeros((3, 6))
        for usertrace in self.usertraces:
            indices = usertrace[2]
            heatMapArr[indices[1]][indices[0]] += 1
        sb.heatmap(heatMapArr)
        plt.savefig('heatmap.png')
        

def main():
    data = DataParser('D:/Projects/vr-user-behavior-analysis/')
    data.generateHeatMap()

if __name__ == "__main__":
    main()