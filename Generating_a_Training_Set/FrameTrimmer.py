"""
B Forys, brandon.forys@alumni.ubc.ca

This script deletes unlabelled images. Only use this script if the bodypart you
are analyzing does not appear in every frame, and if there are frames that you
did not label.
Please back up all images before running the script, as this script deletes
images.
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd().split('Generating_a_Training_Set')[0])
from myconfig import Task, bodyparts, Scorers

basefolder = 'data-' + Task + '/'

for scorer in Scorers:
    os.chdir(basefolder)
    # Make list of different video data sets / each one has its own folder
    folders = [
        videodatasets for videodatasets in os.listdir(os.curdir)
        if os.path.isdir(videodatasets)
    ]

    for folder in folders:
        os.chdir(folder)
        files_img = np.sort([fn for fn in os.listdir(os.curdir)
        if ("img" in fn and ".png" in fn and "_labelled" not in fn)])
        print(folder)

        files_csv = [
            fn for fn in os.listdir(os.curdir)
            if("csv" in fn)
        ]

        for file in files_csv:
            dframe = pd.read_csv(file)
            df_select = dframe[dframe.index % 6 == 0]
            slice = df_select["Slice"]
            a = 0
            c = 0
            f = 0
            i = []
            for image in files_img:
                ir = (a, image)
                i.append(ir)
                a += 1

            for s in slice:
                ## Uncomment below for debugging

                # try:
                #     print("image " + str(i[f][0]))
                # except:
                #     print("Reached end of images!")
                #     break
                # print("slice " + str(s))

                if(i[f][0] != s and os.path.isfile(i[f][1])):
                    removal_list = []

                    for image in range(i[f][0], s-1):
                        il = (i[f][0])
                        removal_list.append(il)
                        f += 1
                        #print("f = " + str(i[f][0]))

                    for img in removal_list:
                        c += 1
                        ## Comment the next line out for testing without deletion
                        os.remove(files_img[img])
                        print("Image " + str(files_img[img]) + " removed as it did not correspond to slice " + str(s))
                    removal_list = []
                    f = s
            print(str(c) + " images removed from " + folder + "!")
