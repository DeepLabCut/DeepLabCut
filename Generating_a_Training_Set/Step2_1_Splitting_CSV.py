# credits to @sneakers-the-rat
import pandas as pd
import tkinter
from tkinter import filedialog
import sys
import os
sys.path.append(os.getcwd().split('Generating_a_Training_Set')[0])
from myconfig import Task, bodyparts

# ask user for location of single .csv output from fiji
root = tkinter.Tk()
filename = filedialog.askopenfilename(parent=root, initialdir="/",
                                        title='Please select a csv file')

# get basefolder, and the name of the data folder assuming there is only 1 video for now
basefolder = 'data-' + Task + '/'
folder = [name for name in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder, name))][0]

# load csv, iterate over nth value in a grouping by frame, save to bodyparts files
dframe = pd.read_csv(filename)
frame_grouped = dframe.groupby('Slice')
for i, part in enumerate(bodyparts):
    part_df = frame_grouped.nth(i)
    part_fn = basefolder + folder + '/{}.csv'.format(part)
    part_df.to_csv(part_fn)
