import os
os.environ['DLClight'] = 'True'
import deeplabcut

####################################################### 
##### Visualizing scmap, locref and paf
#######################################################

# Fully integrated version that stores plots in /evaluation-results just like other results.

path_config_file = '/examplepath/config.yaml'


deeplabcut.extract_save_all_maps(path_config_file, shuffle=shuffle) #, Indices=[0, 5])

'''
# A version with more control 
# that stores plots wherever you like and allows you to adjust plots:

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
os.environ['DLClight'] = 'True'
from deeplabcut.pose_estimation_tensorflow import visualizemaps
from deeplabcut.utils.auxiliaryfunctions import attempttomakefolder

dest_folder = os.path.join(os.getcwd(), 'maps')
attempttomakefolder(dest_folder)

data = visualizemaps.extract_maps(path_config_file, 1)
maps = data[0.95][-1] #indexing by 0.95% training data (adjust if nec. )
dest_path = os.path.join(dest_folder, 'Img{}_{}_{}.png')
for imagenr in tqdm(maps): 
    image, scmap, locref, paf, bptnames, pafgraph, imname, trainingframe = maps[imagenr]
    scmap, (locref_x, locref_y), paf = visualizemaps.resize_all_maps(image, scmap, locref, paf)
    fig1, _ = visualizemaps.visualize_scoremaps(image, scmap, labels=bptnames)
    fig2, _ = visualizemaps.visualize_locrefs(image, scmap, locref_x, locref_y, labels=bptnames)
    fig3, _ = visualizemaps.visualize_locrefs(image, scmap, locref_x, locref_y, zoom_width=100, labels=bptnames)
    fig4, _ = visualizemaps.visualize_paf(image, paf, pafgraph, labels=bptnames)

    label = 'train' if trainingframe else 'test'
    fig1.savefig(dest_path.format(imagenr, 'scmap', label))
    fig2.savefig(dest_path.format(imagenr, 'locref', label))
    fig3.savefig(dest_path.format(imagenr, 'locrefzoom', label))
    fig4.savefig(dest_path.format(imagenr, 'paf', label))
    plt.close('all')



'''