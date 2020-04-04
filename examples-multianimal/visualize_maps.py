import matplotlib.pyplot as plt
import os
os.environ['DLClight'] = 'True'
from deeplabcut.pose_estimation_tensorflow import visualizemaps
from deeplabcut.utils.auxiliaryfunctions import attempttomakefolder
from tqdm import tqdm


config = '/Users/Jessy/Documents/PycharmProjects/dlcdev/datasets/mwm-penguins-2020-03-31/config.yaml'
dest_folder = os.path.join(os.getcwd(), 'maps')
attempttomakefolder(dest_folder)
data = visualizemaps.extract_maps(config, 1)
maps = data[0.95][-1]
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
