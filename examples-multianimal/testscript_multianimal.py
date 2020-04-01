import os
os.environ['DLClight'] = 'True'
import deeplabcut
import numpy as np
import pandas as pd
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions

TASK = 'multi_birdie'
SCORER = 'dlc_team'
NUM_FRAMES = 5
TRAIN_SIZE = 0.8
NET = 'resnet_50'
N_ITER = 5


basepath = os.path.dirname(os.path.abspath(__file__))
video = 'montblanc.mov'
video_path = os.path.join(basepath, f'MontBlanc-Daniel-2019-12-16/videos/{video}')
print('Creating project...')
config_path = deeplabcut.create_new_project(TASK, SCORER, [video_path],
                                            copy_videos=True, multianimal=True)
print('Project created.')

print('Editing config...')
cfg = auxiliaryfunctions.read_config(config_path)
cfg['numframes2pick'] = NUM_FRAMES
cfg['TrainingFraction'] = [TRAIN_SIZE]
auxiliaryfunctions.write_config(config_path, cfg)
print('Config edited.')

print('Extracting frames...')
deeplabcut.extract_frames(config_path, mode='automatic', userfeedback=False)
print('Frames extracted.')

print('Creating artificial data...')
rel_folder = os.path.join('labeled-data', os.path.splitext(video)[0])
image_folder = os.path.join(cfg['project_path'], rel_folder)
images = auxiliaryfunctions.listfilesofaparticulartypeinfolder(image_folder, 'png')
n_animals = len(cfg['individuals'])
animals, bodyparts_single, bodyparts_multi = auxfun_multianimal.extractindividualsandbodyparts(cfg)
animals_id = [i for i in range(n_animals) for _ in bodyparts_multi] + [n_animals] * len(bodyparts_single)
map_ = dict(zip(range(len(animals)), animals))
individuals = [map_[ind] for ind in animals_id for _ in range(2)]
scorer = [SCORER] * len(individuals)
coords = ['x', 'y'] * len(animals_id)
bodyparts = [bp for _ in range(n_animals) for bp in bodyparts_multi for _ in range(2)]
bodyparts += [bp for bp in bodyparts_single for _ in range(2)]
columns = pd.MultiIndex.from_arrays([scorer, individuals, bodyparts, coords],
                                    names=['scorer', 'individuals', 'bodyparts', 'coords'])
index = [os.path.join(rel_folder, image) for image in images]
fake_data = np.tile(np.repeat(50 * np.arange(len(animals_id)) + 100, 2), (len(index), 1))
df = pd.DataFrame(fake_data, index=index, columns=columns)
output_path = os.path.join(image_folder, f'CollectedData_{SCORER}.csv')
df.to_csv(output_path)
df.to_hdf(output_path.replace('csv', 'h5'), 'df_with_missing', format='table', mode='w')
print('Artificial data created.')

print('Checking labels...')
deeplabcut.check_labels(config_path, draw_skeleton=False)
print('Labels checked.')

print('Creating train dataset...')
deeplabcut.create_multianimaltraining_dataset(config_path, net_type=NET)
print('Train dataset created.')

print('Editing pose config...')
model_folder = auxiliaryfunctions.GetModelFolder(cfg['TrainingFraction'][0], 1, cfg, cfg['project_path'])
pose_config_path = os.path.join(model_folder, 'train/pose_cfg.yaml')
pose_cfg = deeplabcut.auxiliaryfunctions.read_plainconfig(pose_config_path)
pose_cfg['global_scale'] = .1
pose_cfg['save_iters'] = N_ITER
pose_cfg['display_iters'] = N_ITER // 2
pose_cfg['multi_step'] = [[0.001, N_ITER]]

deeplabcut.auxiliaryfunctions.write_plainconfig(pose_config_path, pose_cfg)
print('Pose config edited.')

print('Training network...')
deeplabcut.train_network(config_path)
print('Network trained.')

print('Evaluating network...')
deeplabcut.evaluate_network(config_path, plotting=True)
print('Network evaluated.')

new_video_path = deeplabcut.ShortenVideo(video_path, start='00:00:00', stop='00:00:00.4',
                                         outsuffix='short', outpath=os.path.join(cfg['project_path'], 'videos'))

print('Analyzing video...')
deeplabcut.analyze_videos(config_path, [new_video_path], save_as_csv=True, dynamic=(True, .1, 5))
print('Video analyzed.')


#model???
#deeplabcut.create_video_with_all_detections(path_config_file, video[0], model)


'''
print('Plotting trajectories...')
deeplabcut.plot_trajectories(config_path, [new_video_path])
print('Trajectory plotted.')

print('Creating labeled video...')
deeplabcut.create_labeled_video(config_path, [new_video_path], save_frames=False, color_by='animal')
print('Labeled video created.')
'''
