import copy
import os
import pandas as pd
from shutil import copytree, rmtree
import numpy as np
import yaml

from deeplabcut.create_project import yaml_config_template


def combine_coder_labels(config_file1, config_file2):
    with open(str(config_file1), 'r') as ymlfile1:
        cfg1 = yaml.load(ymlfile1)
    with open(str(config_file2), 'r') as ymlfile2:
        cfg2 = yaml.load(ymlfile2)

    scorer1 = cfg1['scorer']
    scorer2 = cfg2['scorer']
    assert(cfg1['bodyparts']==cfg2['bodyparts'])
    project1_path=cfg1['project_path']
    project2_path = cfg2['project_path']
    bodyParts = cfg1['bodyparts']

    new_project_path=project1_path.replace(scorer1,'%s%s' % (scorer1,scorer2))
    os.mkdir(new_project_path)
    os.mkdir(os.path.join(new_project_path,'dlc-models'))
    copytree(os.path.join(project1_path,'labeled-data'),os.path.join(new_project_path,'labeled-data'))
    os.mkdir(os.path.join(new_project_path,'training-datasets'))
    copytree(os.path.join(project1_path,'videos'), os.path.join(new_project_path,'videos'))

    new_cfg = copy.copy(cfg1)
    new_cfg['scorer'] = '%s%s' % (scorer1, scorer2)
    new_cfg['project_path'] = new_project_path
    new_video_sets={}
    for video_fname in cfg1['video_sets']:
        new_video_sets[video_fname.replace(scorer1,'%s%s' % (scorer1,scorer2))]=cfg1['video_sets'][video_fname]
    new_cfg['video_sets']=new_video_sets
    yaml_config_template(os.path.join(new_project_path,'config.yaml'), new_cfg)

    for label_dir in os.listdir(os.path.join(project1_path,'labeled-data')):
        if os.path.isdir(os.path.join(project1_path,'labeled-data',label_dir)) and not 'labeled' in label_dir:
            df1_name=os.path.join(project1_path, 'labeled-data', label_dir, 'CollectedData_' + scorer1 + '.h5')
            df1=pd.read_hdf(df1_name, 'df_with_missing')

            df2_name = os.path.join(project2_path, 'labeled-data', label_dir, 'CollectedData_' + scorer2 + '.h5')
            df2 = pd.read_hdf(df2_name, 'df_with_missing')

            new_df=None
            a = np.empty((len(df1.index.tolist()), 2,))
            a[:] = np.nan
            for bodypart in bodyParts:
                index = pd.MultiIndex.from_product([['%s%s' % (scorer1,scorer2)], [bodypart], ['x', 'y']],
                                                   names=['scorer', 'bodyparts', 'coords'])
                # frame = pd.DataFrame(a, columns = index, index = self.index)
                frame = pd.DataFrame(a, columns=index, index=df1.index.tolist())
                new_df = pd.concat([new_df, frame], axis=1)

            for bodypart in bodyParts:
                for frame_filename in df1.index.tolist():
                    coord1_x = df1.loc[frame_filename][scorer1, bodypart, 'x']
                    coord1_y = df1.loc[frame_filename][scorer1, bodypart, 'y']
                    coord2_x = df2.loc[frame_filename.replace('/','\\')][scorer2, bodypart, 'x']
                    coord2_y = df2.loc[frame_filename.replace('/','\\')][scorer2, bodypart, 'y']
                    new_df.loc[frame_filename]['%s%s' % (scorer1,scorer2), bodypart, 'x']=np.nanmean([coord1_x, coord2_x])
                    new_df.loc[frame_filename]['%s%s' % (scorer1,scorer2), bodypart, 'y']=np.nanmean([coord1_y, coord2_y])

            new_df.to_csv(os.path.join(new_project_path, 'labeled-data', label_dir, "CollectedData_" + scorer1 + scorer2 + ".csv"))
            new_df.to_hdf(os.path.join(new_project_path, 'labeled-data', label_dir, "CollectedData_" + scorer1 + scorer2 + '.h5'), 'df_with_missing',
                          format='table', mode='w')
            os.remove(os.path.join(new_project_path,'labeled-data',label_dir,"CollectedData_" + scorer1 + ".csv"))
            os.remove(os.path.join(new_project_path, 'labeled-data', label_dir, "CollectedData_" + scorer1 + ".h5"))
        else:
            rmtree(os.path.join(new_project_path,'labeled-data',label_dir))

if __name__=='__main__':
    combine_coder_labels('/home/bonaiuto/Dropbox/Projects/inProgress/tool_learning/data/kinematics_tracking/monkey_grasp_front-Jimmy-2019-01-15/config.yaml',
                         '/home/bonaiuto/Dropbox/Projects/inProgress/tool_learning/data/kinematics_tracking/monkey_grasp_front-Sebastien-2019-01-15/config.yaml')