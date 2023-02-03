import argparse
import deeplabcut.pose_estimation_pytorch as dlc
import numpy as np
import os
import torch
from deeplabcut import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis.utils import build_pose_model
from deeplabcut.pose_estimation_pytorch.solvers.inference import get_prediction, get_scores
from deeplabcut.pose_estimation_pytorch.solvers.utils import get_paths, get_results_filename, save_predictions
from deeplabcut.pose_estimation_tensorflow import Plotting
from typing import Union


def inference_network(
        config_path: str,
        shuffle: int = 0,
        model_prefix: str = "",
        load_epoch: int = 49,
        stride: int = 8,
        transform: object = None,
        plot: bool = False,
        evaluate: bool = True):
    # reading pytorch config
    cfg = auxiliaryfunctions.read_config(config_path)
    train_fraction = cfg["TrainingFraction"]
    modelfolder = os.path.join(
        cfg["project_path"],
        auxiliaryfunctions.get_model_folder(
            train_fraction[0], shuffle, cfg, modelprefix=model_prefix,
        ),
    )
    pytorch_config_path = os.path.join(modelfolder, "train", "pytorch_config.yaml")
    config = auxiliaryfunctions.read_config(pytorch_config_path)

    batch_size = config['batch_size']
    project = dlc.DLCProject(shuffle=shuffle,
                             proj_root=config['project_root'])

    valid_dataset = dlc.PoseDataset(project,
                                    transform=transform,
                                    mode='test')
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
    names = get_paths(train_fraction=train_fraction[0],
                      model_prefix=model_prefix,
                      shuffle=shuffle,
                      cfg=valid_dataset.cfg,
                      train_iterations=load_epoch)

    results_filename = get_results_filename(names['evaluation_folder'],
                                            names['dlc_scorer'],
                                            names['dlc_scorer_legacy'],
                                            names['model_path'][:-3])

    pose_cfg = auxiliaryfunctions.read_config(config['pose_cfg_path'])
    model = build_pose_model(config['model'], pose_cfg)
    model.load_state_dict(torch.load(names['model_path']))

    target_df = valid_dataset.dataframe
    predicted_poses = []
    model.eval()
    with torch.no_grad():
        for item in valid_dataloader:
            if isinstance(item, tuple) or (isinstance, list):
                item = item[0]
            output = model(item)
            predictions = get_prediction(pose_cfg, output, stride)
            predicted_poses.append(predictions)
        predicted_poses = np.array(predicted_poses)

    predicted_df = save_predictions(names,
                                    pose_cfg,
                                    target_df.index,
                                    predicted_poses.reshape(target_df.index.shape[0], -1),
                                    results_filename)
    if plot:
        foldername = f'{names["evaluation_folder"]}/LabeledImages_{names["dlc_scorer"]}-{load_epoch}'
        auxiliaryfunctions.attempttomakefolder(foldername)
        combined_df = predicted_df.merge(target_df,
                                         left_index=True,
                                         right_index=True)
        Plotting(valid_dataset.cfg,
                 valid_dataset.cfg['bodyparts'],
                 names['dlc_scorer'],
                 predicted_df.index,
                 combined_df,
                 foldername)
    if evaluate:
        rmse, rmes_p = get_scores(pose_cfg,
                                  predicted_df,
                                  target_df)
        print(f'RMSE: {rmse}, RMSE pcutoff: {rmes_p}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--modelprefix", type=str, default="")
    parser.add_argument("--load_epoch", type=int, default=49)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=True)
    args = parser.parse_args()
    inference_network(config_path=args.config_path,
                      shuffle=args.shuffle,
                      model_prefix=args.modelprefix,
                      load_epoch=args.load_epoch,
                      plot=args.plot,
                      evaluate=args.evaluate)
