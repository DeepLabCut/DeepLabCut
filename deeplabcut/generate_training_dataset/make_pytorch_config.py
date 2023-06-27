import torch
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions

BACKBONE_OUT_CHANNELS = {
    'resnet-50': 2048,
    'resnet-50': 2048,
    'resnet-50': 2048,
    'mobilenet_v2_1.0': 1280,
    'mobilenet_v2_0.75': 1280,
    'mobilenet_v2_0.5': 1280,
    'mobilenet_v2_0.35': 1280,
    'efficientnet-b0': 1280,
    'efficientnet-b1': 1280, 
    'efficientnet-b2': 1408,
    'efficientnet-b3': 1536,
    'efficientnet-b4': 1792,
    'efficientnet-b5': 2048,
    'efficientnet-b6': 2304,
    'efficientnet-b7': 2560,
    'efficientnet-b8': 2816,
    'hrnet_w18': 270,
    'hrnet_w32': 480,
    'hrnet_w48': 720,
}


def make_pytorch_config(project_config: dict, net_type: str, augmenter_type: str = 'default', config_template: dict=None):
    '''
    Currently supported net types :
        Single Animal :
            - resnet-50
            - mobilenet_v2_1.0
            - mobilenet_v2_0.75
            - mobilenet_v2_0.5
            - mobilenet_v2_0.35
            - efficientnet-b0
            - efficientnet-b1
            - efficientnet-b2
            - efficientnet-b3
            - efficientnet-b4
            - efficientnet-b5
            - efficientnet-b6
            - efficientnet-b7
            - efficientnet-b8
            - hrnet_w18
            - hrnet_w32
            - hrnet_w48

        Multi Animal:
            - dekr_w18
            - dekr_w32
            - dekr_w48

    '''

    single_animal_nets = ['resnet_50'
            , 'mobilenet_v2_1.0'
            , 'mobilenet_v2_0.75'
            , 'mobilenet_v2_0.5'
            , 'mobilenet_v2_0.35'
            , 'efficientnet-b0'
            , 'efficientnet-b1'
            , 'efficientnet-b2'
            , 'efficientnet-b3'
            , 'efficientnet-b4'
            , 'efficientnet-b5'
            , 'efficientnet-b6'
            , 'efficientnet-b7'
            , 'efficientnet-b8'
            , 'hrnet_w18'
            , 'hrnet_w32'
            , 'hrnet_w48']
    
    multi_animal_nets = ['dekr_w18'
            , 'dekr_w32'
            , 'dekr_w48']

    bodyparts = auxiliaryfunctions.get_bodyparts(project_config)
    num_joints = len(bodyparts)
    pytorch_config = config_template
    pytorch_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if net_type in single_animal_nets:
        pytorch_config['model']['heatmap_head']['channels'][-1] = num_joints
        pytorch_config['model']['locref_head']['channels'][-1] = 2*num_joints
        pytorch_config['model']['target_generator']['num_joints'] = num_joints
        pytorch_config['predictor']['num_animals'] = 1

        if 'efficientnet' in net_type:
            raise NotImplementedError('efficientnet config not yet implemented')
        elif 'mobilenetv2' in net_type:
            raise NotImplementedError('mobilenet config not yet implemented')
        elif 'hrnet' in net_type:
            raise NotImplementedError('hrnet config not yet implemented')

    elif net_type in multi_animal_nets:
        num_animals = len(project_config.get('individuals', [0]))
        if 'dekr' in net_type:
            version = net_type.split('_')[-1]
            backbone_type = 'hrnet_' + version
            num_offset_per_kpt = 15
            pytorch_config['data']['auto_padding'] = {
                'min_height': 64,
                'min_width': 64,
                'pad_width_divisor': 32,
                'pad_height_divisor': 32,
            }
            pytorch_config['model']['backbone'] = {
                'type': 'HRNet',
                'model_name': 'hrnet_' + version
            }
            pytorch_config['model']['heatmap_head']= {
                'type': 'HeatmapDEKRHead',
                'channels': [
                    BACKBONE_OUT_CHANNELS[backbone_type],
                    64,
                    num_joints + 1
                ], # +1 since we need center
                'num_blocks': 1,
                'dilation_rate': 1,
                'final_conv_kernel': 1,
            }
            pytorch_config['model']['locref_head']= {
                'type': 'OffsetDEKRHead',
                'channels': [
                    BACKBONE_OUT_CHANNELS[backbone_type],
                    num_offset_per_kpt*num_joints,
                    num_joints
                ],
                'num_offset_per_kpt' : num_offset_per_kpt,
                'num_blocks': 1,
                'dilation_rate': 1,
                'final_conv_kernel': 1
            }
            pytorch_config['model']['target_generator']= {
                'type': 'DEKRGenerator',
                'num_joints': num_joints,
                'pos_dist_thresh': 17,
            }

            pytorch_config['predictor']= {
                'type': 'DEKRPredictor',
                'num_animals':  num_animals,
            }

            pytorch_config['with_center'] = True
        else:
            raise NotImplementedError('Currently no other model than dekr are implemented')
    
    else:
        raise ValueError('This net type is not supported by pytorch verison')
    
    if augmenter_type == None:
        pytorch_config['data'] = {}
    elif augmenter_type != 'default' and augmenter_type != None:
        raise NotImplementedError('Other augmentations than default are not implemented')
    
    return pytorch_config
