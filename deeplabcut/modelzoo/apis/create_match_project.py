import modelzoo
from modelzoo.datasets import (MaDLCPoseDataset,
                               MultiSourceDataset,
                               COCOPoseDataset,
                               SingleDLCPoseDataset)
import matplotlib.pyplot as plt
import numpy as np
import json
import deeplabcut
import os
import sys
import pathlib

class DomainMatcher:
    def __init__(self,
                 pretrain_weights,
                 supermodel_name,
                 ref_proj_root,
                 ref_modelprefix = '',
                 scale_list = [],
                 shuffle = 0,
                 dataset_name = 'ref_dataset'
                 ):
        # assuming target project is a single DLC project   
        # maybe save the cost_matrix somewhere
        self.pretrain_weights = pretrain_weights
        self.supermodel_name = supermodel_name
        self.ref_proj_root = ref_proj_root
        self.ref_modelprefix = ref_modelprefix
        self.scale_list = scale_list
        self.shuffle = shuffle
        self.dataset_name = dataset_name

    def plot_cost_matrix(self, cost_obj, output_folder):

        matrix = cost_obj['matrix']
        gt_bodyparts = cost_obj['gt_bodyparts']
        pseudo_bodyparts = cost_obj['pseudo_bodyparts']

        matrix/= np.max(matrix)
        
        fig, ax = plt.subplots()        
        heatmap = ax.pcolor(matrix, cmap = plt.cm.Blues, vmin = 0, vmax = 1)
        ax.set_xticks(np.arange(matrix.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(matrix.shape[0]) + 0.5, minor=False)
        ax.set_xlim(0, int(matrix.shape[1]))
        ax.set_ylim(0, int(matrix.shape[0]))
        ax.set_yticklabels(gt_bodyparts, minor=False)    
        ax.set_xticklabels(pseudo_bodyparts, minor=False)
        ax.set_title('cost matrix')
        plt.xticks(rotation=90)    

        fig = plt.gcf()
        fig.tight_layout()

        filename = os.path.join(output_folder,
                                f'cost_matrix_{self.dataset_name}.png')
        
        plt.savefig(filename,
                    loc ='center',
                    dpi = 900
                )

    def export_conversion_table(self,
                                conversion_table,
                                new_proj_root
                                ):
        with open(os.path.join(new_proj_root,
                               'conversion_table.json'), 'w') as f:
            json.dump(conversion_table, f, indent = 4)        
        
    def match(self, adaptation_video = ''):

        if adaptation_video:

            assert os.path.exists(adaptation_video)
            
            from modelzoo.apis import SpatioTemporalAdaptation
            
            adapter = SpatioTemporalAdaptation(adaptation_video,
                                               self.pretrain_weights,
                                               self.ref_proj_root,
                                               self.supermodel_name,
                                               scale_list = self.scale_list,
                                               videotype = pathlib(adaptation_video).suffix)
            
            adapter.before_adapt_inference()
            adapter.adaptation_training()
            
            
            # no need to do spatial pyramid again
            self.scale_list = []
            # WIP
            pass
                               
                
        conversion_table, cost_obj = modelzoo.match_domain(self.pretrain_weights,
                                                              self.supermodel_name,
                                                              self.ref_proj_root,
                                                              target_modelprefix = self.ref_modelprefix,

                                                              target_shuffle = self.shuffle,
                                                              scale_list = self.scale_list)


        
        '''

        dataset = SingleDLCPoseDataset(self.ref_proj_root,
                                       self.dataset_name,
                                       shuffle = self.shuffle)

        dataset.project_with_conversion_table(table_dict = conversion_table)
        
        dataset.summary()

        # generates a new project math
        new_project_root = self.ref_proj_root + '_supermodel'

        dataset.materialize(new_project_root,
                            framework = 'madlc')
        

        # save the conversion table in json format

        with open(os.path.join(new_project_root,
                       'conversion_table.json'), 'w' ) as f:
    
            json.dump(conversion_table, f, indent = 4)        

        # save the cost matrix in png format


        self.plot_cost_matrix(cost_obj, new_project_root)
        
        '''
        return conversion_table, cost_obj
