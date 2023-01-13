import os
import pickle
from typing import List
import numpy as np
import pandas as pd
import json
from .base import BaseProject

class COCOProject(BaseProject):
    """
    TODO
    """

    def __init__(self,
                 proj_root,
                 shuffle: int = 0,
                 image_id_offset: int = 0,
                 keys_to_load: List[str] = ['images', 'annotations']):
        super().__init__()
        self.proj_root = proj_root
        self.keys_to_load = keys_to_load

        self.train_json_obj = self._load_json('train.json') if shuffle is None else self._load_json(f'train_shuffle{shuffle}.json')
        self.test_json_obj = self._load_json('test.json') if shuffle is None else self._load_json(f'test_shuffle{shuffle}.json')
        
    def _load_json(self, json_fn):        
        path = os.path.join(self.proj_root, 'annotations', json_fn)        
        with open(path, 'r') as f:            
            json_obj = json.load(f)
        return json_obj

    def load_split(self):
        '''
        We expected that coco project has train test split in train test json already
        '''
        pass

    def convert2dict(self,
                     mode: str = 'train'):

        json_obj = getattr(self, f'{mode}_json_obj')

        
        for image in self.images
            image_path = image['file_name']
            #if os.sep not in image_path:
            # assuming the file_name is mmpose style, i.e. only the image name is stored
            # so we need to add back absolute path
            image['file_name'] = os.path.join(self.proj_root, 'images', image_path)
        

        for key in self.keys_to_load:
            setattr(self, key, json_obj[key])
