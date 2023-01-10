import os
import os.path
import pickle
import time
from pathlib import Path
import glob
import deeplabcut
from deeplabcut.modelzoo.api import SpatiotemporalAdaptation

def video_inference_superanimal(
    videos,
    superanimal_name,
    scale_list = [],
    videotype = '.mp4',
    video_adapt = False,
    pcutoff = 0.1
):

    
    for video in videos:
        vname = Path(video).stem        
        modelfolder = superanimal_name + '_' + vname + '_weights'
        adapter = SpatiotemporalAdaptation(video,
                                           superanimal_name,
                                           modelfolder = modelfolder,
                                           videotype = videotype,
                                           scale_list = scale_list)

        if not video_adapt:
            adapter.before_adapt_inference(make_video = True,
                                           pcutoff = pcutoff)
        else:
            adapter.before_adapt_inference(make_video = False)
            adapter.adaptation_training()
            adapter.after_adapt_inference(pcutoff = pcutoff)
