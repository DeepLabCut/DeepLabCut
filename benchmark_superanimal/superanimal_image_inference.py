import deeplabcut
from deeplabcut.pose_estimation_pytorch.apis.analyze_images import superanimal_analyze_images
import glob

superanimal_name = 'superanimal_quadruped'
model_name = 'hrnetw32'
device = 'cuda'
max_individuals = 3

ret = superanimal_analyze_images(superanimal_name,
                                 model_name,
                                 'test_rodent_images',
                                 max_individuals,
                                 'vis_test_rodent_images')
                                 
