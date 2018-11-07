'''
Adapted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''
import logging, os


def setup_logging():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join('log.txt'), filemode='w',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO, format=FORMAT)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
