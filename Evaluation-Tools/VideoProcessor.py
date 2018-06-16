"""
Author:
Hao Wu

This is the helper class for 
"""

import cv2
import numpy as np

class VideoProcessor(object):
    '''
    Base class for a video processing unit
    '''
    def __init__(self,fname='',sname='',cropping=False,x1 = 0, y1 = 0, x2 = 0, y2 = 0, nframes = -1, fps = 30, verbose = False):
        self.fname = fname
        self.sname = sname
        self.cropping = cropping
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.nframes = nframes
        self.verbose = verbose
        
        self.h = 0 
        self.w = 0
        self.sh = 0
        self.sw = 0
        self.FPS = fps
        self.nc = 3
        self.i = 0
        
        try:
            if self.fname != '':
                self.vid = self.get_video()
                self.get_info()
            
            if self.sname != '':
                if cropping:
                    self.check_crop()
                else:
                    self.sh = self.h
                    self.sw = self.w
                    
                self.svid = self.create_video()

        except Exception as ex:
            print('Error: %s', ex)
            
    def check_crop(self):
        if self.x1>=0 and self.x2<=self.w and self.y1>=0 and self.y2<=h:
            sw = self.x2 - self.x1
            sh = self.y2 - self.y1
            if sw>0 and sh>0:
                self.sw = sw
                self.sh = sh
            else:
                raise Exception('Please check the order of cropping parameter')
        else:
            raise Exception('Please check the boundary of cropping')
            
    def crop(self,frame):
        return frame[self.x1:self.x2,self.y1:self.y2]

    def load_frame(self):
        try:
            frame = self._read_frame()
            self.i += 1
            return frame
        except Exception as ex:
            print('Error: %s', ex)
    
    def height(self):
        return self.h
    
    def width(self):
        return self.w
    
    def fps(self):
        return self.FPS
    
    def counter(self):
        return self.i
    
    def frame_count(self):
        return self.nframes
        
                       
    def get_video(self):
        '''
        implement your own
        '''
        pass
    
    def get_info(self):
        '''
        implement your own
        '''
        pass

    def create_video(self):
        '''
        implement your own
        '''
        pass
    

        
    def _read_frame(self):
        '''
        implement your own
        '''
        pass
    
    def save_frame(self,frame):
        '''
        implement your own
        '''
        pass
    
    def close(self):
        '''
        implement your own
        '''
        pass


    
class VideoProcessorCV(VideoProcessor):
    
    def __init__(self, *args, **kwargs):
        super(VideoProcessorCV, self).__init__(*args, **kwargs)
    
    def get_video(self):
         return cv2.VideoCapture(self.fname)
        
    def get_info(self):
        self.w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3
        if self.nframes == -1 or self.nframes>all_frames:
            self.nframes = all_frames
            
    def create_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        return cv2.VideoWriter(self.sname,fourcc, self.fps, (self.h,self.w),True)
    
    def _read_frame(self):
        return np.flip(self.vid.read()[1],2)
    
    def save_frame(self,frame):
        self.svid.write(np.flip(frame,2))
    
    def close(self):
        self.svid.release()
        self.vid.release()