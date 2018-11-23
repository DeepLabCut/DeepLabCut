"""
Author: Hao Wu
hwu01@g.harvard.edu


This is the helper class for video reading and saving in DeepLabCut.

You can set various codecs below,
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
i.e. 'XVID'
"""

import numpy as np
import cv2

class VideoProcessor(object):
    '''
    Base class for a video processing unit, 
    implementation is required for video loading and saving
    '''
    def __init__(self,fname='',sname='', nframes = -1, fps = 30,codec='X264'):
        self.fname = fname
        self.sname = sname
        self.nframes = nframes
        self.codec=codec
        
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
                self.sh = self.h
                self.sw = self.w
                self.svid = self.create_video()

        except Exception as ex:
            print('Error: %s', ex)
            
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
    '''
    OpenCV implementation of VideoProcessor
    requires opencv-python==3.4.0.12
    '''
    def __init__(self, *args, **kwargs):
        super(VideoProcessorCV, self).__init__(*args, **kwargs)
    
    def get_video(self):
         return cv2.VideoCapture(self.fname)
        
    def get_info(self):
        self.w = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        all_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = self.vid.get(cv2.CAP_PROP_FPS)
        self.nc = 3
        if self.nframes == -1 or self.nframes>all_frames:
            self.nframes = all_frames
        print(self.nframes)
            
    def create_video(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        return cv2.VideoWriter(self.sname,fourcc, self.FPS, (self.w,self.h),True)
    
    def _read_frame(self): #return RGB (rather than BGR)!
        #return cv2.cvtColor(np.flip(self.vid.read()[1],2), cv2.COLOR_BGR2RGB)
        return np.flip(self.vid.read()[1],2)
    
    def save_frame(self,frame):
        self.svid.write(np.flip(frame,2))
    
    def close(self):
        self.svid.release()
        self.vid.release()
