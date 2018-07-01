"""
Author: Hao Wu
hwu01@g.harvard.edu

This is the helper class for video reading and saving in DeepLabCut

"""
import numpy as np
import skvideo.io
import skvideo

class VideoProcessor(object):
    '''
    Base class for a video processing unit, 
    implementation is required for video loading and saving
    '''
    def __init__(self,fname='',sname='', nframes = -1, fps = 30):
        self.fname = fname
        self.sname = sname

        self.nframes = nframes
        
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


class VideoProcessorSK(VideoProcessor):
    '''
    Video Processor using skvideo.io
    requires sk-video in python,
    and ffmpeg installed in the operating system
    '''
    def __init__(self, *args, **kwargs):
        super(VideoProcessorSK, self).__init__(*args, **kwargs)
    
    def get_video(self):
         return skvideo.io.FFmpegReader(self.fname)
        
    def get_info(self):
        infos = skvideo.io.ffprobe(self.fname)['video']
        self.h = int(infos['@height'])
        self.w = int(infos['@width'])
        self.FPS = eval(infos['@avg_frame_rate'])
        vshape = self.vid.getShape()
        all_frames = vshape[0]
        self.nc = vshape[3]

        if self.nframes == -1 or self.nframes>all_frames:
            self.nframes = all_frames
            
    def create_video(self):
        return skvideo.io.FFmpegWriter(self.sname, outputdict={'-r':str(self.FPS)})

    def _read_frame(self):
        return self.vid._readFrame()
    
    def save_frame(self,frame):
        self.svid.writeFrame(frame)
    
    def close(self):
        self.svid.close()
        self.vid.close()