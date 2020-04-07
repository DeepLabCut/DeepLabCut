"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import wx
import os,sys,pydoc
import deeplabcut
from deeplabcut.refine_training_dataset import tracklets


media_path = os.path.join(deeplabcut.__path__[0], 'gui' , 'media')
logo = os.path.join(media_path,'logo.png')


class Refine_tracklets(wx.Panel):
    """
    """

    def __init__(self, parent,gui_size,cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.config = cfg
        self.datafile = ''
        self.video = ''
        # design the panel
        sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Tracklets: Extract and Refine")
        sizer.Add(text, pos=(0, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM,border=15)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo))
        sizer.Add(icon, pos=(0, 4), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,border=5)

        line1 = wx.StaticLine(self)
        sizer.Add(line1, pos=(1, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=10)


        self.cfg_text = wx.StaticText(self, label="Select the config file")
        sizer.Add(self.cfg_text, pos=(2, 0), flag=wx.TOP|wx.LEFT, border=5)

        if sys.platform=='darwin':
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="*.yaml")
        else:
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        # self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        sizer.Add(self.sel_config, pos=(2, 1),span=(1,3),flag=wx.TOP|wx.EXPAND, border=5)
        self.sel_config.SetPath(self.config)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)

        self.video_text = wx.StaticText(self, label="Select the video")
        sizer.Add(self.video_text, pos=(3, 0), flag=wx.TOP | wx.LEFT, border=5)
        self.sel_video = wx.FilePickerCtrl(self, path="", style=wx.FLP_USE_TEXTCTRL, message="Open video")
        sizer.Add(self.sel_video, pos=(3, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_video.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_video)

        self.data_text = wx.StaticText(self, label="Select the tracklet data")
        sizer.Add(self.data_text, pos=(4, 0), flag=wx.TOP | wx.LEFT, border=5)
        self.sel_datafile = wx.FilePickerCtrl(self, path="", style=wx.FLP_USE_TEXTCTRL,
                                              message="Open tracklet data", wildcard="Pickle files (*.pickle)|*.pickle")
        sizer.Add(self.sel_datafile, pos=(4, 1), span=(1, 3), flag=wx.TOP | wx.EXPAND, border=5)
        self.sel_datafile.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_datafile)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        slider_swap_text = wx.StaticBox(self, label="Specify the min swap fraction")
        slider_swap_sizer = wx.StaticBoxSizer(slider_swap_text, wx.VERTICAL)
        self.slider_swap = wx.SpinCtrl(self, value='1')
        slider_swap_sizer.Add(self.slider_swap, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox.Add(slider_swap_sizer, 10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        slider_track_text = wx.StaticBox(self, label="Specify the min relative tracklet length")
        slider_track_sizer = wx.StaticBoxSizer(slider_track_text, wx.VERTICAL)
        self.slider_track = wx.SpinCtrl(self, value='1')
        slider_track_sizer.Add(self.slider_track, 20, wx.EXPAND | wx.TOP | wx.BOTTOM, 10)
        hbox.Add(slider_track_sizer, 10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        sizer.Add(hbox, pos=(5, 0), flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT , border=10)

        self.help_button = wx.Button(self, label='Help')
        sizer.Add(self.help_button, pos=(6, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Ok")
        sizer.Add(self.ok, pos=(6, 3))
        self.ok.Bind(wx.EVT_BUTTON, self.refine_tracklets)

        self.save = wx.Button(self, label='Save')
        sizer.Add(self.save, pos=(6, 4))
        self.save.Bind(wx.EVT_BUTTON, self.save_tracklets)
        self.save.Enable(False)

        self.reset = wx.Button(self, label="Reset")
        sizer.Add(self.reset, pos=(6, 1), flag=wx.BOTTOM|wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_refine_tracklets)

        sizer.AddGrowableCol(2)

        self.SetSizer(sizer)
        sizer.Fit(self)

    def help_function(self,event):

        filepath= 'help.txt'
        f = open(filepath, 'w')
        sys.stdout = f
        fnc_name = 'deeplabcut.refine_training_dataset.tracklets.TrackletManager'
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt","r+")
        help_text = help_file.read()
        wx.MessageBox(help_text,'Help',wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove('help.txt')

    def select_config(self,event):
        """
        """
        self.config = self.sel_config.GetPath()

    def select_datafile(self, event):
        self.datafile = self.sel_datafile.GetPath()

    def select_video(self, event):
        self.video = self.sel_video.GetPath()

    def refine_tracklets(self,event):
        self.manager = tracklets.TrackletManager(self.config, self.slider_swap.GetValue() / 100,
                                                 self.slider_track.GetValue() / 100)
        self.manager.load_tracklets_from_pickle(self.datafile)
        self.manager.find_swapping_bodypart_pairs()
        self.viz = tracklets.TrackletVisualizer(self.manager, self.video, 50)
        self.viz.show()
        self.save.Enable(True)

    def save_tracklets(self, event):
        self.manager.save()

    def reset_refine_tracklets(self,event):
        """
        Reset to default
        """
        self.config = ''
        self.datafile = ''
        self.video = ''
        self.sel_config.SetPath("")
        self.sel_datafile.SetPath("")
        self.sel_video.SetPath("")
        self.slider_swap.SetValue(1)
        self.slider_track.SetValue(1)
        self.save.Enable(False)
