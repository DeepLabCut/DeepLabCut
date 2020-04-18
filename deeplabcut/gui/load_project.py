"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import wx
import deeplabcut
import os
from deeplabcut.gui.create_new_project import Create_new_project
media_path = os.path.join(deeplabcut.__path__[0], 'gui' , 'media')
logo = os.path.join(media_path,'logo.png')
class Load_project(wx.Panel):
    """
    """

    def __init__(self, parent,gui_size,cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # variable initilization
        self.config = cfg
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut Load project")
        self.sizer.Add(text, pos=(0, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM,border=15)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo))
        self.sizer.Add(icon, pos=(0, 4), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,border=5)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=10)

        self.cfg = wx.StaticText(self, label="Select the config file")
        self.sizer.Add(self.cfg, pos=(2, 0), flag=wx.TOP|wx.LEFT, border=5)

        if sys.platform=='darwin':
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="*.yaml")
        else:
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        # self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        if self.config == None:
            self.config = "Please select the config file"
#            self.sel_config.SetPath("Please select the config file")
#        else:
        self.sel_config.SetPath(self.config)

        self.sizer.Add(self.sel_config, pos=(2, 1),span=(1,3),flag=wx.TOP|wx.EXPAND, border=5)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)

        button3 = wx.Button(self, label='Help')
        self.sizer.Add(button3, pos=(4, 0), flag=wx.LEFT, border=10)

        self.ok = wx.Button(self, label="Ok")
        self.sizer.Add(self.ok, pos=(4, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.load_project)

        self.cancel = wx.Button(self, label="Reset")
        self.sizer.Add(self.cancel, pos=(4, 3), span=(1, 1),flag=wx.BOTTOM|wx.RIGHT, border=10)
        self.cancel.Bind(wx.EVT_BUTTON, self.cancel_load_project)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def select_config(self,event):
        """
        """
        self.config = self.sel_config.GetPath()

    def load_project(self,event):
        print(self.config)

    def cancel_load_project(self,event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
