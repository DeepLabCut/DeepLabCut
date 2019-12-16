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
media_path = os.path.join(deeplabcut.__path__[0], 'gui' , 'media')
logo = os.path.join(media_path,'logo.png')

from deeplabcut.utils import auxiliaryfunctions
class Evaluate_network(wx.Panel):
    """
    """

    def __init__(self, parent,gui_size,cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # variable initilization
        self.config = cfg
        self.bodyparts = []
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Step 6. Evaluate Network")
        self.sizer.Add(text, pos=(0, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM,border=15)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo))
        self.sizer.Add(icon, pos=(0, 4), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,border=5)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 5),flag=wx.EXPAND|wx.BOTTOM, border=10)

        self.cfg_text = wx.StaticText(self, label="Select the config file")
        self.sizer.Add(self.cfg_text, pos=(2, 0), flag=wx.TOP|wx.LEFT, border=5)

        if sys.platform=='darwin':
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="*.yaml")
        else:
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        # self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")
        self.sizer.Add(self.sel_config, pos=(2, 1),span=(1,3),flag=wx.TOP|wx.EXPAND, border=5)
        self.sel_config.SetPath(self.config)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)


        sb = wx.StaticBox(self, label="Optional Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
#        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        shuffles_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffles_text_boxsizer = wx.StaticBoxSizer(shuffles_text, wx.VERTICAL)
        self.shuffles = wx.SpinCtrl(self, value='1',min=1,max=100)
        shuffles_text_boxsizer.Add(self.shuffles,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        trainingset = wx.StaticBox(self, label="Specify the trainingset index")
        trainingset_boxsizer = wx.StaticBoxSizer(trainingset, wx.VERTICAL)
        self.trainingset = wx.SpinCtrl(self, value='0',min=0,max=100)
        trainingset_boxsizer.Add(self.trainingset,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        self.plot_choice = wx.RadioBox(self, label='Want to plot predictions?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
        self.plot_choice.SetSelection(1)

        self.bodypart_choice = wx.RadioBox(self, label='Compare all bodyparts?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
        self.bodypart_choice.Bind(wx.EVT_RADIOBOX,self.chooseOption)

        config_file = auxiliaryfunctions.read_config(self.config)
        bodyparts = config_file['bodyparts']
        self.bodyparts_to_compare = wx.CheckListBox(self, choices=bodyparts, style=0,name = "Select the bodyparts")
        self.bodyparts_to_compare.Bind(wx.EVT_CHECKLISTBOX,self.getbp)
        self.bodyparts_to_compare.Hide()
#        self.SetSizer(self.sizer)
#        self.sizer.Fit(self)

        self.hbox1.Add(shuffles_text_boxsizer,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox1.Add(trainingset_boxsizer,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox1.Add(self.plot_choice,5, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox1.Add(self.bodypart_choice,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox1.Add(self.bodyparts_to_compare,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        boxsizer.Add(self.hbox1,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)
#        boxsizer.Add(self.hbox2,5, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        self.sizer.Add(boxsizer, pos=(3, 0), span=(1, 5),flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT , border=10)

        self.help_button = wx.Button(self, label='Help')
        self.sizer.Add(self.help_button, pos=(4, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Ok")
        self.sizer.Add(self.ok, pos=(4, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.evaluate_network)

        self.cancel = wx.Button(self, label="Reset")
        self.sizer.Add(self.cancel, pos=(4, 1), span=(1, 1),flag=wx.BOTTOM|wx.RIGHT, border=10)
        self.cancel.Bind(wx.EVT_BUTTON, self.cancel_evaluate_network)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def help_function(self,event):

        filepath= 'help.txt'
        f = open(filepath, 'w')
        sys.stdout = f
        fnc_name = 'deeplabcut.evaluate_network'
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt","r+")
        help_text = help_file.read()
        wx.MessageBox(help_text,'Help',wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove('help.txt')

    def chooseOption(self,event):
        if self.bodypart_choice.GetStringSelection() == 'No':
            self.bodyparts_to_compare.Show()
            self.getbp(event)
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
        if self.bodypart_choice.GetStringSelection() == 'Yes':
            self.bodyparts_to_compare.Hide()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
            self.bodyparts = 'all'

    def getbp(self,event):
        self.bodyparts = list(self.bodyparts_to_compare.GetCheckedStrings())


    def select_config(self,event):
        """
        """
        self.config = self.sel_config.GetPath()

    def evaluate_network(self,event):

        #shuffle = self.shuffle.GetValue()
        trainingsetindex = self.trainingset.GetValue()

        shuffle = [self.shuffles.GetValue()]
        if self.plot_choice.GetStringSelection() == "Yes":
            plotting = True
        else:
            plotting = False

        if len(self.bodyparts)==0:
            self.bodyparts='all'
        deeplabcut.evaluate_network(self.config,Shuffles=shuffle,trainingsetindex=trainingsetindex,plotting=plotting,show_errors=True,comparisonbodyparts=self.bodyparts,gputouse=None)

    def cancel_evaluate_network(self,event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.plot_choice.SetSelection(1)
        self.bodypart_choice.SetSelection(0)
        self.shuffles.SetValue(1)
        if self.bodyparts_to_compare.IsShown():
            self.bodyparts_to_compare.Hide()
