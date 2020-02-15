"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import wx
import os,sys,pydoc,platform
import webbrowser,subprocess
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.gui.extract_frames import Extract_frames
from deeplabcut.gui.label_frames import Label_frames
from deeplabcut.gui.create_training_dataset import Create_training_dataset
from deeplabcut.gui.train_network import Train_network
from deeplabcut.gui.analyze_videos import Analyze_videos
from deeplabcut.gui.evaluate_network import Evaluate_network
from deeplabcut.gui.extract_outlier_frames import Extract_outlier_frames
from deeplabcut.gui.refine_labels import Refine_labels
from deeplabcut.gui.create_videos import Create_Labeled_Videos

media_path = os.path.join(deeplabcut.__path__[0], 'gui' , 'media')
logo = os.path.join(media_path,'logo.png')
class Create_new_project(wx.Panel):
    def __init__(self, parent,gui_size):
        wx.Panel.__init__(self, parent)
        self.gui_size = gui_size
        self.parent = parent
        h=gui_size[0]
        w=gui_size[1]
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER,size=(h,w))
        # variable initilization
        self.filelist = []
        self.dir = None
        self.copy = False
        self.cfg = None
        self.loaded = False

        # design the panel
        self.sizer = wx.GridBagSizer(10, 15)

        text1 = wx.StaticText(self, label="DeepLabCut - Step 1. Create New Project")
        self.sizer.Add(text1, pos=(0, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM,border=15)

        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo))
        self.sizer.Add(icon, pos=(0,7), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,border=5)

        line = wx.StaticLine(self)
        self.sizer.Add(line, pos=(1, 0), span=(1, 8),flag=wx.EXPAND|wx.BOTTOM, border=10)

        # Add all the options
        self.proj = wx.RadioBox(self, label='Please choose an option:', choices=['Create new project', 'Load existing project'],majorDimension=0, style=wx.RA_SPECIFY_COLS)
        self.sizer.Add(self.proj, pos=(2, 0), span=(1, 5), flag=wx.LEFT, border=15)
        self.proj.Bind(wx.EVT_RADIOBOX,self.chooseOption)

        line = wx.StaticLine(self)
        self.sizer.Add(line, pos=(3, 0), span=(1, 8),flag=wx.EXPAND|wx.BOTTOM, border=10)

        self.proj_name = wx.StaticText(self, label="Name of the project:")
        self.sizer.Add(self.proj_name, pos=(4, 0), flag=wx.LEFT, border=15)

        self.proj_name_txt_box = wx.TextCtrl(self)
        self.sizer.Add(self.proj_name_txt_box, pos=(4, 1), span=(1, 2), flag=wx.TOP|wx.EXPAND)

        self.exp = wx.StaticText(self, label="Name of the experimenter:")
        self.sizer.Add(self.exp, pos=(5, 0), flag=wx.LEFT|wx.TOP, border=15)

        self.exp_txt_box = wx.TextCtrl(self)
        self.sizer.Add(self.exp_txt_box, pos=(5, 1), span=(1, 2), flag=wx.TOP|wx.EXPAND,border=5)

        self.vids = wx.StaticText(self, label="Choose the videos:")
        self.sizer.Add(self.vids, pos=(6, 0), flag=wx.TOP|wx.LEFT, border=10)

        self.sel_vids = wx.Button(self, label="Load Videos")
        self.sizer.Add(self.sel_vids, pos=(6, 1), flag=wx.TOP|wx.EXPAND, border=6)
        self.sel_vids.Bind(wx.EVT_BUTTON, self.select_videos)
#
        sb = wx.StaticBox(self, label="Optional Attributes")
        self.boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)

        self.change_workingdir = wx.CheckBox(self, label="Select the directory where project will be created")
        hbox2.Add(self.change_workingdir)
        hbox2.AddSpacer(20)
        self.change_workingdir.Bind(wx.EVT_CHECKBOX,self.activate_change_wd)
        self.sel_wd = wx.Button(self, label="Browse")
        self.sel_wd.Enable(False)
        self.sel_wd.Bind(wx.EVT_BUTTON, self.select_working_dir)
        hbox2.Add(self.sel_wd,0, wx.ALL, -1)
        self.boxsizer.Add(hbox2)

        self.copy_choice = wx.CheckBox(self, label="Do you want to copy the videos?")
        hbox3.Add(self.copy_choice)
        self.boxsizer.Add(hbox3)
        self.sizer.Add(self.boxsizer, pos=(7, 0), span=(1, 7),flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT , border=10)

        self.cfg_text = wx.StaticText(self, label="Select the config file")
        self.sizer.Add(self.cfg_text, pos=(8, 0), flag=wx.LEFT|wx.EXPAND, border=15)

        if sys.platform == 'darwin':
            self.sel_config = wx.FilePickerCtrl(self, path="", style=wx.FLP_USE_TEXTCTRL, message="Choose the config.yaml file", wildcard="*.yaml")
        else:
            self.sel_config = wx.FilePickerCtrl(self, path="", style=wx.FLP_USE_TEXTCTRL, message="Choose the config.yaml file", wildcard="config.yaml")
        self.sizer.Add(self.sel_config, pos=(8, 1), span=(1,3), flag=wx.TOP|wx.EXPAND, border=5)
        self.sel_config.Bind(wx.EVT_BUTTON, self.create_new_project)
        self.sel_config.SetPath("")
        # Hide the button as this is not the default option
        self.sel_config.Hide()
        self.cfg_text.Hide()

        self.help_button = wx.Button(self, label='Help')
        self.sizer.Add(self.help_button, pos=(9, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Ok")
        self.sizer.Add(self.ok, pos=(9, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.create_new_project)

        self.edit_config_file = wx.Button(self, label="Edit config file")
        self.sizer.Add(self.edit_config_file, pos=(9, 2))
        self.edit_config_file.Bind(wx.EVT_BUTTON, self.edit_config)
        self.edit_config_file.Enable(False)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(self.reset, pos=(9, 1),flag=wx.BOTTOM|wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_project)
        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def help_function(self,event):

        filepath= 'help.txt'
        f = open(filepath, 'w')
        sys.stdout = f
        fnc_name = 'deeplabcut.create_new_project'
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt","r+")
        help_text = help_file.read()
        wx.MessageBox(help_text,'Help',wx.OK | wx.ICON_INFORMATION)
        help_file.close()
        os.remove('help.txt')

    def chooseOption(self,event):
        if self.proj.GetStringSelection() == 'Load existing project':

            if self.loaded:
                self.sel_config.SetPath(self.cfg)
            self.proj_name.Enable(False)
            self.proj_name_txt_box.Enable(False)
            self.exp.Enable(False)
            self.exp_txt_box.Enable(False)
            self.sel_vids.Enable(False)
            self.change_workingdir.Enable(False)
            self.copy_choice.Enable(False)
            self.sel_config.Show()
            self.cfg_text.Show()
            #self.SetSizer(self.sizer)
            #self.sizer.Add(self.sizer, pos=(3, 0), span=(1, 8),flag=wx.EXPAND|wx.BOTTOM, border=15)
            self.sizer.Fit(self)
        else:
            self.proj_name.Enable(True)
            self.proj_name_txt_box.Enable(True)
            self.exp.Enable(True)
            self.exp_txt_box.Enable(True)
            self.sel_vids.Enable(True)
            self.change_workingdir.Enable(True)
            self.copy_choice.Enable(True)
            if self.sel_config.IsShown():
                self.sel_config.Hide()
                self.cfg_text.Hide()
#                self.ok.Enable(False)
#            else:
#                self.ok.Enable(True)

            self.SetSizer(self.sizer)
            self.sizer.Fit(self)

    def edit_config(self, event):
        """
        """
        if self.cfg!="":
            # For mac compatibility
            if platform.system() == 'Darwin':
                self.file_open_bool = subprocess.call(['open',self.cfg])
                self.file_open_bool = True
            else:
                self.file_open_bool = webbrowser.open(self.cfg)
            if self.file_open_bool:
                self.pose_cfg = auxiliaryfunctions.read_config(self.cfg)
            else:
                raise FileNotFoundError("File not found!")

    def select_videos(self,event):
        """
        Selects the videos from the directory
        """
        cwd = os.getcwd()
        dlg = wx.FileDialog(self, "Select videos to add to the project", cwd, "", "*.*", wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.vids = dlg.GetPaths()
            self.filelist = self.filelist + self.vids
            self.sel_vids.SetLabel("Total %s Videos selected" %len(self.filelist))

    def activate_change_wd(self,event):
        """
        Activates the option to change the working directory
        """
        self.change_wd = event.GetEventObject()
        if self.change_wd.GetValue() == True:
            self.sel_wd.Enable(True)
        else:
            self.sel_wd.Enable(False)

    def select_working_dir(self,event):
        cwd = os.getcwd()
        dlg = wx.DirDialog(self, "Choose the directory where your project will be saved:",cwd, style = wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.dir = dlg.GetPath()

    def create_new_project(self,event):
        """
        Finally create the new project
        """
        if self.sel_config.IsShown():
            self.cfg = self.sel_config.GetPath()
            if self.cfg == "":
                wx.MessageBox('Please choose the config.yaml file to load the project', 'Error', wx.OK | wx.ICON_ERROR)
                self.loaded = False
            else:
                wx.MessageBox('Project Loaded!', 'Info', wx.OK | wx.ICON_INFORMATION)
                self.loaded = True
                self.edit_config_file.Enable(True)
        else:
            self.task = self.proj_name_txt_box.GetValue()
            self.scorer = self.exp_txt_box.GetValue()
            if self.task and self.scorer and len(self.filelist):
                self.cfg = deeplabcut.create_new_project(self.task, self.scorer, self.filelist, self.dir, self.copy_choice.IsChecked())
            else:
                wx.MessageBox('Some of the entries are missing.\n\nMake sure that the task and experimenter name are specified and videos are selected!', 'Error', wx.OK | wx.ICON_ERROR)
                self.cfg = False
                self.loaded = False
            if self.cfg:
                wx.MessageBox('New Project Created', 'Info', wx.OK | wx.ICON_INFORMATION)
                self.loaded = True
                self.edit_config_file.Enable(True)

        # Remove the pages in case the user goes back to the create new project and creates/load a new project
        if self.parent.GetPageCount() > 3:
            for i in range(2, self.parent.GetPageCount()):
                self.parent.RemovePage(2)
                self.parent.Layout()

        # Add all the other pages
        if self.loaded:
            self.edit_config_file.Enable(True)
            if self.parent.GetPageCount() < 3:
                page3 = Extract_frames(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page3, "Extract frames")
                page4 = Label_frames(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page4, "Label frames")
                page5 = Create_training_dataset(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page5, "Create training dataset")
                page6 = Train_network(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page6, "Train network")
                page7 = Evaluate_network(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page7, "Evaluate network")
                page8 = Analyze_videos(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page8, "Analyze videos")
                page11 = Create_Labeled_Videos(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page11, "Create Videos")
                page9 = Extract_outlier_frames(self.parent,self.gui_size,self.cfg)
                self.parent.AddPage(page9, "Extract outlier frames")
                page10 = Refine_labels(self.parent,self.gui_size,self.cfg,page5)
                self.parent.AddPage(page10, "Refine labels")
                self.edit_config_file.Enable(True)

    def reset_project(self,event):
        self.loaded=False
        if self.sel_config.IsShown():
            self.sel_config.SetPath("")
            self.proj.SetSelection(0)
            self.sel_config.Hide()
            self.cfg_text.Hide()

        self.sel_config.SetPath("")
        self.proj_name_txt_box.SetValue("")
        self.exp_txt_box.SetValue("")
        self.filelist = []
        self.sel_vids.SetLabel("Load Videos")
        self.dir = os.getcwd()
        self.edit_config_file.Enable(False)
        self.proj_name.Enable(True)
        self.proj_name_txt_box.Enable(True)
        self.exp.Enable(True)
        self.exp_txt_box.Enable(True)
        self.sel_vids.Enable(True)
        self.change_workingdir.Enable(True)
        self.copy_choice.Enable(True)
        self.copy_choice.SetValue(False)
        try:
            self.change_wd.SetValue(False)
        except:
            pass
        self.sel_wd.Enable(False)
