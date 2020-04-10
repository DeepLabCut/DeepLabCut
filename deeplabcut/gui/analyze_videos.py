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
from deeplabcut.utils import auxiliaryfunctions
media_path = os.path.join(deeplabcut.__path__[0], 'gui' , 'media')
logo = os.path.join(media_path,'logo.png')

class Analyze_videos(wx.Panel):
    """
    """

    def __init__(self, parent,gui_size,cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        # variable initilization
        self.filelist = []
        self.picklelist =[]
        self.bodyparts = []
        self.config = cfg
        self.cfg = auxiliaryfunctions.read_config(self.config)
        self.draw = False
        # design the panel
        self.sizer = wx.GridBagSizer(5, 10)

        if self.cfg.get('multianimalproject', False):
            text = wx.StaticText(self, label="DeepLabCut - Step 7. Analyze Videos and Detect Tracklets")
        else:
            text = wx.StaticText(self, label="DeepLabCut - Step 7. Analyze Videos ....")

        self.sizer.Add(text, pos=(0, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM,border=15)
        # Add logo of DLC
        icon = wx.StaticBitmap(self, bitmap=wx.Bitmap(logo))
        self.sizer.Add(icon, pos=(0, 9), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,border=5)

        line1 = wx.StaticLine(self)
        self.sizer.Add(line1, pos=(1, 0), span=(1, 10),flag=wx.EXPAND|wx.BOTTOM, border=10)

        self.cfg_text = wx.StaticText(self, label="Select the config file")
        self.sizer.Add(self.cfg_text, pos=(2, 0), flag=wx.TOP|wx.LEFT, border=10)

        if sys.platform=='darwin':
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="*.yaml")
        else:
            self.sel_config = wx.FilePickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the config.yaml file", wildcard="config.yaml")

        self.sizer.Add(self.sel_config, pos=(2, 1),span=(1,3),flag=wx.TOP|wx.EXPAND, border=5)
        self.sel_config.SetPath(self.config)
        self.sel_config.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_config)

        self.vids = wx.StaticText(self, label="Choose the videos")
        self.sizer.Add(self.vids, pos=(3, 0), flag=wx.TOP|wx.LEFT, border=10)

        self.sel_vids = wx.Button(self, label="Select videos to analyze")
        self.sizer.Add(self.sel_vids, pos=(3, 1), flag=wx.TOP|wx.EXPAND, border=5)
        self.sel_vids.Bind(wx.EVT_BUTTON, self.select_videos)


        sb = wx.StaticBox(self, label="Additional Attributes")
        boxsizer = wx.StaticBoxSizer(sb, wx.VERTICAL)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)

        videotype_text = wx.StaticBox(self, label="Specify the videotype")
        videotype_text_boxsizer = wx.StaticBoxSizer(videotype_text, wx.VERTICAL)

        videotypes = ['.avi', '.mp4', '.mov']
        self.videotype = wx.ComboBox(self,choices = videotypes,style = wx.CB_READONLY)
        self.videotype.SetValue('.avi')
        videotype_text_boxsizer.Add(self.videotype,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        shuffle_text = wx.StaticBox(self, label="Specify the shuffle")
        shuffle_boxsizer = wx.StaticBoxSizer(shuffle_text, wx.VERTICAL)
        self.shuffle = wx.SpinCtrl(self, value='1',min=0,max=100)
        shuffle_boxsizer.Add(self.shuffle,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        trainingset = wx.StaticBox(self, label="Specify the trainingset index")
        trainingset_boxsizer = wx.StaticBoxSizer(trainingset, wx.VERTICAL)
        self.trainingset = wx.SpinCtrl(self, value='0',min=0,max=100)
        trainingset_boxsizer.Add(self.trainingset,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        destfolder_text = wx.StaticBox(self, label="Specify destination folder")
        destfolderboxsizer = wx.StaticBoxSizer(destfolder_text, wx.VERTICAL)
        self.sel_destfolder = wx.DirPickerCtrl(self, path="",style=wx.FLP_USE_TEXTCTRL,message="Choose the destination folder")
        self.sel_destfolder.SetPath("None")
        self.destfolder = None
        self.sel_destfolder.Bind(wx.EVT_FILEPICKER_CHANGED, self.select_destfolder)
        destfolderboxsizer.Add(self.sel_destfolder,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        hbox1.Add(videotype_text_boxsizer,5, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        hbox1.Add(shuffle_boxsizer,5, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        hbox1.Add(trainingset_boxsizer,5, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        hbox2.Add(destfolderboxsizer,5, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        boxsizer.Add(hbox1,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        #dealing with maDLC:

        if self.cfg.get('multianimalproject', False):

            self.create_video_with_all_detections = wx.RadioBox(self, label='Create video for checking detections', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
            self.create_video_with_all_detections.SetSelection(1)
            hbox2.Add(self.create_video_with_all_detections, 5, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
            boxsizer.Add(hbox2,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

            tracker_text = wx.StaticBox(self, label="Specify the Tracker Method!")
            tracker_text_boxsizer = wx.StaticBoxSizer(tracker_text, wx.VERTICAL)

            trackertypes = ['skeleton', 'box', 'clowncats']
            self.trackertypes = wx.ComboBox(self,choices = trackertypes,style = wx.CB_READONLY)
            self.trackertypes.SetValue('box')
            tracker_text_boxsizer.Add(self.trackertypes,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)
            hbox2.Add(tracker_text_boxsizer,5, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        else:
            self.csv = wx.RadioBox(self, label='Want to save result(s) as csv?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
            self.csv.SetSelection(1)

            self.dynamic = wx.RadioBox(self, label='Want to dynamically crop bodyparts?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
            self.dynamic.SetSelection(1)

            self.filter = wx.RadioBox(self, label='Want to filter the predictions?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
            self.filter.SetSelection(1)

            self.trajectory = wx.RadioBox(self, label='Want to plot the trajectories?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
            self.trajectory.Bind(wx.EVT_RADIOBOX,self.chooseOption)
            self.trajectory.SetSelection(1)

            hbox2.Add(self.csv,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
            hbox2.Add(self.filter,10,wx.EXPAND|wx.TOP|wx.BOTTOM,5)
            boxsizer.Add(hbox2,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

            hbox3.Add(self.dynamic,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
            hbox3.Add(self.trajectory,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        config_file = auxiliaryfunctions.read_config(self.config)
        if config_file.get('multianimalproject', False):
            bodyparts = config_file['multianimalbodyparts']
        else:
            bodyparts = config_file['bodyparts']
        self.trajectory_to_plot = wx.CheckListBox(self, choices=bodyparts, style=0,name = "Select the bodyparts")
        self.trajectory_to_plot.Bind(wx.EVT_CHECKLISTBOX,self.getbp)
        self.trajectory_to_plot.SetCheckedItems(range(len(bodyparts)))
        self.trajectory_to_plot.Hide()



        self.draw_skeleton = wx.RadioBox(self, label='Include the skeleton in the video?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
        self.draw_skeleton.Bind(wx.EVT_RADIOBOX, self.choose_draw_skeleton_options)
        self.draw_skeleton.SetSelection(1)
        self.draw_skeleton.Hide()

        self.trail_points_text = wx.StaticBox(self, label="Specify the number of trail points")
        trail_pointsboxsizer = wx.StaticBoxSizer(self.trail_points_text, wx.VERTICAL)
        self.trail_points = wx.SpinCtrl(self, value='1')
        trail_pointsboxsizer.Add(self.trail_points,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)
        self.trail_points_text.Hide()
        self.trail_points.Hide()

        hbox3.Add(self.trajectory_to_plot,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        boxsizer.Add(hbox3,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        hbox4.Add(self.draw_skeleton,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        hbox4.Add(trail_pointsboxsizer,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        boxsizer.Add(hbox4,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)
        self.sizer.Add(boxsizer, pos=(5, 0), span=(1, 10),flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT , border=10)



        self.help_button = wx.Button(self, label='Help')
        self.sizer.Add(self.help_button, pos=(7, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Analyze Videos")
        self.sizer.Add(self.ok, pos=(7, 7), flag=wx.BOTTOM|wx.RIGHT, border=10)
        self.ok.Bind(wx.EVT_BUTTON, self.analyze_videos)

        if config_file.get('multianimalproject', False):
            self.ok = wx.Button(self, label="Step 2: Convert to Tracklets")
            self.sizer.Add(self.ok, pos=(7, 8), flag=wx.BOTTOM|wx.RIGHT, border=10)
            self.ok.Bind(wx.EVT_BUTTON, self.convert2_tracklets)


            #self.ok = wx.Button(self, label="Step 2: Qual check Tracklets")
            #self.sizer.Add(self.ok, pos=(8, 8), flag=wx.BOTTOM|wx.RIGHT, border=10)
            #self.ok.Bind(wx.EVT_BUTTON, self.video_tracklets)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(self.reset, pos=(7, 1), span=(1, 1),flag=wx.BOTTOM|wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_analyze_videos)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def help_function(self,event):

        filepath= 'help.txt'
        f = open(filepath, 'w')
        sys.stdout = f
        fnc_name = 'deeplabcut.analyze_videos'
        pydoc.help(fnc_name)
        f.close()
        sys.stdout = sys.__stdout__
        help_file = open("help.txt","r+")
        help_text = help_file.read()
        wx.MessageBox(help_text,'Help',wx.OK | wx.ICON_INFORMATION)
        os.remove('help.txt')

    def select_destfolder(self,event):
        self.destfolder = self.sel_destfolder.GetPath()

    def select_config(self,event):
        """
        """
        self.config = self.sel_config.GetPath()

    def convert2_tracklets(self,event):
        shuffle = self.shuffle.GetValue()
        trainingsetindex = self.trainingset.GetValue()
        deeplabcut.convert_detections2tracklets(self.config, self.filelist, videotype=self.videotype.GetValue(),
                                                    shuffle=shuffle, trainingsetindex=trainingsetindex, track_method=self.trackertypes.GetValue())

    #def video_tracklets(self,event):
    #    shuffle = self.shuffle.GetValue()
    #    trainingsetindex = self.trainingset.GetValue()
    #    deeplabcut.create_video_from_pickled_tracks(self.filelist, picklefile, pcutoff=0.6)

    def select_videos(self,event):
        """
        Selects the videos from the directory
        """
        cwd = os.getcwd()
        dlg = wx.FileDialog(self, "Select videos to analyze", cwd, "", "*.*", wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.vids = dlg.GetPaths()
            self.filelist = self.filelist + self.vids
            self.sel_vids.SetLabel("Total %s Videos selected" %len(self.filelist))

    def choose_draw_skeleton_options(self,event):
        if self.draw_skeleton.GetStringSelection() == "Yes":
            self.draw = True
        else:
            self.draw = False

    def analyze_videos(self,event):

        shuffle = self.shuffle.GetValue()
        trainingsetindex = self.trainingset.GetValue()

        if self.cfg.get('multianimalproject', False):
            print("what is going on")
        else:
            if self.csv.GetStringSelection() == "Yes":
                save_as_csv = True
            else:
                save_as_csv = False
            if self.dynamic.GetStringSelection() == "No":
                dynamic = (False, .5, 10)
            else:
                dynamic = (True, .5, 10)
            if self.filter.GetStringSelection() == "No":
                filter = None
            else:
                filter = True


        if self.cfg['cropping']=='True':
            crop = self.cfg['x1'], self.cfg['x2'], self.cfg['y1'], self.cfg['y2']
        else:
            crop = None

        if self.cfg.get('multianimalproject', False):
            scorername = deeplabcut.analyze_videos(self.config, self.filelist, videotype=self.videotype.GetValue(), shuffle=shuffle,
                                                     trainingsetindex=trainingsetindex, gputouse=None, destfolder=self.destfolder, cropping=crop)
            if self.filter.GetStringSelection() == "Yes":
                deeplabcut.filterpredictions(self.config, self.filelist, videotype=self.videotype.GetValue(), shuffle=shuffle,
                                              trainingsetindex=trainingsetindex, filtertype='median', windowlength=5, save_as_csv=True,
                                              destfolder=self.destfolder)

        else:
            scorername = deeplabcut.analyze_videos(self.config, self.filelist, videotype=self.videotype.GetValue(), shuffle=shuffle,
                                                     trainingsetindex=trainingsetindex, gputouse=None, save_as_csv=save_as_csv,
                                                      destfolder=self.destfolder, cropping=crop, dynamic=dynamic)
            if self.filter.GetStringSelection() == "Yes":
                deeplabcut.filterpredictions(self.config, self.filelist, videotype=self.videotype.GetValue(), shuffle=shuffle,
                                                trainingsetindex=trainingsetindex, filtertype='median', windowlength=5,
                                                save_as_csv=True, destfolder=self.destfolder)

        if self.cfg.get('multianimalproject', False):
            if self.create_video_with_all_detections.GetStringSelection() == "Yes":
                trainFrac = self.cfg['TrainingFraction'][trainingsetindex]
                scorername, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(self.cfg,shuffle,trainFraction=trainFrac)
                print(scorername)
                deeplabcut.create_video_with_all_detections(self.config, self.filelist, DLCscorername=scorername)

        if self.trajectory.GetStringSelection() == "Yes":
            deeplabcut.plot_trajectories(self.config, self.filelist, displayedbodyparts=self.bodyparts,
                                           videotype=self.videotype.GetValue(), shuffle=shuffle, trainingsetindex=trainingsetindex, filtered=True, showfigures=False, destfolder=self.destfolder)


    def reset_analyze_videos(self,event):
        """
        Reset to default
        """
        if self.cfg.get('multianimalproject', True):
            self.create_video_with_all_detections.SetSelection(1)
        else:
            self.csv.SetSelection(1)
            self.filter.SetSelection(1)
            self.trajectory.SetSelection(1)
            self.dynamic.SetSelection(1)
        self.config = []
        self.sel_config.SetPath("")
        self.videotype.SetStringSelection(".avi")
        self.sel_vids.SetLabel("Select videos to analyze")
        self.filelist = []
        self.shuffle.SetValue(1)
        self.trainingset.SetValue(0)
        self.sel_destfolder.SetPath("None")
        if self.draw_skeleton.IsShown():
            self.draw_skeleton.SetSelection(1)
            self.draw_skeleton.Hide()
            self.trail_points_text.Hide()
            self.trail_points.Hide()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)

    def chooseOption(self,event):
        if self.trajectory.GetStringSelection() == 'Yes':
            self.trajectory_to_plot.Show()
            self.getbp(event)
        if self.trajectory.GetStringSelection() == 'No':
            self.trajectory_to_plot.Hide()
            self.bodyparts = []
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)

    def getbp(self,event):
        self.bodyparts = list(self.trajectory_to_plot.GetCheckedStrings())
