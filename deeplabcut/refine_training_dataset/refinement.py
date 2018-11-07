"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
T Nath, nath@rowland.harvard.edu
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu
"""

import sys
import wx
import os
import pandas as pd
import numpy as np
#from skimage import io
import PIL
import glob

#import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os.path
import argparse
import yaml
from pathlib import Path
from deeplabcut.refine_training_dataset import auxfun_drag
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################

class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, parent,config):
        wx.Frame.__init__(self, parent,title="DeepLabCut2.0 - Labels Refining ToolBox", size=(1200, 980))

# Add SplitterWindow panels top for figure and bottom for buttons
        self.split_win = wx.SplitterWindow(self)
        # self.top_split = wx.Panel(self.split_win, style=wx.SUNKEN_BORDER)
        self.top_split = MatplotPanel(self.split_win,config) # This call/link the MatplotPanel and MainFrame classes which replaces the above line
        self.bottom_split = wx.Panel(self.split_win, style=wx.SUNKEN_BORDER)
        self.split_win.SplitHorizontally(self.top_split, self.bottom_split, 880)

        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")
# Add Buttons to the bottom_split window and bind them to plot functions
        self.Button1 = wx.Button(self.bottom_split, -1, "Load Labels", size=(195, 40), pos=(80, 25))
        self.Button1.Bind(wx.EVT_BUTTON, self.browseDir)

        self.Button5 = wx.Button(self.bottom_split, -1, "Help", size=(60, 40), pos=(310, 25))
        self.Button5.Bind(wx.EVT_BUTTON, self.help)
        self.Button5.Enable(True)

        self.Button3 = wx.Button(self.bottom_split, -1, "Previous Image", size=(150, 40), pos=(420, 25))
        self.Button3.Bind(wx.EVT_BUTTON, self.prevImage)
        self.Button3.Enable(False)

        self.Button2 = wx.Button(self.bottom_split, -1, "Next Image", size=(130, 40), pos=(640, 25))
        self.Button2.Bind(wx.EVT_BUTTON, self.nextImage)
        self.Button2.Enable(False)

        self.Button4 = wx.Button(self.bottom_split, -1, "Save", size=(100, 40), pos=(840, 25))
        self.Button4.Bind(wx.EVT_BUTTON, self.save)
        self.Button4.Enable(False)

        self.close = wx.Button(self.bottom_split, -1, "Quit", size=(100, 40), pos=(990, 25))
        self.close.Bind(wx.EVT_BUTTON,self.quitButton)
        self.close.Enable(False)

        self.adjustLabelCheck = wx.CheckBox(self.top_split, label = 'Adjust original labels?',pos = (80, 855))
        self.adjustLabelCheck.Bind(wx.EVT_CHECKBOX,self.adjustLabel)
        
        self.Button5 = wx.Button(self.top_split,-1,"Zoom", size=(60,20),pos=(840,855))
        self.Button5.Bind(wx.EVT_BUTTON,self.zoom)
        
        self.Button6 = wx.Button(self.top_split,-1,"Pan", size=(60,20),pos=(940,855))
        self.Button6.Bind(wx.EVT_BUTTON,self.pan)
        
        self.Button7 = wx.Button(self.top_split,-1,"Home", size=(60,20),pos=(1040,855))
        self.Button7.Bind(wx.EVT_BUTTON,self.home)
         
        self.Bind(wx.EVT_CLOSE,self.closewindow)

        self.currentDirectory = os.getcwd()
        self.index = []
        self.iter = []
        self.threshold = []
        self.file = 0
        with open(str(config), 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        self.humanscorer = cfg['scorer']
        self.move2corner = cfg['move2corner']
        self.center = cfg['corner2move2']
        self.colormap = plt.get_cmap(cfg['colormap'])
        self.markerSize = cfg['dotsize']
        self.adjust_original_labels = False
        self.alpha = cfg['alphavalue']
        self.iterationindex = cfg['iteration']
        self.project_path=cfg['project_path']
        self.droppedframes=[] #will collect files that were removed
        
# ###########################################################################
# functions for button responses
# ###########################################################################
    def closewindow(self,event):
        self.Destroy()

    def adjustLabel(self, event):

      self.chk = event.GetEventObject()
      if self.chk.GetValue() == True:
          self.adjust_original_labels = True
      else:
          self.adjust_original_labels = False
        
    def zoom(self,event):
        self.statusbar.SetStatusText("Zoom")
        self.toolbar.zoom()
        
    def home(self,event):
        self.statusbar.SetStatusText("Home")
        self.toolbar.home()
         
    def pan(self,event):
        self.statusbar.SetStatusText("Pan")
        self.toolbar.pan()
    
    def OnSliderScroll(self, event):
        """
        Adjust marker size for plotting the annotations
        """
        self.drs = []
        self.updatedCoords = []
        plt.close(self.fig1)
        self.fig1, (self.ax1f1) = plt.subplots(figsize=(12, 7.8),facecolor = "None")
        self.markerSize = (self.slider.GetValue())
        imagename1 = os.path.join(self.project_path,self.index[self.iter])
        im = PIL.Image.open(imagename1)
        im_axis = self.ax1f1.imshow(im,self.colormap)
        if self.adjust_original_labels == False:
            self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem) + " "+ " Threshold chosen is: " + str("{0:.2f}".format(self.threshold))))
        else:
            self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem)))
        self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
        MainFrame.plot(self,im,im_axis)
        self.toolbar = NavigationToolbar(self.canvas)
        MainFrame.confirm(self)


    def browseDir(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """

        self.adjustLabelCheck.Enable(False)

        if self.adjust_original_labels == True:
            dlg = wx.FileDialog(self, "Choose the labeled dataset file(CollectedData_*.h5 file)", "", "", "All CollectedData_*.h5 files(CollectedData_*.h5)|CollectedData_*.h5", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        else:
            fname = str('machinelabels-iter'+str(self.iterationindex)+'.h5')
            dlg = wx.FileDialog(self, "Choose the machinelabels file for current iteration.",
                                "", "", wildcard=fname,
                                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        if dlg.ShowModal() == wx.ID_OK:
            self.data_file = dlg.GetPath()
            self.dir = str(Path(self.data_file).parents[0])
            print(self.dir)
            self.fileName = str(Path(self.data_file).stem)
            self.Button1.Enable(False)
            self.Button2.Enable(True)
            self.Button4.Enable(True)
            self.Button5.Enable(True)
            self.close.Enable(True)
        else:
            dlg.Destroy()
            self.Close(True)
        dlg.Destroy()

        self.fig1, (self.ax1f1) = plt.subplots(figsize=(12, 7.8),facecolor = "None")
        try:
            self.dataname = str(self.data_file)
        except:
            print("No machinelabels file found!")
            self.Destroy()
        self.iter = 0

        if os.path.isfile(self.dataname):
            self.Dataframe = pd.read_hdf(self.dataname,'df_with_missing')
            self.scorer = self.Dataframe.columns.get_level_values(0)[0]
            bodyParts = self.Dataframe.columns.get_level_values(1)
            _, idx = np.unique(bodyParts, return_index=True)
            self.bodyparts2plot =  bodyParts[np.sort(idx)]
            self.num_joints = len(self.bodyparts2plot)
            self.index = list(self.Dataframe.iloc[:,0].index)
            self.drs = []
            self.updatedCoords = []

            imdropped=True
            while imdropped==True:
                # Reading images
                try:
                    imagename1 = os.path.join(self.project_path,self.index[self.iter])
                    im = PIL.Image.open(imagename1)
                    # Plotting
                    im_axis = self.ax1f1.imshow(im,self.colormap)
                    imdropped=False
                except FileNotFoundError: #based on this flag, the image will be removed. 
                    imdropped=True
                    im=0
    
                self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
                if np.max(im) == 0 or imdropped==True:
                    msg = wx.MessageBox('Invalid/Deleted image. Click Yes to remove the image from the annotation file.', 'Error!', wx.YES_NO | wx.ICON_WARNING)
                    if msg == 2:
                        self.Dataframe = self.Dataframe.drop(self.index[self.iter])
                        self.droppedframes.append(self.index[self.iter])
                        self.index = list(self.Dataframe.iloc[:,0].index)
                    
            if self.file == 0:
                self.checkBox = wx.CheckBox(self.top_split, label = 'Adjust marker size.',pos = (500, 855))
                self.checkBox.Bind(wx.EVT_CHECKBOX,self.onChecked)
                self.slider = wx.Slider(self.top_split, -1, self.markerSize, 0, 20,size=(200, -1),  pos=(500, 780),style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
                self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
                self.slider.Enable(False)
            
            self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
            self.colorparams = list(range(0,(self.num_joints+1)))
            MainFrame.plot(self,im,im_axis)
            self.toolbar = NavigationToolbar(self.canvas)
            
            if self.adjust_original_labels == False:

                instruction = wx.MessageBox('1. Enter the likelihood threshold. \n\n2. Each prediction will be shown with a unique color. \n All the data points above the threshold will be marked as circle filled with a unique color. All the data points below the threshold will be marked with a hollow circle. \n\n3. Enable the checkbox to adjust the marker size. \n\n4.  Hover your mouse over data points to see the labels and their likelihood. \n\n5. Left click and drag to move the data points. \n\n6. Right click on any data point to remove it. Be careful, you cannot undo this step. \n Click once on the zoom button to zoom-in the image.The cursor will become cross, click and drag over a point to zoom in. \n Click on the zoom button again to disable the zooming function and recover the cursor. \n Use pan button to pan across the image while zoomed in. Use home button to go back to the full;default view. \n\n7. When finished click \'Save\' to save all the changes. \n\n8. Click OK to continue', 'User instructions', wx.OK | wx.ICON_INFORMATION)

                if instruction == 4 :
                    """
                    If ok is selected then the image is updated with the thresholded value of the likelihood
                    """
                    textBox = wx.TextEntryDialog(self, "Select the likelihood threshold",caption = "Enter the threshold",value="0.1")
                    textBox.ShowModal()
                    self.threshold = float(textBox.GetValue())
                    textBox.Destroy()
                    self.drs = []
                    self.updatedCoords = []
                    plt.close(self.fig1)
                    self.canvas.Destroy()
                    self.fig1, (self.ax1f1) = plt.subplots(figsize=(12, 7.8),facecolor = "None")
                    
                    imagename1 = os.path.join(self.project_path,self.index[self.iter])
                    im = PIL.Image.open(imagename1)
                    im_axis = self.ax1f1.imshow(im,self.colormap)
                    self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem) + " "+ " Threshold chosen is: " + str("{0:.2f}".format(self.threshold))))
                    self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
                    MainFrame.plot(self,im,im_axis)
                    MainFrame.confirm(self)
                    self.toolbar = NavigationToolbar(self.canvas)
                else:
                    self.threshold = 0.1

                self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem) + " "+ " Threshold chosen is: " + str("{0:.2f}".format(self.threshold))))
            else:
                self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem)))


        else:
            msg = wx.MessageBox('No Machinelabels file found! Want to Retry?', 'Error!', wx.YES_NO | wx.ICON_WARNING)
            if msg == 2:
                self.Button1.Enable(True)
                self.Button2.Enable(False)
                self.Button4.Enable(False)
            else:
                self.Destroy()

    def nextImage(self, event):
        """
        Reads the next image and enables the user to move the annotations
        """
        MainFrame.confirm(self)
        self.canvas.Destroy()
        plt.close(self.fig1)
        self.Button3.Enable(True)
        self.checkBox.Enable(False)
        self.slider.Enable(False)
        self.iter = self.iter + 1
        self.fig1, (self.ax1f1) = plt.subplots(figsize=(12, 7.8),facecolor="None")

        # Checks for the last image and disables the Next button
        if len(self.index) - self.iter == 1:
            self.Button2.Enable(False)

        if len(self.index) > self.iter:
            self.updatedCoords = []
            #read the image
            #imagename1 = os.path.join(self.dir,self.index[self.iter])
            imagename1 = os.path.join(self.project_path,self.index[self.iter])
            
            try:
                im = PIL.Image.open(imagename1)
                #Plotting
                im_axis = self.ax1f1.imshow(im,self.colormap)
    
                self.ax1f1.imshow(im)
                if self.adjust_original_labels == True:
                    self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem)))
                else:
                    self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem) + " "+ " Threshold chosen is: " + str("{0:.2f}".format(self.threshold))))
                
                imdropped=False
            except FileNotFoundError: #based on this flag, the image will be removed. 
                imdropped=True
                im=0

            self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
            if np.max(im) == 0 or imdropped==True:
                msg = wx.MessageBox('Invalid/Deleted image. Click Yes to remove the image from the annotation file.', 'Error!', wx.YES_NO | wx.ICON_WARNING)
                if msg == 2:
                    self.Dataframe = self.Dataframe.drop(self.index[self.iter])
                    self.droppedframes.append(self.index[self.iter])
                    self.index = list(self.Dataframe.iloc[:,0].index)
                    
                self.iter = self.iter - 1 #then display previous image.
                plt.close(self.fig1)

                #imagename1 = os.path.join(self.dir,self.index[self.iter])
#                im=io.imread(imagename1)
                imagename1 = os.path.join(self.project_path,self.index[self.iter])
                im = PIL.Image.open(imagename1)

                #Plotting
                im_axis = self.ax1f1.imshow(im,self.colormap)

                self.ax1f1.imshow(im)
                if self.adjust_original_labels == True:
                    self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem)))
                else:
                    self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ self.index[self.iter] + " "+ " Threshold chosen is: " + str("{0:.2f}".format(self.threshold))))
                self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
                print(self.iter)
            MainFrame.plot(self,im,im_axis)
            self.toolbar = NavigationToolbar(self.canvas)
        else:
            self.Button2.Enable(False)

    def prevImage(self, event):
        """
        Checks the previous Image and enables user to move the annotations.
        """

        MainFrame.confirm(self)
        self.canvas.Destroy()
        self.Button2.Enable(True)
        self.checkBox.Enable(False)
#        self.cb.SetValue(False)
        self.slider.Enable(False)
        plt.close(self.fig1)
        self.fig1, (self.ax1f1) = plt.subplots(figsize=(12, 7.8),facecolor="None")
        self.iter = self.iter - 1

        # Checks for the first image and disables the Previous button
        if self.iter == 0:
            self.Button3.Enable(False)

        if self.iter >= 0:
            self.updatedCoords = []
            self.drs = []
            # Reading Image
#            imagename1 = os.path.join(self.dir,"file%04d.png" % self.index[self.iter])
            #imagename1 = os.path.join(self.dir,self.index[self.iter])
#            im=io.imread(imagename1)
            imagename1 = os.path.join(self.project_path,self.index[self.iter])
            im = PIL.Image.open(imagename1)

            # Plotting
            im_axis = self.ax1f1.imshow(im,self.colormap)
#            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            if self.adjust_original_labels == True:
                self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem)))
            else:
                self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ str(Path(self.index[self.iter]).stem) + " "+ " Threshold chosen is: " + str("{0:.2f}".format(self.threshold))))
            self.canvas = FigureCanvas(self.top_split, -1, self.fig1)
            MainFrame.plot(self,im,im_axis)
            self.toolbar = NavigationToolbar(self.canvas)
        else:
            self.Button3.Enable(False)


    def quitButton(self, event):
        """
        Quits the GUI
        """
        plt.close('all')
        print("Closing..The refined labels are stored in a subdirectory under labeled-data. Use the function 'create_training_datasets' to augment the training dataset and train the network!")
        self.Destroy()

    def help(self,event):
        """
        Opens Instructions
        """
        if self.adjust_original_labels == True:
            wx.MessageBox('1. Each label will be shown with a unique color. \n\n3.  Hover your mouse over data points to see the labels. \n\n4. Left click and drag to move the data points. \n\n5. Right click on any data point to remove it. Be careful, you cannot undo this step.\n Click once on the zoom button to zoom-in the image.The cursor will become cross, click and drag over a point to zoom in. \n Click on the zoom button again to disable the zooming function and recover the cursor. \n Use pan button to pan across the image while zoomed in. Use home button to go back to the full;default view. \n\n6. When finished click \'Save\' to save all the changes. \n\n7. Click OK to continue', 'Instructions to use!', wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox('1. Enter the threshold of likelihood you want to check. \n\n2. Each prediction will be shown with a unique color. \n All the data points above the threshold will be marked as circle filled with a unique color. \n All the data points below the threshold will be marked as hollow circle. \n\n3.  Hover your mouse over data points to see the labels and their likelihood. \n\n4. Left click and drag to move the data points. \n\n5. Right click on any data point to remove it. Be careful, you cannot undo this step. \n Click once on the zoom button to zoom-in the image.The cursor will become cross, click and drag over a point to zoom in. \n Click on the zoom button again to disable the zooming function and recover the cursor. \n Use pan button to pan across the image while zoomed in. Use home button to go back to the full;default view. \n\n6. When finished click \'Save\' to save all the changes. \n\n7. Click OK to continue', 'Instructions to use!', wx.OK | wx.ICON_INFORMATION)

    def onChecked(self, event):
      MainFrame.confirm(self)
      self.cb = event.GetEventObject()
      if self.cb.GetValue() == True:
          self.slider.Enable(True)
      else:
          self.slider.Enable(False)
    
    def check_labels(self):
        print("Checking labels if they are outside the image")
        for i in self.Dataframe.index:
            image_name = os.path.join(self.project_path,i)
            im = PIL.Image.open(image_name)
            width, height = im.size
            for bpindex, bp in enumerate(self.bodyparts2plot):
                testCondition = self.Dataframe.loc[i,(self.scorer,bp,'x')] > width or self.Dataframe.loc[i,(self.scorer,bp,'x')] < 0 or self.Dataframe.loc[i,(self.scorer,bp,'y')] > height or self.Dataframe.loc[i,(self.scorer,bp,'y')] <0
                if testCondition:
                    print("Found %s outside the image %s.Setting it to NaN" %(bp,i))
                    self.Dataframe.loc[i,(self.scorer,bp,'x')] = np.nan
                    self.Dataframe.loc[i,(self.scorer,bp,'y')] = np.nan
        return(self.Dataframe)
        
    def save(self, event):

        MainFrame.confirm(self)
        plt.close(self.fig1)
        
        if self.adjust_original_labels == True:
            self.Dataframe = MainFrame.check_labels(self)
            self.Dataframe.to_hdf(os.path.join(self.dir,'CollectedData_'+self.humanscorer+'.h5'), key='df_with_missing', mode='w')
            self.Dataframe.to_csv(os.path.join(self.dir,'CollectedData_'+self.humanscorer+".csv"))
        else:
            self.Dataframe = MainFrame.check_labels(self)
            self.Dataframe.columns.set_levels([self.scorer.replace(self.scorer,self.humanscorer)],level=0,inplace=True)
            self.Dataframe = self.Dataframe.drop('likelihood',axis=1,level=2)

        if Path(self.dir,'CollectedData_'+self.humanscorer+'.h5').is_file():
            print("A training dataset file is already found for this video. The refined machine labels are merged to this data!")
            DataU1 = pd.read_hdf(os.path.join(self.dir,'CollectedData_'+self.humanscorer+'.h5'), 'df_with_missing')
            #combine datasets Original Col. + corrected machinefiles:
            DataCombined = pd.concat([self.Dataframe,DataU1])
            # Now drop redundant ones keeping the first one [this will make sure that the refined machine file gets preference]
            DataCombined = DataCombined[~DataCombined.index.duplicated(keep='first')]
            if len(self.droppedframes)>0: #i.e. frames were dropped/corrupt. also remove them from original file (if they exist!)
                for fn in self.droppedframes:
                    try:
                        DataCombined.drop(fn,inplace=True)
                    except KeyError:
                        pass
                    
                    
            DataCombined.to_hdf(os.path.join(self.dir,'CollectedData_'+ self.humanscorer +'.h5'), key='df_with_missing', mode='w')
            DataCombined.to_csv(os.path.join(self.dir,'CollectedData_'+ self.humanscorer +'.csv'))
        else:
            self.Dataframe.to_hdf(os.path.join(self.dir,'CollectedData_'+ self.humanscorer+'.h5'), key='df_with_missing', mode='w')
            self.Dataframe.to_csv(os.path.join(self.dir,'CollectedData_'+ self.humanscorer +'.csv'))
            self.Button2.Enable(False)
            self.Button3.Enable(False)
            self.slider.Enable(False)
            self.checkBox.Enable(False)

        nextFilemsg = wx.MessageBox('File saved. Do you want to refine another file?', 'Repeat?', wx.YES_NO | wx.ICON_INFORMATION)
        if nextFilemsg == 2:
            self.file = 1
            self.canvas.Destroy()
            plt.close(self.fig1)
            self.Button1.Enable(True)
            self.slider.Enable(False)
            self.checkBox.Enable(False)
            MainFrame.browseDir(self, event)
        else:
            print("Closing.. The refined labels are stored in a subdirectory under labeled-data. Use the function 'create_training_datasets' to augment the training dataset and re-train the network!")
            self.Destroy()

# ###########################################################################
# Other functions
# ###########################################################################
    def confirm(self):
        """
        Updates the dataframe for the current image with the new datapoints
        """
        plt.close(self.fig1)
        for bpindex, bp in enumerate(self.bodyparts2plot):
            if self.updatedCoords[bpindex]:
                self.Dataframe.loc[self.Dataframe.index[self.iter],(self.scorer,bp,'x')] = self.updatedCoords[bpindex][-1][0]
                self.Dataframe.loc[self.Dataframe.index[self.iter],(self.scorer,bp,'y')] = self.updatedCoords[bpindex][-1][1]

    def plot(self,im,im_axis):
        """
        Plots and call auxfun_drag class for moving and removing points.
        """

        # self.canvas = FigureCanvas(self, -1, self.fig1)
        cbar = self.fig1.colorbar(im_axis, ax = self.ax1f1)
        #small hack in case there are any 0 intensity images!

        maxIntensity = np.max(im)
        if maxIntensity == 0:
            maxIntensity = np.max(im) + 255
        cbar.set_ticks(range(12,np.max(im),int(np.floor(maxIntensity/self.num_joints))))
#        cbar.set_ticks(range(12,np.max(im),8))
        cbar.set_ticklabels(self.bodyparts2plot)
        normalize = mcolors.Normalize(vmin=np.min(self.colorparams), vmax=np.max(self.colorparams))
            # Calling auxfun_drag class for moving points around

        for bpindex, bp in enumerate(self.bodyparts2plot):
            color = self.colormap(normalize(bpindex))
            if 'CollectedData_' in self.fileName:
                self.points = [self.Dataframe[self.scorer][bp]['x'].values[self.iter],self.Dataframe[self.scorer][bp]['y'].values[self.iter],1.0]
                self.likelihood = self.points[2]
            else:
                self.points = [self.Dataframe[self.scorer][bp]['x'].values[self.iter],self.Dataframe[self.scorer][bp]['y'].values[self.iter],self.Dataframe[self.scorer][bp]['likelihood'].values[self.iter]]
                self.likelihood = self.points[2]

            if self.move2corner==True:
                ny,nx=np.shape(im)[0],np.shape(im)[1]
                if self.points[0]>nx or self.points[0]<0:
                    self.points[0]=self.center[0]
                if self.points[1]>ny or self.points[1]<0:
                    self.points[1]= self.center[1]

            if not ('CollectedData_' in self.fileName) and self.likelihood < self.threshold:
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, facecolor = 'None', edgecolor = color)]
            else:
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = color, alpha=self.alpha)]

            self.ax1f1.add_patch(circle[0])
            self.dr = auxfun_drag.DraggablePoint(circle[0],bp,self.likelihood,self.adjust_original_labels)
            self.dr.connect()
            self.drs.append(self.dr)
            self.updatedCoords.append(self.dr.coords)
# ###########################################################################
# Class for MatPlotLib Panel
# ###########################################################################

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

class MatplotPanel(wx.Panel):
    def __init__(self, parent,config):
        wx.Panel.__init__(self, parent,-1,size=(100,100))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)

def show(config):
    app = wx.App()
    frame = MainFrame(None,config).Show()
    app.MainLoop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()
