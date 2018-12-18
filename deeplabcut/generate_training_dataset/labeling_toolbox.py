"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu
"""

import os
import glob
import wx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os.path
from pathlib import Path
import argparse
import yaml
from deeplabcut.generate_training_dataset import auxfun_drag_label
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################


#minic small screen: 
#displaysize = (400, 400) 

#Note, if the variable Screens = 2, it assumes two screens in landscape next to eachother! If you use a different configuration, consider changing displaysize to your known display size. see troubleshooting for more information https://github.com/AlexEMG/DeepLabCut/wiki/Troubleshooting-Tips.

#On Windows, there can be a issue with the sizing on start, so you can scale it down then resize on your screen. Namely, set winHack=.5 and this solves this issue. Thanks to Federico Claudi for troubleshooting this with us! 
   
class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""

    def __init__(self, parent, config,Screens,scale_w,scale_h, winHack, img_scale):
        displaysize = wx.GetDisplaySize()

        w = displaysize[0]
        h = displaysize[1]
        self.gui_width = (w*scale_w)/Screens
        self.gui_height = (h*scale_h)


        #print("Scaled GUI width", self.gui_width, "and height", self.gui_height)
        if self.gui_width<600 or self.gui_height<500:
                print("Your screen width", w, "and height", h)
                print("Scaled GUI width", self.gui_width, "and height", self.gui_height)
                print("Please adjust scale_h and scale_w, or get a bigger screen!")
        
        self.size=displaysize
        
        wx.Frame.__init__(self, None, title="DeepLabCut2.0 - Labeling GUI", size=(self.gui_width*winHack, self.gui_height*winHack), style= wx.DEFAULT_FRAME_STYLE)

        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed) 

        self.SetBackgroundColour("#ffffff")

        buttons_list = []
        self.Button1 = wx.Button(self, -1, "Load Frames", size=(150, 40), pos=(self.gui_width*.1, self.gui_height*.9))
        self.Button1.Bind(wx.EVT_BUTTON, self.browseDir)
        self.Button1.Enable(True)
        buttons_list.append(self.Button1)

        self.Button5 = wx.Button(self, -1, "Help", size=(80, 40), pos=(self.gui_width*.3, self.gui_height*.9))
        self.Button5.Bind(wx.EVT_BUTTON, self.help)
        self.Button5.Enable(True)
        buttons_list.append(self.Button5)

        self.Button2 = wx.Button(self, -1, "Next Frame", size=(120, 40), pos=(self.gui_width*.4, self.gui_height*.9))
        self.Button2.Bind(wx.EVT_BUTTON, self.nextImage)
        self.Button2.Enable(False)
        buttons_list.append(self.Button2)
        
        self.Button4 = wx.Button(self, -1, "Save", size=(80, 40), pos=(self.gui_width*.6, self.gui_height*.9))
        self.Button4.Bind(wx.EVT_BUTTON, self.save)
        self.Button4.Enable(False)
        self.close = wx.Button(self, -1, "Quit", size=(80, 40), pos=(self.gui_width*.8, self.gui_height*.9))
        self.close.Bind(wx.EVT_BUTTON,self.quitButton)
        buttons_list.append(self.Button4)
        buttons_list.append(self.close)

# add buttons for  zoom
        # radio buttons position: (1250, 65)

        self.Button8 = wx.Button(self,-1,"Zoom", size=(60,30),pos=(self.gui_width*.65, self.gui_height*.85))
        self.Button8.Bind(wx.EVT_BUTTON,self.zoom)
        buttons_list.append(self.Button8)

        self.Button7 = wx.Button(self,-1,"Pan", size=(60,30),pos=(self.gui_width*.75, self.gui_height*.85))
        self.Button7.Bind(wx.EVT_BUTTON,self.pan)
        buttons_list.append(self.Button7)

        self.Button6 = wx.Button(self,-1,"Home", size=(60,30),pos=(self.gui_width*.85, self.gui_height*.85))
        self.Button6.Bind(wx.EVT_BUTTON,self.home)
        buttons_list.append(self.Button6)

        #for btn in buttons_list:
        #    btn.SetBackgroundColour((160, 160, 160))

# Define variables

        self.currentDirectory = os.getcwd()
        self.index = []
        self.iter = []
        self.colormap = cm.hsv #note will be overwritten by colormap from config file during execution
        
        self.file = 0

        self.updatedCoords = []

        self.dataFrame = None
        self.flag = True
        self.file = 0
        self.config_file = config
        self.addLabel = wx.CheckBox(self, label = 'Add new labels to existing dataset?',pos = (self.gui_width*.1, self.gui_height*.85))
        self.addLabel.Bind(wx.EVT_CHECKBOX,self.newLabel)
        self.new_labels = False
        imgW = self.gui_width*img_scale #was 12 inches (perhaps add dpi!)
        imgH = self.gui_height*img_scale    #was 7 inches 

        self.img_size = (imgW, imgH)  # width, height in inches. 
        
    def newLabel(self, event):
        self.chk = event.GetEventObject()
        if self.chk.GetValue() == True:
            self.new_labels = True
            self.addLabel.Enable(False)
        else:
            self.new_labels = False

# BUTTONS FUNCTIONS
    def OnKeyPressed(self, event=None):
        if event.GetKeyCode() == wx.WXK_RIGHT:
            self.nextImage(event=None)

    def zoom(self,event):
        self.statusbar.SetStatusText("Zoom")
        self.toolbar.zoom()
        self.Refresh(eraseBackground=True)
        
    def home(self,event):
        self.statusbar.SetStatusText("Home")
        self.toolbar.home()
        self.Refresh(eraseBackground=True)
         
    def pan(self,event):
        self.statusbar.SetStatusText("Pan")
        self.toolbar.pan()
        self.Refresh(eraseBackground=True)

    def quitButton(self, event):
        """
        Quits the GUI
        """
        self.Destroy()

    def help(self,event):
        """
        Opens Instructions
        """
        wx.MessageBox('1. Select one of the body parts from the radio buttons to add a label (if necessary change config.yaml first to edit the label names). \n\n2. RIGHT clicking on the image will add the selected label. \n The label will be marked as circle filled with a unique color. \n\n3. Hover your mouse over this newly added label to see its name. \n\n4. LEFT click and drag to move the label position. \n\n5. To change the marker size mark the checkbox and move the slider. Uncheck this after it is adjusted! Then advance to the next frame (you cannot zoom or pan again on this image). \n Change the markersize only after finalizing the position of your FIRST LABEL! \n\n6. Once you are happy with the position, select another body part from the radio button. \n Be careful, once you add a new body part, you will not be able to move the old labels. \n\n7. Click Next Frame to move to the next image. \n\n8. When finished labeling all the images, click \'Save\' to save all the labels as a .h5 file. \n\n9. Click OK to continue using the labeling GUI.', 'User instructions', wx.OK | wx.ICON_INFORMATION)

    def onClick(self,event):
        x1 = event.xdata
        y1 = event.ydata
        self.drs = []
        normalize = mcolors.Normalize(vmin=np.min(self.colorparams), vmax=np.max(self.colorparams))
        if event.button == 3:
            if self.rdb.GetSelection() in self.buttonCounter :
                try:
                    new_sel = self.buttonCounter[-1]+1
                    self.rdb.Select(new_sel)
                    self.buttonCounter.append(new_sel)
                except:
                    # fallback: warn user 
                    wx.MessageBox('%s is already annotated. \n Select another body part to annotate.' % (str(self.bodyparts[self.rdb.GetSelection()])), 'Error!', wx.OK | wx.ICON_ERROR)
            
            if self.flag == len(self.bodyparts):
                wx.MessageBox('All body parts are annotated! Click \'Save\' to save the changes. \n Click OK to continue.', 'Done!', wx.OK | wx.ICON_INFORMATION)
                self.canvas.mpl_disconnect(self.onClick)

            color = self.colormap(normalize(self.rdb.GetSelection()))
            circle = [patches.Circle((x1, y1), radius = self.markerSize, fc=color, alpha=0.5)]
            self.num.append(circle)
            self.ax1f1.add_patch(circle[0])
            self.dr = auxfun_drag_label.DraggablePoint(circle[0],self.bodyparts[self.rdb.GetSelection()])
            self.dr.connect()
            self.buttonCounter.append(self.rdb.GetSelection())
            self.dr.coords = [[x1,y1,self.bodyparts[self.rdb.GetSelection()],self.rdb.GetSelection()]]
            self.drs.append(self.dr)
            self.updatedCoords.append(self.dr.coords)
        elif event.button == 2:
            self.zoom(None)
        self.canvas.mpl_disconnect(self.onClick)

    def browseDir(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """
        from skimage import io
        dlg = wx.DirDialog(self, "Choose the directory where your extracted frames are saved:",
                           os.path.join(os.getcwd(), 'labeled-data'), style = wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.dir = dlg.GetPath()
            self.Button1.Enable(False)
            self.Button2.Enable(True)
            self.Button5.Enable(True)
        else:
            dlg.Destroy()
            self.Close(True)
        dlg.Destroy()
        with open(str(self.config_file), 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile)
        self.scorer = self.cfg['scorer']
        self.bodyparts = self.cfg['bodyparts']
        self.videos = self.cfg['video_sets'].keys()
        self.markerSize = self.cfg['dotsize']
        self.colormap = plt.get_cmap(self.cfg['colormap'])
        self.project_path=self.cfg['project_path']
        self.index = glob.glob(os.path.join(self.dir,'*.png'))
        print('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
        
        #self.relativeimagenames=self.index ##[n.split(self.project_path+'/')[1] for n in self.index]
        #self.relativeimagenames=[n.split(self.project_path+'/')[1] for n in self.index]
        self.relativeimagenames=['labeled'+n.split('labeled')[1] for n in self.index]
        
        self.fig1, (self.ax1f1) = plt.subplots(figsize=self.img_size,facecolor = "None")
        self.iter = 0
        self.buttonCounter = []
        im = io.imread(self.index[self.iter])

        im_axis = self.ax1f1.imshow(im, self.colormap)

        img_name = Path(self.index[self.iter]).name # self.index[self.iter].split('/')[-1]
        self.ax1f1.set_title(str(str(self.iter+1)+"/"+str(len(self.index)) +" "+ img_name ))
        self.canvas = FigureCanvasWxAgg(self,-1,self.fig1)
        self.toolbar = NavigationToolbar(self.canvas)

        #checks for unique bodyparts
        if len(self.bodyparts)!=len(set(self.bodyparts)):
          print("Error! bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again. Quiting for now!")
          self.Destroy()
          
        if self.new_labels == True:
          self.oldDF = pd.read_hdf(os.path.join(self.dir,'CollectedData_'+self.scorer+'.h5'),'df_with_missing')
          oldBodyParts = self.oldDF.columns.get_level_values(1)
          _, idx = np.unique(oldBodyParts, return_index=True)
          oldbodyparts2plot =  list(oldBodyParts[np.sort(idx)])
          self.bodyparts =  list(set(self.bodyparts) - set(oldbodyparts2plot))
          self.rdb = wx.RadioBox(self, id=1, label="Select a body part to annotate",pos=(self.gui_width*.83, self.gui_height*.1), choices=self.bodyparts, majorDimension =1,style=wx.RA_SPECIFY_COLS,validator=wx.DefaultValidator, name=wx.RadioBoxNameStr)
          self.option = self.rdb.Bind(wx.EVT_RADIOBOX,self.onRDB)
          cbar = self.fig1.colorbar(im_axis, ax = self.ax1f1)
          cbar.set_ticks(range(12,np.max(im),int(np.floor(np.max(im)/len(self.bodyparts)-1))))
          cbar.set_ticklabels(self.bodyparts)
        else:
          self.addLabel.Enable(False)
          cbar = self.fig1.colorbar(im_axis, ax = self.ax1f1)
          cbar.set_ticks(range(12,np.max(im),int(np.floor(np.max(im)/len(self.bodyparts)-1))))
          cbar.set_ticklabels(self.bodyparts)
          self.rdb = wx.RadioBox(self, id=1, label="Select a body part to annotate",pos=(self.gui_width*.83, self.gui_height*.1), choices=self.bodyparts, majorDimension =1,style=wx.RA_SPECIFY_COLS,validator=wx.DefaultValidator, name=wx.RadioBoxNameStr)
          self.option = self.rdb.Bind(wx.EVT_RADIOBOX,self.onRDB)


        self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
        self.flag = 0
        self.num = []
        self.counter = []
        self.presentCoords = []

        self.colorparams = list(range(0,len(self.bodyparts)+1))

        a = np.empty((len(self.index),2,))
        a[:] = np.nan
        for bodypart in self.bodyparts:
            index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
            #frame = pd.DataFrame(a, columns = index, index = self.index)
            frame = pd.DataFrame(a, columns = index, index = self.relativeimagenames)
            self.dataFrame = pd.concat([self.dataFrame, frame],axis=1)

        if self.file == 0:
            self.checkBox = wx.CheckBox(self, label = 'Adjust marker size.',pos = (self.gui_width*.43, self.gui_height*.85))
            self.checkBox.Bind(wx.EVT_CHECKBOX,self.onChecked)
            self.slider = wx.Slider(self, -1, 18, 0, 20,size=(200, -1),  pos=(self.gui_width*.40, self.gui_height*.78),style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.slider.Enable(True)

    def onRDB(self,event):
       self.option = self.rdb.GetSelection()
       self.counter.append(self.option)

    def nextImage(self,event):
        """
        Moves to next image
        """
        from skimage import io
        # Checks for the last image and disables the Next button + diesbt load the next if RIGHT arrow key pressed
        if len(self.index) - self.iter == 1:
            self.Button2.Enable(False)
            self.Button4.Enable(True)
            return

        self.file = 1
        MainFrame.saveEachImage(self)
        self.canvas.Destroy()
        plt.close(self.fig1)
        self.ax1f1.clear()
        self.iter = self.iter + 1
        #Refreshing the button counter
        self.buttonCounter = []
        self.rdb.SetSelection(0)
        self.fig1, (self.ax1f1) = plt.subplots(figsize=self.img_size,facecolor = "None")

        if len(self.index) > self.iter:
            self.updatedCoords = []
            #read the image
            im = io.imread(self.index[self.iter])
            #Plotting
            im_axis = self.ax1f1.imshow(im,self.colormap)
            cbar = self.fig1.colorbar(im_axis, ax = self.ax1f1)
            cbar.set_ticks(range(12,np.max(im),int(np.floor(np.max(im)/len(self.bodyparts)))))
            cbar.set_ticklabels(self.bodyparts)
            img_name = Path(self.index[self.iter]).name # self.index[self.iter].split('/')[-1]
            self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ img_name ))
            self.canvas = FigureCanvasWxAgg(self, -1, self.fig1)
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)

        # Recreate toolbar for zooming
        self.toolbar = NavigationToolbar(self.canvas)

    def saveEachImage(self):
        """
        Saves data for each image
        """
        plt.close(self.fig1)

        for idx, bp in enumerate(self.updatedCoords):
            self.dataFrame.loc[self.relativeimagenames[self.iter]][self.scorer, bp[0][-2],'x' ] = bp[-1][0]
            self.dataFrame.loc[self.relativeimagenames[self.iter]][self.scorer, bp[0][-2],'y' ] = bp[-1][1]

    def save(self,event):
        """
        Saves the final dataframe
        """
        MainFrame.saveEachImage(self)
        if self.new_labels == True:
            self.dataFrame = pd.concat([self.oldDF,self.dataFrame],axis=1)
        # Windows compatible
        self.dataFrame.to_csv(os.path.join(self.dir,"CollectedData_" + self.scorer + ".csv"))
        self.dataFrame.to_hdf(os.path.join(self.dir,"CollectedData_" + self.scorer + '.h5'),'df_with_missing',format='table', mode='w')

        nextFilemsg = wx.MessageBox('File saved. Do you want to label another data set?', 'Repeat?', wx.YES_NO | wx.ICON_INFORMATION)
        if nextFilemsg == 2:
            self.file = 1
            plt.close(self.fig1)
            self.canvas.Destroy()
            self.rdb.Destroy()
            self.buttonCounter = []
            self.updatedCoords = []
            self.dataFrame = None
            self.counter = []
            self.bodyparts = []
            self.Button1.Enable(True)
            self.slider.Enable(False)
            self.checkBox.Enable(False)
            self.new_labels = self.new_labels
            MainFrame.browseDir(self, event)
        else:
            self.Destroy()
            print("You can now check the labels, using 'check_labels' before proceeding. Then,  you can use the function 'create_training_dataset' to create the training dataset.")

    def onChecked(self, event):
      self.cb = event.GetEventObject()
      if self.cb.GetValue() == True:
          self.slider.Enable(True)
          self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
      else:
          self.slider.Enable(False)

    def OnSliderScroll(self, event):
        """
        Adjust marker size for plotting the annotations
        """
        from skimage import io
        self.drs = []
        plt.close(self.fig1)
        self.canvas.Destroy()
        self.fig1, (self.ax1f1) = plt.subplots(figsize=self.img_size,facecolor = "None")
        self.markerSize = (self.slider.GetValue())
        im = io.imread(self.index[self.iter])
        im_axis = self.ax1f1.imshow(im,self.colormap)
        cbar = self.fig1.colorbar(im_axis, ax = self.ax1f1)
        cbar.set_ticks(range(12,np.max(im),int(np.floor(np.max(im)/len(self.bodyparts)))))
        cbar.set_ticklabels(self.bodyparts)
        img_name = Path(self.index[self.iter]).name #self.index[self.iter].split('/')[-1]
        self.ax1f1.set_title(str(str(self.iter)+"/"+str(len(self.index)-1) +" "+ img_name ))
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig1)
        normalize = mcolors.Normalize(vmin=np.min(self.colorparams), vmax=np.max(self.colorparams))

        for idx, bp in enumerate(self.updatedCoords):
            col = self.updatedCoords[idx][-1][-1]
            #color = self.colormap(normalize(col))
            x1 = self.updatedCoords[idx][-1][0]
            y1 = self.updatedCoords[idx][-1][1]
            circle = [patches.Circle((x1, y1), radius=self.markerSize, alpha=0.5)]
            self.ax1f1.add_patch(circle[0])
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, Toolbar

class MatplotPanel(wx.Panel):
    def __init__(self, parent,config):
        panel1 = wx.Panel.__init__(self, parent)
        self.figure = Figure()
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.toolbar = Toolbar(self.canvas)  # matplotlib toolbar
        self.toolbar.EnableTool(1,False)
        self.toolbar.Realize()
        # self.toolbar.set_active([0,1])
        self.toolbar.update()
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        # Best to allow the toolbar to resize!
        #sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
        self.Refresh(eraseBackground=True)

def GetToolBar(self): #https://matplotlib.org/examples/user_interfaces/embedding_in_wx3.html
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar        
     

def show(config,Screens=1,scale_w=.8,scale_h=.9, winHack=1, img_scale=0.0075):
    app = wx.App()
    frame = MainFrame(None,config,Screens,scale_w,scale_h, winHack, img_scale).Show()
    app.MainLoop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()

