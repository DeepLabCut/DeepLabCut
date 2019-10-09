"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import wx
import os
import sys
import pydoc
import deeplabcut
media_path = os.path.join(deeplabcut.__path__[0], 'gui' , 'media')
logo = os.path.join(media_path,'logo.png')
#import wx.lib.scrolledpanel as SP

class Create_training_dataset(wx.Panel):
    """
    """

    def __init__(self, parent,gui_size,cfg):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)

        # variable initilization
        self.method = "automatic"
        self.config = cfg
        # design the panel
        self.sizer = wx.GridBagSizer(5, 5)

        text = wx.StaticText(self, label="DeepLabCut - Step 4. Create training dataset")
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
        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)

        shuffles_text = wx.StaticBox(self, label="Specify the number of shuffles")
        shuffles_text_boxsizer = wx.StaticBoxSizer(shuffles_text, wx.VERTICAL)
        self.shuffles = wx.SpinCtrl(self, value='1',min=1,max=100)
        shuffles_text_boxsizer.Add(self.shuffles,1, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        net_text = wx.StaticBox(self, label="Select the network")
        netboxsizer = wx.StaticBoxSizer(net_text, wx.VERTICAL)
        self.net_choice = wx.ComboBox(self, style=wx.CB_READONLY)
        options = ['resnet_50', 'resnet_101','resnet_152','mobilenet_v2_1.0','mobilenet_v2_0.75','mobilenet_v2_0.5','mobilenet_v2_0.35']
        self.net_choice.Set(options)
        self.net_choice.SetValue('resnet_50')
        netboxsizer.Add(self.net_choice,20, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        aug_text = wx.StaticBox(self, label="Select the augmentation method")
        augboxsizer = wx.StaticBoxSizer(aug_text, wx.VERTICAL)
        self.aug_choice = wx.ComboBox(self, style=wx.CB_READONLY)
        options = ['default', 'tensorpack','imgaug']
        self.aug_choice.Set(options)
        self.aug_choice.SetValue('default')
        augboxsizer.Add(self.aug_choice,20, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)


        self.hbox1.Add(shuffles_text_boxsizer,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox1.Add(netboxsizer,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox1.Add(augboxsizer,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        self.model_comparison_choice = wx.RadioBox(self, label='Want to compare models?', choices=['Yes', 'No'],majorDimension=1, style=wx.RA_SPECIFY_COLS)
        self.model_comparison_choice.Bind(wx.EVT_RADIOBOX,self.chooseOption)
        self.model_comparison_choice.SetSelection(1)

        networks = ['resnet_50', 'resnet_101', 'resnet_152', 'mobilenet_v2_1.0','mobilenet_v2_0.75', 'mobilenet_v2_0.5', 'mobilenet_v2_0.35']
        augmentation_methods = ['default','tensorpack','imgaug']
        self.networks_to_compare = wx.CheckListBox(self, choices=networks, style=0,name = "Select the networks")
        self.networks_to_compare.Bind(wx.EVT_CHECKLISTBOX,self.get_network_names)

        self.augmentation_to_compare = wx.CheckListBox(self, choices=augmentation_methods, style=0,name = "Select the networks")
        self.augmentation_to_compare.Bind(wx.EVT_CHECKLISTBOX,self.get_augmentation_method_names)

        self.hbox2.Add(self.model_comparison_choice,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox2.Add(self.networks_to_compare,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)
        self.hbox2.Add(self.augmentation_to_compare,10, wx.EXPAND|wx.TOP|wx.BOTTOM, 5)

        self.networks_to_compare.Hide()
        self.augmentation_to_compare.Hide()

        boxsizer.Add(self.hbox1,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)
        boxsizer.Add(self.hbox2,0, wx.EXPAND|wx.TOP|wx.BOTTOM, 10)

        self.sizer.Add(boxsizer, pos=(3, 0), span=(1, 5),flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT , border=10)

        self.help_button = wx.Button(self, label='Help')
        self.sizer.Add(self.help_button, pos=(4, 0), flag=wx.LEFT, border=10)
        self.help_button.Bind(wx.EVT_BUTTON, self.help_function)

        self.ok = wx.Button(self, label="Ok")
        self.sizer.Add(self.ok, pos=(4, 4))
        self.ok.Bind(wx.EVT_BUTTON, self.create_training_dataset)

        self.reset = wx.Button(self, label="Reset")
        self.sizer.Add(self.reset, pos=(4, 1), span=(1, 1),flag=wx.BOTTOM|wx.RIGHT, border=10)
        self.reset.Bind(wx.EVT_BUTTON, self.reset_create_training_dataset)

        self.sizer.AddGrowableCol(2)

        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
        self.Layout()

    def on_focus(self,event):
        pass


    def help_function(self,event):

        filepath= 'help.txt'
        f = open(filepath, 'w')
        sys.stdout = f
        fnc_name = 'deeplabcut.create_training_dataset'
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

    def chooseOption(self,event):
        if self.model_comparison_choice.GetStringSelection() == 'Yes':
            self.networks_to_compare.Show()
            self.augmentation_to_compare.Show()
            self.net_choice.Enable(False)
            self.aug_choice.Enable(False)
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)
            self.get_network_names(event)
            self.get_augmentation_method_names(event)
        else:
            self.net_choice.Enable(True)
            self.aug_choice.Enable(True)
            self.networks_to_compare.Hide()
            self.augmentation_to_compare.Hide()
            self.SetSizer(self.sizer)
            self.sizer.Fit(self)

    def get_network_names(self,event):
        self.net_type = list(self.networks_to_compare.GetCheckedStrings())

    def get_augmentation_method_names(self,event):
        self.aug_type = list(self.augmentation_to_compare.GetCheckedStrings())

    def create_training_dataset(self,event):
        """
        """
        num_shuffles = self.shuffles.GetValue()
        if self.model_comparison_choice.GetStringSelection() == 'No':
            deeplabcut.create_training_dataset(self.config,num_shuffles,net_type=self.net_choice.GetValue(),augmenter_type = self.aug_choice.GetValue())
        if self.model_comparison_choice.GetStringSelection() == 'Yes':
            deeplabcut.create_training_model_comparison(self.config,num_shuffles,net_types=self.net_type,augmenter_types=self.aug_type)


    def reset_create_training_dataset(self,event):
        """
        Reset to default
        """
        self.config = []
        self.sel_config.SetPath("")
        self.shuffles.SetValue('1')
        self.net_choice.SetValue('resnet_50')
        self.aug_choice.SetValue('default')
        self.model_comparison_choice.SetSelection(1)
        self.networks_to_compare.Hide()
        self.augmentation_to_compare.Hide()
        self.net_choice.Enable(True)
        self.aug_choice.Enable(True)
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
        self.Layout()
