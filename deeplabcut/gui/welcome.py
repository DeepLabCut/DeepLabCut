"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""

import os

import wx


from deeplabcut.gui import MEDIA_PATH


class Welcome(wx.Panel):
    def __init__(self, parent, gui_size):
        h = gui_size[0]
        w = gui_size[1]
        wx.Panel.__init__(self, parent, -1, style=wx.SUNKEN_BORDER, size=(h, w))

        ##         design the panel
        sizer = wx.GridBagSizer(10, 7)
        # Add image of DLC
        icon = wx.StaticBitmap(
            self, bitmap=wx.Bitmap(os.path.join(MEDIA_PATH, "dlc_1-01.png"))
        )
        sizer.Add(icon, pos=(0, 0), span=(0, 8), flag=wx.EXPAND | wx.BOTTOM, border=10)
        line = wx.StaticLine(self)
        sizer.Add(line, pos=(1, 0), span=(1, 8), flag=wx.EXPAND | wx.BOTTOM, border=10)

        from deeplabcut import VERSION
        # if editing this text make sure you add the '\n' to get the new line. The sizer is unable to format lines correctly.
        description = f"DeepLabCut™ is an open source tool for markerless\npose estimation of user-defined body parts with deep learning.\nA. and M.W. Mathis Labs | http://www.deeplabcut.org\n \nWelcome to the DeepLabCut Project Manager GUI v{VERSION}!\nTo get started, please click on the 'Manage Project'\n tab to create or load an existing project. \n "

        self.proj_name = wx.StaticText(self, label=description, style=wx.ALIGN_CENTRE)
        sizer.Add(self.proj_name, pos=(2, 3), border=10)
        sizer.AddGrowableCol(2)
        self.SetSizer(sizer)
        sizer.Fit(self)
