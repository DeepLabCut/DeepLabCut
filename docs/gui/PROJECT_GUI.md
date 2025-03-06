(project-manager-gui)=
# Interactive Project Manager GUI

As some users may be more comfortable working with an interactive interface, we wanted to provide an easy-entry point to the software. All the main functionality is available in an  easy-to-deploy GUI interface. Thus, while the many advanced features are not fully available in this Project GUI, we hope this gets more users up-and-running quickly.

**Release notes:** As of DeepLabCut 2.1+ now provide a full front-end user experience for DeepLabCut, and as of 2.3+ we changed the GUI from wxPython to PySide6 with napari support.

## Get Started:

(1) Install DeepLabCut using the simple-install with Anaconda found [here!](how-to-install)*.
Now you have DeepLabCut installed, but if you want to update it, either follow the prompt in the GUI which will ask you to upgrade when a new version is available, or just go into your env (activate DEEPLABCUT) then run:

` pip install 'deeplabcut[gui,modelzoo]'` *but please see [full install guide](how-to-install)!


(2) Open the terminal and run: `python -m deeplabcut`


<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/07ae2633-dc3e-4b6d-beec-27c08d9f8531/ezgif.com-gif-maker+%284%29.gif?format=2500w" width="80%">
</p>

Start at the Project Management Tab and work your way through the tabs to built your customized model and deploy it on new data.
We recommend to keep the terminal visible (as well as the GUI) so you can see the ongoing processes as you step through your project, or any errors that might arise.

- For specific napari-based labeling features, see the ["napari gui" docs](napari-gui-usage).
- To change from dark to light mode, set appearance at the top:
<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/5e41b01d-3101-40b2-9c53-129d8988370f/Screen+Shot+2022-10-09+at+3.45.46+PM.png?format=2500w
" width="30%">
</p>

## Video Demos: How to launch and run the Project Manager GUI:

**Click on the images!**

Note that currently the video demo is the wxPython version, but the logic is the same!

[![Watch the video](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572824438905-QY9XQKZ8LAJZG6BLPWOQ/ke17ZwdGBToddI8pDm48kIIa76w436aRzIF_cdFnEbEUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcLthF_aOEGVRewCT7qiippiAuU5PSJ9SSYal26FEts0MmqyMIhpMOn8vJAUvOV4MI/guilaunch.jpg?format=1000w)](https://youtu.be/KcXogR-p5Ak)

### Using the Project Manager GUI with the latest DLC code (single animals, plus objects): ⬇️

[![Watch the video](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1589046800303-OV1CCNZINWDMF1PZWCWE/ke17ZwdGBToddI8pDm48kB4PVlRPKDmSlQNbUD3wvXgUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYxCRW4BPu10St3TBAUQYVKcaja1QZ1SznGf7WzFOi-J6zLusnaF2VdeZcKivwxvFiDfGDqVYuwbAlftad9hfoui/dlc_gui_22.png?format=1000w)](https://www.youtube.com/watch?v=JDsa8R5J0nQ)

[Read more here](important-info-regd-usage)

### Using the Project Manager GUI with the latest DLC code (multiple identical-looking animals, plus objects):

[![Watch the video](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1589047147498-G1KTFA5BXR4PVHOOR7OG/ke17ZwdGBToddI8pDm48kJDij24pM2COisBTLIGjR1pZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PIel60EThn7SDFlTiSprUhmjQQHn9bhdY9dnQSKs8bCCo/Untitled.png?format=1000w)](https://www.youtube.com/watch?v=Kp-stcTm77g)

[Read more here](important-info-regd-usage)

## VIDEO DEMO: How to benchmark your data with the new networks and data augmentation pipelines:

[Watch the video](https://youtu.be/WXCVr6xAcCA)
