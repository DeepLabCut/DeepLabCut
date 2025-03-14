## DeepLabCut Self-paced Course

::::{warning}
This course was designed for DLC 2.
An updated version for DLC 3 is in the works.
::::

Do you have video of animal behaviors? Step 1: Get Poses ... 

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="150" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">

This document is an outline of resources for a course for those wanting to learn to use `Python` and `DeepLabCut`.
We expect it to take *roughly* 1-2 weeks to get through if you do it rigorously. To get the basics, it should take 1-2 days.

[CLICK HERE to launch the interactive graphic to get started!](https://view.genial.ly/5fb40a49f8a0ef13943d4e5e/horizontal-infographic-review-learning-to-use-deeplabcut) (mini preview below) Or, jump in below!

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1605642639913-OIUBVR8R0JLYZQPRIYIR/ke17ZwdGBToddI8pDm48kMMAxEenKbh651VJujierMxZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpw-CO5bsXt3Lwn3O5kv-PfTgGtLU9oye8D4J7Fixq38Gl-o9tfrEtbnqpPzC5bXTas/ezgif.com-gif-maker.gif?format=750w" width="95%">
</p>


## Installation:

You need Python and DeepLabCut installed!
- [See these "beginner docs" for help!](beginners-guide)

- **WATCH:** overview of conda:  [Python Tutorial: Anaconda - Installation and Using Conda](https://www.youtube.com/watch?v=YJC6ldI3hWk)


## Outline:

### **The basics of computing in Python, terminal, and overview of DeepLabCut:**

- **Learning:** Using the program terminal / cmd on your computer: [Video Tutorial!](https://www.youtube.com/watch?v=5XgBd6rjuDQ)

- **Learning:** although minimal to no Python coding is required (i.e. you could use the DLC GUI to run the full program without it), here are some resources you may want to check out. [Software Carpentry: Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/)

- **Learning:** learning and teaching signal processing, and overview from Prof. Demba Ba [talk at JupyterCon](https://www.youtube.com/watch?v=ywz-LLYwkQQ)

- **DEMO:** Can I DEMO DEEPLABCUT (DLC) quickly?
    - Yes: [you can click through this DEMO notebook](https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/COLAB/COLAB_DEMO_mouse_openfield.ipynb)
    - AND follow along with me: [Video Tutorial!](https://www.youtube.com/watch?v=DRT-Cq2vdWs)


- **WATCH:** How do you know DLC is installed properly? (i.e. how to use our test script!) [Video Tutorial!](https://youtu.be/IOWtKn3l33s)


<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1587608364285-A8R2F24K4DCP0KLAYI91/ke17ZwdGBToddI8pDm48kOhrDvKq54Xu9oStUCFZX0R7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z4YTzHvnKhyp6Da-NYroOW3ZGjoBKy3azqku80C789l0p4XabXLlNWpcJMv7FrN_NLe3GEN018us8vX03EdtIDHsW7dEh7nvL5CemxAxOy1gg/EKlIEXyXUAE0cy3.jpeg?format=1000w" width="350" title="DLC" alt="review!" align="right" vspace = "50">

- **REVIEW PAPER:** The state of animal pose estimation w/ deep learning i.e. "Deep learning tools for the measurement of animal behavior in neuroscience" [arXiv](https://arxiv.org/abs/1909.13868) & [published version](https://www.sciencedirect.com/science/article/pii/S0959438819301151)

- **REVIEW PAPER:** [A Primer on Motion Capture with Deep Learning: Principles, Pitfalls and Perspectives](https://www.sciencedirect.com/science/article/pii/S0896627320307170)


- **WATCH:** There are a lot of docs... where to begin: [Video Tutorial!](https://www.youtube.com/watch?v=A9qZidI7tL8)

### **Module 1: getting started on data**

**What you need:** any videos where you can see the animals/objects, etc.
You can use our demo videos, grab some from the internet, or use whatever older data you have. Any camera, color/monochrome, etc will work. Find diverse videos, and label what you want to track well :)
- IF YOU ARE PART OF THE COURSE: you will be contributing to the DLC Model Zoo 😊

   - **Slides:** [Overview of starting new projects](https://github.com/DeepLabCut/DeepLabCut-Workshop-Materials/blob/main/part1-labeling.pdf)
   - **READ ME PLEASE:** [DeepLabCut, the science](https://rdcu.be/4Rep)
   - **READ ME PLEASE:** [DeepLabCut, the user guide](https://rdcu.be/bHpHN)
   - **WATCH:** Video tutorial 1: [using the Project Manager GUI](https://www.youtube.com/watch?v=KcXogR-p5Ak)
     - Please go from project creation (use >1 video!) to labeling your data, and then check the labels!
   - **WATCH:** Video tutorial 2: [using the Project Manager GUI for multi-animal pose estimation](https://www.youtube.com/watch?v=Kp-stcTm77g)
     - Please go from project creation (use >1 video!) to labeling your data, and then check the labels!
   - **WATCH:** Video tutorial 3: [using ipython/pythonw (more functions!)](https://www.youtube.com/watch?v=7xwOhUcIGio)
      - multi-animal DLC: [labeling](https://www.youtube.com/watch?v=Kp-stcTm77g)
      - Please go from project creation (use >1 video!) to labeling your data, and then check the labels!


### **Module 2: Neural Networks**

   - **Slides:** [Overview of creating training and test data, and training networks](https://github.com/DeepLabCut/DeepLabCut-Workshop-Materials/blob/main/part2-network.pdf)
   - **READ ME PLEASE:** [What are convolutional neural networks?](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

   - **READ ME PLEASE:** Here is a new paper from us describing challenges in robust pose estimation, why PRE-TRAINING really matters - which was our major scientific contribution to low-data input pose-estimation - and it describes new networks that are available to you. [Pretraining boosts out-of-domain robustness for pose estimation](https://paperswithcode.com/paper/pretraining-boosts-out-of-domain-robustness)

       - **MORE DETAILS:** ImageNet: check out the original paper and dataset: http://www.image-net.org/

   - **REVIEW PAPER:** [A Primer on Motion Capture with Deep Learning: Principles, Pitfalls and Perspectives](https://www.sciencedirect.com/science/article/pii/S0896627320307170)


 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1603101997909-7IEYAYZYE9C8AX6SK7GR/ke17ZwdGBToddI8pDm48kND1NDuHF9nqrgeclEdLoeR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z4YTzHvnKhyp6Da-NYroOW3ZGjoBKy3azqku80C789l0qN_-Z3B7EvygvPOPmeOryWYMQ3pkjXJ5SX4aMqPMuK4PimCRlyu3R6yKl-KltrlZA/networks.jpg?format=2500w" width="350" title="DLC" alt="review!" align="right" vspace = "50">

  Before you create a training/test set, please read/watch:
   - **More information:** [Which types neural networks are available, and what should I use?](https://github.com/DeepLabCut/DeepLabCut/wiki/What-neural-network-should-I-use%3F-(Trade-offs,-speed-performance,-and-considerations))
   - **WATCH:** Video tutorial 1: [How to test different networks in a controlled way](https://www.youtube.com/watch?v=WXCVr6xAcCA)
     - Now, decide what model(s) you want to test.
        - IF you want to train on your CPU, then run the step `create_training_dataset`, in the GUI etc. on your own computer.
        - IF you want to use GPUs on google colab, [**(1)** watch this FIRST/follow along here!](https://www.youtube.com/watch?v=qJGs8nxx80A) **(2)** move your whole project folder to Google Drive, and then [**use this notebook**](https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/COLAB/COLAB_YOURDATA_TrainNetwork_VideoAnalysis.ipynb)

        **MODULE 2 webinar**: https://youtu.be/ILsuC4icBU0


### **Module 3: Evaluation of network performance**

   - **Slides** [Evaluate your network](https://github.com/DeepLabCut/DeepLabCut-Workshop-Materials/blob/master/part3-analysis.pdf)
   - **WATCH:** [Evaluate the network in ipython](https://www.youtube.com/watch?v=bgfnz1wtlpo)
      - why evaluation matters; how to benchmark; analyzing a video and using scoremaps, conf. readouts, etc.

### **Module 4: Scaling your analysis to many new videos**

Once you have good networks, you can deploy them. You can create "cron jobs" to run a timed analysis script, for example. We run this daily on new videos collected in the lab. Check out a simple script to get started, and read more below:

   - [Analyzing videos in batches, over many folders, setting up automated data processing](https://github.com/DeepLabCut/DLCutils/tree/master/SCALE_YOUR_ANALYSIS)

  - How to automate your analysis in the lab: [datajoint.io](https://datajoint.io), Cron Jobs: [schedule your code runs](https://www.ostechnix.com/a-beginners-guide-to-cron-jobs/)

### **Module 5: Got Poses? Now what ...**

Pose estimation took away the painful part of digitizing your data, but now what? There is a rich set of tools out there to help you create your own custom analysis, or use others (and edit them to your needs). Check out more below:

   - [Helper code and packages for use on DLC outputs](https://github.com/DeepLabCut/DLCutils)

   - Create your own machine learning classifiers: https://scikit-learn.org/stable/

   - **REVIEW PAPER:**  [Toward a Science of Computational Ethology](https://www.sciencedirect.com/science/article/pii/S0896627314007934)

   - **REVIEW PAPER:** The state of animal pose estimation w/ deep learning i.e. "Deep learning tools for the measurement of animal behavior in neuroscience" [arXiv](https://arxiv.org/abs/1909.13868) & [published version](https://www.sciencedirect.com/science/article/pii/S0959438819301151)

   - **REVIEW PAPER:** [Big behavior: challenges and opportunities in a new era of deep behavior profiling](https://www.nature.com/articles/s41386-020-0751-7)

   - **READ**: [Automated measurement of mouse social behaviors using depth sensing, video tracking, and machine learning](https://www.pnas.org/content/112/38/E5351)


*compiled and edited by Mackenzie Mathis*
