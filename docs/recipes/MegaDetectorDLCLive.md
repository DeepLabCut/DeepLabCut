# Introduction to MegaDetector+DeepLabCut-Live!
[DeepLabCut-Live](https://github.com/DeepLabCut/DeepLabCut-live) is an open source and free real-time package from DeepLabCut that allows for real-time, low-latency pose estimation.  MegaDetector is a free open software trained to detect animals, people, and vehicles from camera trap images. Check [here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) for further information. 

In this recipe, we should you how to use MegaDetector to detect animals and run DeepLabCut-Live to get the pose estimation and is contributed by 2022 DLC AI Resident [Nirel Kadzo](https://github.com/Kadzon) 💜! This has then downstream appications in behavioral analysis and species identifaction. 

**MegaDetector**

MegaDetector detects an animal and generates a bounding box around the animal. Thanks to [Sara Beery](https://beerys.github.io/) for visting the #DLCAIResidents in the summer of 2022 to tell us more about this amazing project. Example results are shown below:

```{image} ../images/anim2.jpg
:alt: fishy
:class: bg-primary mb-1
:width: 200px
:align: center
```
```{image} ../images/anim1.jpg
:alt: fishy
:class: bg-primary mb-1
:width: 200px
:align: center
```
**MegaDetector+DeepLabCut-Live**

The combination of MegaDetector and DeepLabCut has enables animal pose estimation on images.
```{image} ../images/monkmddlc.png
:alt: fishy
:class: bg-primary mb-1
:width: 300px
:align: center
```

## 🤗 HuggingFace App

We use the [Hugging Face](huggingface.co) application to create an App of MegaDetector+DeepLabCut-Live for you to interact with it and to try it out for yourself! Thanks to [Merve Noyan](https://github.com/merveenoyan) who visted the #DLCAIResidents in the summer of 2022 to teach us about their ecosystem, and thanks to the other App co-authors from the DLC Residency Program(see [App page](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut)). 

Let's get into the details of how to use the App:

1. Click on this link to be redirected to the [MegaDetector+DeepLabCut-Live application](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut) on **Hugging Face**.

2. Upload your image on the *Input Image* section or drag and drop.

3. Choose the features for your image
```{image} ../images/toggle.PNG
:alt: fishy
:class: bg-primary mb-1
:width: 500px
:align: center

```
``Select MegaDetector model`` lets you choose between md_v5a and md_v5b, you can find out more about them [here](https://github.com/microsoft/CameraTraps/releases). They run on YOLOv5 which makes it 3x-4x faster than prior versions.

``Select DeepLabCut Model`` choose the relevant ModelZoo model closest to the image you uploaded. The selected model will run on the image to predict the keypoints on the animal. 

```{warning}
To get close to accurate keypoints in your model, the animal you upload into the interface should have the animal model listed in "Select DeepLabCut model" panel.
```

``Run DLClive`` checkbox allows you to run DeepLabCut-Live directly on the image without MegaDetector. However, MegaDetector often simplifies the pose estimation by blocking out the pixels outside the bounding box. But no harm to run it (just might be slower), test it out for yourself ;)

``Set confidence threshold for animal detections`` in the example above, the confidence threshold is set for 0.8, this means MegaDetector will put a bounding box if it is >0.8 sure it is an animal. The image displayed has a 0.94 confidence level.

``Set confidence threshold for keypoints`` suggests how confident the model is about predicting the accurate key points on the animal. This is displayed by the opacity of the coloured keypoints on the animal.

``Set marker size, Set font size, Select keypoiny label font`` are design specs you can choose for yourself - we all love pretty plots!

4. Once set and you are satisfied with the image and features, submit the image. The expected output will display your input image: with the animal(s) surrounded by a bounding box (if used), the tracked keypoints, and the labels. A downloadable `JSON` file with the markings as shown below:

```{image} ../images/outputdog.PNG
:alt: fishy
:class: bg-primary mb-1
:width: 500px
:align: center
```
All information seen on the output image is recorded on the **Download JSON file**. The snippet below is commented on to give you an overall understanding of what the code means 😀
```
{
 "date": "2022-08-26",
 "MD_model": "md_v5a",  //Megadetector model used
 "file": "image0.jpg",  //image filename uploaded
 "number_of_bb": 1,     //number of bounding boxes detected on the image
 "dlc_model": "full_dog",  //model used
 "bb_0": {              
  "corner_1": [          //top left corner
   76.08082580566406,    //x  
   91.02932739257812     //y
  ],
  "corner_2": [          //bottom right corner
   393.8626708984375,    //x
   399.9506530761719     //y
  ],
  "predict MD": "animal",  // MegaDetector predicts the image is in class animal
  "confidence MD": 0.9437874555587769,  // 0.94% confident it's an animal
  "dlc_pred": {     //keypoints prediction coordinates
   "Nose": [        //label
    264.89501953125,    //x
    89.19121551513672,  //y
    0.9611953496932983  //z
   ],
   ...
}
```


```{hint}
To experiment with more camera trap images, check out [Lila Science](https://lila.science/)
```

Examples have also been added to the Hugging Face interface where you can try out a variety of animals to get a feel of things and also add your own!

**Examples!** 

We encourage you to try out and experiment on your camera trap or other animal images. Indeed, we found it is not only limited to camera trap images you can test it out with photos taken from your camera. Have a look at a 🦊picture Mackenzie took in Geneva and used the MegaDetector+DeepLabCut-Live [Hugging Face](huggingface.co) App on.

```{image} ../images/foxmac.avif
:alt: fishy
:class: bg-primary mb-1
:width: 200px
:align: center
```

Or these lil' cuties 🐶🐶🙀🐶 outside a restaurant, from the [Twitter meme](https://twitter.com/standardpuppies/status/1563188163962515457?s=21&t=f2kM2HoUygyLmmAH7Ho-HQ).

```{image} ../images/pupscat.PNG
:alt: fishy
:class: bg-primary mb-1
:width: 300px
:align: center
```

```{note}
DLC-Live allows you to process videos and frames in bulk, however the current release of MegaDetetctor+DeepLabCut-Live allows you to process one image at a time. But stay tuned for further releases, we are just getting started ;)
```


### Developer Mode:
To run it locally you can `git clone` the repository on your terminal and explore MegaDetector+DeepLabCut-Live for yourself :) 

In your terminal:
```bash
git clone git+https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut.git

conda env create --name MegaDLC python=3.8
conda activate MegaDLC

pip install -e .

cd MegaDetector_DeepLabCut 
python3 app.py
```

It should then print out links for you to locally test and edit the code. Have fun and happy hacking!

