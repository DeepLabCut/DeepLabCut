# Introduction to MegaDetector-DeepLabCut-Live
[DeepLabCut-Live](https://github.com/DeepLabCut/DeepLabCut-live) is a new package from DeepLabCut, a deep learning tool that allows for realtime pose estimation. 
MegaDetector is a free open software trained to detect animals, people, and vehicles from camera trap images. Check [here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) for further information. In this recipe, we use MegaDetector to detect animals and run DeepLabCut-Live to get the pose estimation!

**MegaDetector**

MegaDetector detects an animal and generates a bounding box around the animal as shown below.

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
**MegaDetector-DeepLabCut-Live**

The combination of MegaDetector and DeepLabCut has enabled animal pose estimation on images.
```{image} ../images/monkmddlc.png
:alt: fishy
:class: bg-primary mb-1
:width: 300px
:align: center
```

**Considerations** 

We encourage you to try out and experiment on your camera trap images. It is not only limited to camera trap images you can test it out with photos taken from your camera.
Have a look at picture Mackenzie took from her backyard and used the MegaDetector-DeepLabCut-Live app.

```{image} ../images/foxmac.avif
:alt: fishy
:class: bg-primary mb-1
:width: 200px
:align: center
```

Or these lil cuties outside a restaurant.

```{image} ../images/pupscat.PNG
:alt: fishy
:class: bg-primary mb-1
:width: 300px
:align: center
```

## How this works
We use the [Hugging Face](huggingface.co) application to render the application of DeepLabCut-Live and MegaDetector for you to interact with it and to try it out for yourself!
Let's get into the details of how to use the DLC-Live-Megadetector:

1. Click on this link to be redirected to the [MegaDetector v5 + DeepLabCut-Live application](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut) on **Hugging Face**.

2. Upload your image on the *Input Image* section or drag and drop, I find that easier.

3. Choose the features for your image
```{image} ../images/toggle.PNG
:alt: fishy
:class: bg-primary mb-1
:width: 500px
:align: center

```
``Select MegaDetector model`` lets you choose between md_v5a and md_v5b, you can find out more about it [here](https://github.com/microsoft/CameraTraps/releases). They run on yolov5 which makes it 3x-4x faster.

``Select DeepLabCut model`` choose the relevant DLC-Live model closest to the image you uploaded. The selected model will run on the image to predict the keypoints on the animal. 

```{warning}
To get close to accurate keypoints in your model, the animal you upload into the interface should have the animal model listed in "Select DeepLabCut model" panel.
```

``Run DLClive`` checkbox allows you to run DeepLabCut-Live directly on the image without MegaDetector. However, MegaDetector simplifies the pose estimation by blocking out the pixels outside the bounding box. But no harm, test it out for yourself ;)

``Set confidence threshold for animal detections`` in the example above, the confidence threshold is set for 0.8, this means MegaDetector will put a bounding box if it is >0.8 sure it is an animal. The image displayed has a 0.94 confidence level.

``Set confidence threshold for keypoints`` suggests how confident the model is about predicting the accurate key points on the animal. This is displayed by the opacity of the coloured keypoints on the animal.

``Set marker size, Set font size, Select keypoiny label font`` are design specs you can choose for yourself

4. Once set and satisfied with the image and features, submit the image. To know it's working, the expected output will display your input image: with the animal(s) surrounded by a bounding box, tracked keypoints and labels. A downloadable JSON file with the markings as shown below:

```{image} ../images/outputdog.PNG
:alt: fishy
:class: bg-primary mb-1
:width: 500px
:align: center
```
All information seen on the output image is recorded on the **Download JSON file**. The snippet below is commented on to give you an overall understanding of what the code means :)
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

```{note}
DLC-Live allows you to process videos and frames in bulk, however the current release of MegaDetetctor-DLC-Live allows you to process one image at a time. But stay tuned for further releases, we are just getting started ;)
```


```{hint} 
To run it locally you can git clone the repository on your terminal and explore MegaDetector-DeepLabCut-Live for yourself :) 
```
Here are the steps:
```
git clone https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut
```
```
conda activate base
```

```
cd MegaDetector_DeepLabCut 
```

```
python app.py
```

Running on local URL:  http://127.0.0.1:7860/
Running on public URL: https://49728.gradio.app

Have fun and happy hacking!

