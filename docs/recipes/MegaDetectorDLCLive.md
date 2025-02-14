# 💚 MegaDetector+DeepLabCut 💜

[DeepLabCut-Live](https://github.com/DeepLabCut/DeepLabCut-live) is an open source and free real-time package from DeepLabCut that allows for real-time, low-latency pose estimation.  [The DeepLabCut-ModelZoo](http://modelzoo.deeplabcut.org/) is our growing collection of pretrained animal models for rapid deployment; no training is typically required to use these models. MegaDetector is a free open software trained to detect animals, people, and vehicles from camera trap images. Check [here](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md) for further information.

In this #cookbook recipe, we show you how to use MegaDetector to detect animals and run DeepLabCut-Live (using ModelZoo models) to get the pose estimation. This doc is contributed by 2022 DLC AI Resident [Nirel Kadzo](https://github.com/Kadzon) 💜!

## What is MegaDetector?

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661853650953-3L3EZYF69701J8FJZCPT/anim1.jpeg?format=500ww" width="250" title="DLC" alt="DLC" align="right" vspace = "5">

MegaDetector detects an animal and generates a bounding box around the animal. Thanks to [Sara Beery](https://beerys.github.io/) for visiting the #DLCAIResidents in the summer of 2022 to tell us more about this amazing project. An example result is shown:



## DeepLabCut-Live

DeepLabCut-Live! is a real-time package for running DeepLabCut. However, you can also use it as a lighter-weight
package for running DeeplabCut even if you don't need real-time. It's very useful to use in HPC or servers, or in Apps,
as we do here. To read more, check out the [docs](deeplabcut-live).

### MegaDetector meets DeepLabCut

The combination of MegaDetector and DeepLabCut now enables animal pose estimation on animal-bounded images. Here is an example of the `full_macaque` model, which is from  MacaquePose. Model contributed by Jumpei Matsumoto, at the Univ of Toyama. See their paper for many details [here](https://www.biorxiv.org/content/10.1101/2020.07.30.229989v2). If you use this model, please [cite their paper](https://doi.org/10.3389/fnbeh.2020.581154).

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661853652273-KG8FYHYDVJ5IBPY0UDVS/monkmddlc.png?format=500w" width="600" title="DLC" alt="DLC" align="center" vspace = "5">

# 🤗 HuggingFace App

We use the [Hugging Face](huggingface.co) spaces based on gradio to create an App of MegaDetector+DeepLabCut for you to interact with it and to try it out for yourself. Thanks to [Merve Noyan](https://github.com/merveenoyan) who visited the #DLCAIResidents in the summer of 2022 to teach us about their ecosystem, and thanks to the other App co-authors from the DLC Residency Program (see the [App page](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut)).

Let's get into the details of how to use the App:

1. Click on this link to be redirected to the [MegaDetector+DeepLabCut application](https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut) on **Hugging Face**.

2. Upload your image on the *Input Image* section or drag and drop.

3. Choose the features for your image

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661853652069-DDS019L4HA245HZOOI3F/toggle.png?format=500w" width="400" title="DLC" alt="DLC" align="center" vspace = "15">

- ``Select MegaDetector model`` lets you choose between md_v5a and md_v5b, you can find out more about them [here](https://github.com/microsoft/CameraTraps/releases). They run on YOLOv5 which makes it 3x-4x faster than prior versions.

- ``Select DeepLabCut Model`` choose the relevant ModelZoo model closest to the image you uploaded. The selected model will run on the image to predict the keypoints on the animal.

```{hint}
To get close to accurate keypoints in your model, the animal you upload into the interface should have the animal model listed in "Select DeepLabCut Model" panel.
```

- ``Run DLClive`` checkbox allows you to run DeepLabCut-Live directly on the image without MegaDetector. However, MegaDetector often simplifies the pose estimation by blocking out the pixels outside the bounding box. But no harm to run it (just might be slower), test it out for yourself ;)

- ``Set confidence threshold for animal detections`` in the example above, the confidence threshold is set for 0.8, this means MegaDetector will put a bounding box if it is >0.8 sure it is an animal. The image displayed has a 0.94 confidence level.

- ``Set confidence threshold for keypoints`` suggests how confident the model is about predicting the accurate key points on the animal. This is displayed by the opacity of the coloured keypoints on the animal.

- ``Set marker size, Set font size, Select keypoiny label font`` are design specs you can choose for yourself - we all love pretty plots!

4. Once set and you are satisfied with the image and features, submit the image. The expected output will display your input image: with the animal(s) surrounded by a bounding box (if used), the tracked keypoints, and the labels. A downloadable `JSON` file with the markings as shown below:

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661853655974-YUBP0QQ1LR144NT37TLP/outputdog.png?format=500w" width="400" title="DLC" alt="DLC" align="center" vspace = "15">

 Image from [Scientific American](https://www.scientificamerican.com/article/dogs-personalities-arent-determined-by-their-breed/).

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
To experiment with more camera trap images, check out [Lila Science!](https://lila.science/)
```

Examples have also been added to the Hugging Face interface where you can try out a variety of animals to get a feel of things and also add your own.

## Examples

We encourage you to try out and experiment on your camera trap or other animal images. Indeed, we found it is not only limited to camera trap images you can test it out with photos taken from your camera. Have a look at a 🦊picture Mackenzie took in Geneva and used the MegaDetector+DeepLabCut [Hugging Face](huggingface.co).

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661854010041-5RQGQTRSTKUDYU9KSTIE/foxGeneva.png?format=750ww" width="400" title="DLC" alt="DLC" align="center" vspace = "15">

Or these lil' cuties 🐶🐶🙀🐶 outside a restaurant, from the [Twitter meme](https://twitter.com/standardpuppies/status/1563188163962515457?s=21&t=f2kM2HoUygyLmmAH7Ho-HQ).

 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1661853654276-WEA4UUD7I1VEGHXSMIXE/pupscat.png?format=300w" width="400" title="DLC" alt="DLC" align="center" vspace = "15">

```{note}
DLC-Live allows you to process videos and frames in bulk, however the current release of MegaDetetctor+DeepLabCut-Live allows you to process one image at a time. But stay tuned for further releases, we are just getting started ;)
```


### Developer Mode:
To run it locally you can `git clone` the repository on your terminal and explore MegaDetector+DeepLabCut for yourself :)

In your terminal run each line:
```bash
git clone https://huggingface.co/spaces/DeepLabCut/MegaDetector_DeepLabCut

conda create -n megaDLC python==3.8
conda activate megaDLC

cd MegaDetector_DeepLabCut

pip install -r requirements.txt
python3 app.py
```

It should then print out links for you to locally load. Have fun and happy hacking!
