
# Clustering in Napari-DeepLabCut

To increase training performance, one can find the errors in the output and correct them by relabeling. To facilitate this process, here we propose a new way to detect 'outlier frames'.

This recipe aims to show a usecase of the **clustering in napari** and is contributed by 2022 DLC AI Resident [Sabrina Benas](https://twitter.com/Sabrineiitor) 💜.


## Detect Outliers to Refine Labels 

- Open napari and the DeepLabCut plugin and open the CollectedData_<ScorerName>.h5 file. Here is an example of what it should look like: 


```{image} ../images/1.png
:class: bg-primary mb-1
:width: 600px
:align: center
```


- Click on the button cluster and wait a few seconds. It will show a new layer with the cluster: 

```{image} ../images/2.png
:class: bg-primary mb-1
:width: 600px
:align: center
```

- You can click on a point and see the image on the right with the keypoints: 



```{image} ../images/2_b.png
:class: bg-primary mb-1
:width: 600px
:align: center
```

- If you decided to refine that frame, click show img and refine them:

```{image} ../images/3.png
:class: bg-primary mb-1
:width: 600px
:align: center
```


- You can go back to the cluster layer by clicking on close img and refine another image. When you're done, you need to do **ctl-s** to save it. And now you can retrain the network!

```{hint} 
If you want to change the clustering method, you can modify the file [kmeans.py](https://github.com/DeepLabCutAIResidency/napari-deeplabcut/blob/cluster1/src/napari_deeplabcut/kmeans.py)

::::{important}
You have to keep the way the file is opened (pandas dataframe) and the output has to be the cluster points, the points colors in the cluster colors and the frame names (in this order).
::::

```