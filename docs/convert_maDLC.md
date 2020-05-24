### How to convert a pre-2.2 project for use with DeepLabCut 2.2 <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="150" title="DLC-live" alt="DLC LIVE!" align="right" vspace = "50">


If you have a pre-2.2 project (`labeled-data`) with a **single animal** that you want to use with DLC 2.2, i.e. use your older data to now train the new multi-task deep neural network, here is what you need to do. 

(1) We recommend you make a back-up of your project folder. 

(2) Open your `config.yaml` file (in any text editor, or python IDE such as PyCharm, Spyder, VScode, atom, etc). 

<p align="center">
<img src= https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1587946828128-VQQRJYYF4I5Q4TK4R7NF/ke17ZwdGBToddI8pDm48kDUwYPb5NcTX7SbsUW3p69pZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpz5alHTAeHWMjsyxt20uNzeb3sgcN8_6mzgExgMZEG-xs3TaY24DmEIA6oEFne2xjs/Screen+Shot+2020-04-23+at+10.32.53+PM.png?format=750w width="80%">
 </p>

- After `task, scorer, date` please add the following (i.e. in the image above, you would start adding below line 4):

```python
multianimalproject: true
individuals:
uniquebodyparts: []
multianimalbodyparts:
```
- Now, please name the animal you have a new name under individuals, i.e.:
```python
individuals:
- mouse1
```

- `"uniquebodyparts: []` can stay blank. This is used in "true" multi-animal scenarios where you might have other unique objects (i.e. 2 individual black mice plus the corners of the box)

- Please move your "bodyparts:" to "multianimalbodyparts:" (bodypart names must stay the same!)
```python
multianimalbodyparts:
- snout
- leftear
- rightear
- tailbase
```
then you can set `bodyparts: MULTI!`

(3) Save the config.yaml (be sure to double check for spacing or typos first!) and then run:
```python
deeplabcut.convert2_maDLC(path_config_file, userfeedback=True)
```

Now you will see that your data within `labeled-data` are converted to a new format, and the single animal format was saved for you under a new file named `CollectedData_ ...singleanimal.h5` and `.csv` as a back-up!

(4) Now, you can load this project `config.yaml` in the Project Manager GUI and create a multi-animal training set to begin training! 
