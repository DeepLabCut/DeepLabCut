(convert-maDLC)=
# How to convert a pre-2.2 project for use with DeepLabCut 2.2 or later

<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1572296495650-Y4ZTJ2XP2Z9XF1AD74VW/ke17ZwdGBToddI8pDm48kMulEJPOrz9Y8HeI7oJuXxR7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z5QPOohDIaIeljMHgDF5CVlOqpeNLcJ80NK65_fV7S1UZiU3J6AN9rgO1lHw9nGbkYQrCLTag1XBHRgOrY8YAdXW07ycm2Trb21kYhaLJjddA/DLC_logo_blk-01.png?format=1000w" width="150" title="DLC" alt="DLC!" align="right" vspace = "50">


If you have a pre-2.2 project (`labeled-data`) with a **single animal** that you want to use with a multianimal project
in DLC 2.2 or later, i.e. use your older data to now train the new multi-task deep neural network, here is what you
need to do.

(1) We recommend you make a back-up of your project folder.

(2) Open your `config.yaml` file (in any text editor, or python IDE such as PyCharm, Spyder, VScode, atom, etc).

<p align="center">
<img src= https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1587946828128-VQQRJYYF4I5Q4TK4R7NF/ke17ZwdGBToddI8pDm48kDUwYPb5NcTX7SbsUW3p69pZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpz5alHTAeHWMjsyxt20uNzeb3sgcN8_6mzgExgMZEG-xs3TaY24DmEIA6oEFne2xjs/Screen+Shot+2020-04-23+at+10.32.53+PM.png?format=750w width="80%">
 </p>

- After `task, scorer, date, project_path` please add the following (i.e. in the image above, you would start adding
below line 6) Note, the ordering isn't important but useful to keep consistent with the template:

```python
multianimalproject: true
individuals:
uniquebodyparts: []
multianimalbodyparts:
identity: false/true
```
- Now, please name the animal you have a new name under individuals, i.e.:
```python
individuals:
- mouse1
```

- `"uniquebodyparts: []` can stay blank, unless you have other items labeled you want to estimate (consider these as
similar to bodyparts in pre-2.2); i.e. corners of a box, etc. All unique bodyparts should not be connected to the
multianimal bodyparts in the skeleton you will eventually make. See "advanced option" below.

- Please move your "bodyparts:" to "multianimalbodyparts:" (bodypart names must stay the same!) These are the parts
that will always be interconnected fully!
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

Now you will see that your data within `labeled-data` are converted to a new format, and the single animal format was
saved for you under a new file named `CollectedData_ ...singleanimal.h5` and `.csv` as a back-up!

(4) We strongly recommend to first run check_labels and verify that the conversion was as expected before creating a
multianimal training dataset. For instance, you can load this project `config.yaml` in the Project Manager GUI and
check labels then create a multi-animal training set with
```python
deeplabcut.create_multianimaltraining_dataset(path_config_file)
```
to begin training.

**Advanced option:** You can also assign former `bodyparts` to either `uniquebodyparts` or `multianimalbodyparts`
(you can even leave some unassigned, which means they will be dropped in the conversion).

Example: Imagine you had a project with the moon and a rocket with two parts labeled:
`bodyparts: [moon, rocket_tip,rocket_bottom]`

Now you want to use this former project (labeled-data) and work on a new dataset (videos) with one moon but multiple
(3) rockets. Then convert it as follows:
```
individuals: [rocket1, rocket2, rocket3]
uniquebodyparts: [moon]
multianimalbodyparts: [rocket_tip,rocket_bottom]
skeleton: [[[rocket_tip,rocket_bottom]]]
```
In the unusual case, that your data also has multiple moons (e.g. is now carried out around Jupiter), but one rocket:
```
individuals: [Io, Europa, Ganymede, Callisto]
uniquebodyparts: [rocket_tip,rocket_bottom]
multianimalbodyparts: [moon]
```
Note you can use the single object tracker for this situation. What if you have multiple moons and rockets?
