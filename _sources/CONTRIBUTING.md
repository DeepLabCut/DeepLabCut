# How to Contribute to DeepLabCut

DeepLabCut is an actively developed package and we welcome community development and involvement. We are especially seeking people from underrepresented backgrounds in OSS to contribute their expertise and experience. Please get in touch if you want to discuss specific contributions you are interested in developing, and we can help shape a road-map.

We are happy to receive code extensions, bug fixes, documentation updates, etc.

If you are a new user, we recommend checking out the detailed [Github Guides](https://guides.github.com).

## Setting up a development installation

In order to make changes to `deeplabcut`, you will need to [fork](https://guides.github.com/activities/forking/#fork) the
[repository](https://github.com/deeplabcut/deeplabcut).

If you are not familiar with `git`, we recommend reading up on [this guide](https://guides.github.com/introduction/git-handbook/#basic-git).

Here are guidelines for installing deeplabcut locally on your own computer, where you can make changes to the code! We often update the master deeplabcut code base on github, and then ~1 a month we push out a stable release on pypi. This is what most users turn to on a daily basis (i.e. pypi is where you get your `pip install deeplabcut` code from! 

But, sometimes we add things to the repo that are not yet integrated, or you might want to edit the code yourself, or you will need to do this to contribute. Here, we show you how to do this. 

**Step 1:**

- git clone the repo into a folder on your computer:  

- click on this green button and copy the link:

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1581984907363-G8AFGX4V20Y1XD1PSZAK/ke17ZwdGBToddI8pDm48kGJBV0_F4LE4_UtCip_K_3lZw-zPPgdn4jUwVcJE1ZvWEtT5uBSRWt4vQZAgTJucoTqqXjS3CfNDSuuf31e0tVE0ejQCe16973Pm-pux3j5_Oqt57D2H0YbaJ3tl8vn_eR926scO3xePJoa6uVJa9B4/gitclone.png?format=500w)

- then in the terminal type: `git clone https://github.com/DeepLabCut/DeepLabCut.git`

**Step 2:**

- Now you will work from the terminal inside this cloned folder:

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1581985288123-V8XUAY0C0ZDNJ5WBHB7Y/ke17ZwdGBToddI8pDm48kIsGBOdR9tS_SxF6KQXIcDtZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpz3c8X74DzCy4P3pv-ZANOdh-3ZL9iVkcryTbbTskaGvEc42UcRKU-PHxLXKM6ZekE/terminal.png?format=750w)

- Now, when you start `ipython` and `import deeplabcut` you are importing the folder "deeplabcut" - so any changes you make, or any changes we made before adding it to the pip package, are here.

- You can also check which deeplabcut you are importing by running: `deeplabcut.__file__`

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1581985466026-94OCSZJ5TL8U52JLB5VU/ke17ZwdGBToddI8pDm48kNdOD5iqmBzHwUaWGKS6qHBZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpyQPoegsR7K4odW9xcCi1MIHmvHh95_BFXYdKinJaRhV61R4G3qaUq94yWmtQgdj1A/importlocal.png?format=750w)

If you make changes to the code/first use the code, be sure you run `./resinstall.sh`, which you find in the main DeepLabCut folder:

![](https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1609353210708-FRNREI7HUNS4GLDSJ00G/ke17ZwdGBToddI8pDm48kAya1IcSd32bok4WHvykeicUqsxRUqqbr1mOJYKfIPR7LoDQ9mXPOjoJoqy81S2I8N_N4V1vUb5AoIIIbLZhVYy7Mythp_T-mtop-vrsUOmeInPi9iDjx9w8K4ZfjXt2dq18t0tDkB2HMfL2JGcLHN27k5rSOPIU8nEAZT0p1MiSCjLISwBs8eEdxAxTptZAUg/Screen+Shot+2020-12-30+at+7.33.16+PM.png?format=2500w)



Note, before committing to DeepLabCut, please be sure your code is formatted according to `black`. To learn more,
see [`black`'s documentation](https://black.readthedocs.io/en/stable/).

Now, please make a [pull request](https://github.com/DeepLabCut/DeepLabCut/pull/new/) that includes both a **summary of and changes to**:

- How you modified the code and what new functionality it has.
- DOCSTRING update for your change
- A working example of how it works for users. 
- If it's a function that also can be used in downstream steps (i.e. could be plotted) we ask you (1) highlight this, and (2) ideally you provide that functionality as well. If you have any questions, please reach out: admin@deeplabcut.org 

**TestScript outputs:**

- The **OS it has been tested on**
- the **output of the [testscript.py](/examples/testscript.py)** and if you are editing the **3D code the [testscript_3d.py](/examples/testscript_3d.py)**, and if you edit multi-animal code please run the [maDLC test script](https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/testscript_multianimal.py).

**Review & Formatting:**

- Please run black on the code to conform to our Black code style (see more at https://pypi.org/project/black/). 
- Please assign a reviewer, typically @AlexEMG, @mmathislab, or @jeylau (i/e. the [core-developers](https://github.com/orgs/DeepLabCut/teams/core-developers/members))

**Code headers**

- The code headers can be standardized by running `python tools/update_license_headers.py`
- Edit `NOTICE.yml` to update the header. 

**DeepLabCut is an open-source tool and has benefited from suggestions and edits by many individuals:**

- the [authors](/AUTHORS)
- [code contributors](https://github.com/DeepLabCut/DeepLabCut/graphs/contributors) 

