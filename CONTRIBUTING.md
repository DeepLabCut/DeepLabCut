# How to Contribute to DeepLabCut

DeepLabCut is an actively developed package and we welcome community development and involvement. We are especially seeking people from underrepresented backgrounds in OSS to contribute their expertise and experience. Please get in touch if you want to discuss specific contributions you are interested in developing, and we can help shape a road-map.

We are happy to receive code extensions, bug fixes, documentation updates, etc.

**Setting up a development installation:**

Please see our guide here: https://github.com/DeepLabCut/DeepLabCut/wiki/How-to-use-the-latest-GitHub-code

If you want to contribute to the code, please make a [pull request](https://github.com/DeepLabCut/DeepLabCut/pull/new/) that includes both a **summary of and changes to**:

- How you modified the code and what new functionality it has.
- DOCSTRING update for your change
- A working example of how it works for users. 
- If it's a function that also can be used in downstream steps (i.e. could be plotted) we ask you (1) highlight this, and (2) idealy you provide that functionality as well. If you have any questions, please reach out: admin@deeplabcut.org 

**TestScript outputs:**

- The **OS it has been tested on**
- the **output of the [testscript.py](/examples/testscript.py)** and if you are editing the **3D code the [testscript_3d.py](/examples/testscript_3d.py)**, and if you edit multi-animal code please run the [maDLC test script](https://github.com/DeepLabCut/DeepLabCut/blob/master/examples/testscript_multianimal.py).

**Review & Formatting:**

- Please run black on the code to conform to our Black code style (see more at https://pypi.org/project/black/). 
- Please assign a reviewer, typically @AlexEMG, @mmathislab, or @jeylau (i/e. the [core-developers](https://github.com/orgs/DeepLabCut/teams/core-developers/members))


**DeepLabCut is an open-source tool and has benefited from suggestions and edits by many individuals:**

- the [authors](/AUTHORS)
- [code contributors](https://github.com/DeepLabCut/DeepLabCut/graphs/contributors) 

