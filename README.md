# PyQT-Based DLC GUI

This is a re-implementation of the DeepLabCut WxPython-based GUI in PySide2. Various architecture design changes and UX improvements are documented [here](DesignDocumentation.md).
## Installation and running

Additional dependencies:
```bash
pip install PySide2
pip install qdarkstyle #will get rid of this
```

To run:
```bash
python main.py
```

## Missing components:
- Create training dataset
  - Model comparison
- Missing labeling code
  - multi-animal labeling GUI code, as well as 
  - outlier and tracklet refinement GUI code
- GUI cropping in ExtractFrames
