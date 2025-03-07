### QUICK GUIDE to single Animal Training:
**The main steps to take you from project creation to analyzed videos:**

Open ipython in the terminal:
```
ipython
```

Import DeepLabCut:
```
import deeplabcut
```

Create a new project:
```
deeplabcut.create_new_project("project_name", "experimenter", ["path of video 1", "path of video2", ..])
```
    
Set a config_path variable for ease of use + go edit this file!:
```
config_path = "yourdirectory/project_name/config.yaml"
```
        
Extract frames:
```
deeplabcut.extract_frames(config_path)
```

Label frames:
``` 
deeplabcut.label_frames(config_path)
```
  
Check labels [OPTIONAL]:
```
deeplabcut.check_labels(config_path)
```
   
Create training dataset:
```
deeplabcut.create_training_dataset(config_path)
```
 
Train the network:
```
deeplabcut.train_network(config_path)
```

Evaluate the trained network:
```
deeplabcut.evaluate_network(config_path)
```

 Video analysis:
```
deeplabcut.analyze_videos(config_path, ["path of video 1", "path of video2", ..])
```

Filter predictions [OPTIONAL]:
```
deeplabcut.filterpredictions(config_path, ["path of video 1", "path of video2", ..])
```

Plot results (trajectories):
```
deeplabcut.plot_trajectories(config_path, ["path of video 1", "path of video2", ..], filtered=True)
```

Create a video:
```
deeplabcut.create_labeled_video(config_path, ["path of video 1", "path of video2", ..], filtered=True)
```
