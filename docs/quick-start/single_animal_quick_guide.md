---
deeplabcut:
  last_content_updated: '2025-06-30'
  last_metadata_updated: '2026-03-06'
  ignore: false
  visibility: online
  status: viable
  recommendation: archive
  notes: "This is a bit stuck between minimal guide and quick start, as the lack of explanations makes it more into a catalogue of commands (which is an API docs responsibility), and a proper quick start guide that gives users a proper sense of the workflow. This should either be expanded greatly or simply archived. For simplicity, I recommend archiving."
---
# QUICK GUIDE to single Animal Training:
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
