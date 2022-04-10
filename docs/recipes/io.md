# Input/output manipulations with DeepLabCut

## Analyzing very large videos in chunks
Analyzing hour-long videos may take a while, but the task can be
conveniently broken down into the analysis of smaller video clips:

```python
import deeplabcut
import os
from deeplabcut.utils.auxfun_videos import VideoWriter

_, ext = os.path.splitext(video_path)
vid = VideoWriter(video_path)
clips = vid.split(n_splits=10)
deeplabcut.analyze_videos(config_path, clips, ext)
```

## Tips on video re-encoding and preprocessing 

While moving videos between computers or from your computer to cloud storage you can encounter issues with `analyze_videos` or `create_labeled_video` due to video corruption. 
The issue can present itself during those steps and you have to carefully review the traceback. Sometimes it might look like the videos were analyzed but in fact analysis stopped right before the end of the video (corruption of the metadata when more indices are assigned than there are actual frames in a video). 
To tackle this issue, the easiest solution might be to re-encode the video, this will not only help with corruption but can also – if you choose so – compress the video without perceivable loss of quality. Common package used for video processing is FFmpeg which you can use from the terminal inside your DEEPLABCUT environment (without going into iPython).
There are number of video codecs that can be used to re-encode your video and if you want to keep the video in the same container (`.avi`, `.mp4`, `.ts` etc.) you should check which codec allows encoding to a certain container. For instance, for `.avi` it will be MJPEG and for `.mp4` H264 and H265. 
To re-encode your video, simply use:
```
ffmpeg -i "path_to_video" -c:v codec_name "output_path"
```
For instance, to re-encode to `.mp4` format with compression use:
```
ffmpeg -i "path_to_video" -c:v h264 -crf 18 -preset fast "output_path"
```
`-crf` is a quality-size tradeoff from 0 to 63 with 0 being highest quality but lowest compression. Ideally you’d want to use values between 18-23 for similar visual quality to the original.
`-preset` is a quality-speed tradeoff. Higher values give you faster encoding but will result in bigger filesizes and/or worse quality.
For `.avi` files you want to change the codec and the quality metric, since `crf` is used by H264/265 and not MJPEG. For instance, the encoding with some compression would be:
```
ffmpeg -i "path_to_video" -c:v mjpeg -q:v 10 "output_path"
```
`-q:v` is a quality metric with values ranging from 1 to 31 with reasonable values being around 10. 
If you want to compress all your recordings for easier storage or moving to cloud storage, you can use a for loop that will go through all videos in a directory that are in a certain container. Let’s say we want to transcode our `.avi` videos to `.mp4` and make them smaller without quality loss. Note, that the loop has be run from inside the folder the videos are in:
```
for %i in (*.avi) do ffmpeg -i "%i" -c:v libx265 -preset fast -crf 18 "%~ni.mp4" 
```
This command will re-encode all of your videos into an `.mp4` container and save them with the same name as the original (without overwriting them).
Additionally, ffmpeg allows you to also crop or rescale the videos for possible improvement in inference speed further down the line in DLC workflow. To either crop or rescale you need to use 
`-filter:v` parameter after which you’d add either `"crop=Xsize:Ysize:Xstart:Ystart"` for cropping or 
`"scale=Xsize:Ysize"` for rescale. Note that when using “scale” the values how be a result of integer division of the original video size. If you want to keep the aspect ratio, you can simply set either X or Y to `-1` and only give one of the  or you can use `“scale=iw/2:ih/2”` which will simply make the video 2 times smaller in both dimensions. For instance, if you have a videos at 1920x1080 resolution and want to rescale it to 960x540 for faster inference while also reencoding from `.avi` and doing some compression in a loop, the command would be something like this:
```
for %i in (*.avi) do ffmpeg -i "%i" -c:v libx265 -preset fast -crf 18 -filter:v "scale= iw/2:ih/2" "%~ni.mp4"
```
If audio is not a necessary in the videos you can also save some space by requesting specifically for the encoder to not encode any audio stream by adding `-an` just before specifying output filename, like so:
```
for %i in (*.avi) do ffmpeg -i "%i" -c:v libx265 -preset fast -crf 18 -filter:v "scale= iw/2:ih/2" -an "%~ni.mp4"
```
