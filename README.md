# ClippingTools

## Introduction

I started this project because, as a [clipper](https://www.youtube.com/channel/UCAUVpVeks_uHlE3J7w-i_6A) (link goes to my YouTube channel), I feel like there are parts of clipping that can be automated, and I'd like to try and cut away as much of this part of the work as I can. 

I'm currently working on the following Python scripts:

* Removing silence from videos (i.e. creating 'jump cuts' very easily). This is fairly complete. The only downside is that I use a Julia script to extract and process audio, since doing so in Numpy would be cumbersome. 
* Cross-fading a pre-determined segment of the video at the beginning as a preview. This is kind of working, but also slow, so it needs more work.
* Automatically overlaying profile icons on top of a video at times indicated in an `.ass` (Aegisub) file. 

At the core of these scripts are two other technologies:

* `ffmpeg`, which is the bread and butter for all audio, image, and video manipulation, input, and output. 
* `Aegisub`, which is how subtitles are written and formatted. 

I'm not sure what will come in the future, but I initially planned on training some form of voice classifier, but it seems a lot more difficult or impractical than I initially imagined it would be. Secondly, I'd like to finish the `CaptionDataScraper` sub-project I have in this repository, which is planned to basically provide a simple text template for video captions. Finally, I'd like to someday package these scripts into an easy web or desktop app, but that is probably many months in the future.