# ClipSpeechSegmenter

## Introduction

I started this project because, as a [clipper](https://www.youtube.com/channel/UCAUVpVeks_uHlE3J7w-i_6A) (link goes to my YouTube channel), I felt that there should be an easier way to set up timecodes for subtitles. Through my own clipping activities, I have access to about 50 videos' worth of manually labelled data, namely, who's speaking when and their start/end times. I also have what they're saying, but this is beyond the scope of this project.

The approach I will take is a neural network. The training data will be audio features extracted using the `librosa` Python library. I think that I will try implementing the neural network in Python first, and then Julia.

## Resources

librosa features:
<https://librosa.org/doc/main/feature.html>

a similar project for inspiration:
<https://github.com/jurgenarias/Portfolio/blob/master/Voice%20Classification/Code/Speaker_Classifier/Voice_Speaker_Classifier_99.8%25.ipynb>

`audiofile` documentation
<https://audeering.github.io/audiofile/usage.html#write-a-file>
