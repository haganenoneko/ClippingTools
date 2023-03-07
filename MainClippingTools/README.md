## MainClippingTools

This is a very quick write-up about some of the code in this folder. It's not a comprehensive guide, and probably won't be updated.

Link to code (which may not work for you):

> https://github.com/haganenoneko/ClippingTools/tree/main/MainClippingTools

The main ones you might be interested in are: 

1. `remove_silence2.jl` 

This implements two methods to detect and remove silent intervals, and is the main one I use day-to-day. The first method is SVM (support vector machine)-based and fits a probabilistic model to audio data. The second is simply based on signal amplitude. 

The first sounds fancy, but it doesn't work very well. I use the second the most. You can do some basic stuff like setting the following: - minimum threshold for silence (sound quieter than this will be considered 'silent') - minimum duration for silence (intervals shorter than this are ignored, i.e. kept in the file) - padding (i.e. adds a little bit of 'padding' to each cut)
  
2. `concat_video_from_ass.py` 

This is really useful if you're using Aegisub for subtitling. It basically reads your subtitles file (`.ass` extension) and throws out any segment of the video that isn't represented in your `.ass` file For example, if your video is _1 minute_ long and you have subtitles up to _50 seconds_, this script will cut out the last 10 seconds.

This gives you finer control over 'silence,' but also allows you to make jump cuts in the video while you are adding subtitles. I also use it to cut out parts of the video that I don't TL, e.g. small talk or random noise.

3. `overlay.py` 

This probably won't work for you because I wrote it really badly, but it basically adds icons to the video automatically. For example,

![Image](https://media.discordapp.net/attachments/1082763257825198151/1082765282692562954/image.png?width=825&height=408)
   
There are different settings, e.g. you can have the icons stay on-screen the entire time, or use `speaker` mode, where the icons only show up when there's a corresponding subtitle.

It also writes a new `.ass` file that moves the subtitles next to the icon.

4. `crossfade.py` 

This is the last useful script, I think. It just adds transitions between separate video files. You can choose between different video/audio transitions, but I generally just let the script pick random video transitions and use a default audio transition called 'double exponential seat', which makes the audio signal like this:

![Image](https://media.discordapp.net/attachments/1082763257825198151/1082765961024786522/dese.png?width=480&height=270)

Anyway, that's pretty much it. If you have any questions, feel free to let me know. It's been on my to-do list since forever to rewrite this stuff into a more user-friendly app or something, but I just haven't had the time.