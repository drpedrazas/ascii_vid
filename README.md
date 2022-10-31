# ascii_vid
## The script you never knew you wanted:

Converting a normal image into ascii art is one of the first projects a would be programmer takles. As charming as it is, I always found it a little limiting. If images, why not video? This program converts video to ascii. It can be executed with a plethora of options so that you determine how detailed of fluent the final product is. 

## How to use it:

1. Downlaod this repo
2. Put the video you want to convert in the same directory as ascii.py
3. Type python3 ascii.py -h to see your options.
```
python3 ascii.py
usage: ascii.py [-h] [-cols COLS] [--video] [-fps FPS] file_name name

Converting images and video into ascii art.

positional arguments:
  file_name   Image or video you want to convert
  name        Name of your project

optional arguments:
  -h, --help  show this help message and exit
  -cols COLS  Number of columns in your ascii image
  --video     Is your project a video?
  -fps FPS    fps for your video
```
4. After deciding what you want to do, run your command. If your project is a video, it will be droped inside an "out" directory created automatically in the current working directory. Otherwise, you will find your image in the "ascii_images" folder, again, created in the current working directory.
