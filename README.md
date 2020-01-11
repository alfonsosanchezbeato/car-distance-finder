# Vehicle tracker

This repository contains a C++ program that uses a MobileNet SSD network
to detect vehicles in a video stream, and then tracks them using the
CSRT tracker algorithm implemented on OpenCV. The distance from the
camera to the vehicles is then estimated by a simple formula using the
bounding box for the vehicle and the camera's focal distance.  The later
is hard-coded in the code, so do not expect anything accurate: it is
more of an example application.

The program is able to track multiple vehicles while still running the
neural network for detections on the stream. Extensive multi-threading
is used to accomplish this while at the same time trying to be as
real-time as possible. This means that the trackers are fed with the
latest available frames and some times older frames are not processed.

## Compiling

You will need OpenCV and Boost libraries in your system. Build with

    (mkdir build; cd build; cmake ..; make)

## Running the tracker

To run using a system camera (/dev/video<N>):

    build/src/vehicletracker --device <N>

To run on a video (any video format supported by OpenCV should work):

    build/src/vehicletracker --video <video_file> 

## Testing

Some times we would like to test a video fragment. For instance, to get
one minute of video starting at 3:00:

    ffmpeg -ss 00:03:00 -i <input_video> -t 00:01:00 -vcodec copy -acodec copy <output_video>
