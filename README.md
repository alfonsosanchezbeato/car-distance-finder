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

## Installing the snap

On a system with support for snaps, install and connect needed plugs with:

    snap install car-distance-finder
    snap connect car-distance-finder:camera

You can also build the snap locally:

    snap install --classic snapcraft
    snapcraft

and install with

    snap install --dangerous car-distance-finder_*.snap
    snap connect car-distance-finder:camera

For development purposes, you can build using cmake. You will need
OpenCV and Boost libraries in your system. If on Ubuntu, install
dependencies with:

    apt install cmake libboost-log-dev libboost-program-options-dev libopencv-dev

Build with

    (mkdir build; cd build; cmake ..; make)

## Running the tracker

To run using a system camera (`/dev/video<N>`):

    car-distance-finder.vehicletracker --device <N>

To run on a video (any video format supported by OpenCV should work):

    car-distance-finder.vehicletracker --video <video_file>

If you built with cmake, use `build/src/vehicletracker` instead of
`car-distance-finder.vehicletracker`.

Press ESC to exit the program while running.

The tracker, in action:

![The tracker, in action](media/run-capture1.png?raw=true)

## Testing

Some times we would like to test a video fragment. For instance, to get
one minute of video starting at 3:00:

    ffmpeg -ss 00:03:00 -i <input_video> -t 00:01:00 -vcodec copy -acodec copy <output_video>
