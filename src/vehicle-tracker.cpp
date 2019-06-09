/*
 * Copyright (C) 2019 Alfonso Sanchez-Beato
 */
#include <mutex>
#include <thread>
#include <condition_variable>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

// Detects vehicles on video frames
struct VehicleDetector {
    VehicleDetector(void);
    // If a vehicle is detected in frame, returs true and fills bbox with a
    // square containing it.
    bool detectVehicle(const Mat& frame, Rect2d& bbox);

private:
    dnn::Net net_;
};

VehicleDetector::VehicleDetector(void)
{
    string pathData;
    const char *snapDir = getenv("SNAP");

    // We make sure things work well if we are in a snap
    if (snapDir)
        pathData.append(snapDir);
    pathData.append("./data/model/");

    net_ = dnn::readNetFromCaffe(pathData + "MobileNetSSD_deploy.prototxt",
                                 pathData + "MobileNetSSD_deploy.caffemodel");
}

bool VehicleDetector::detectVehicle(const Mat& frame, Rect2d& bbox)
{
    // Detect face using Haar Cascade classifier
    // See http://www.emgu.com/wiki/files/1.5.0.0/Help/html/e2278977-87ea-8fa9-b78e-0e52cfe6406a.htm
    // for flag description. It might be wortwhile to play a bit with the
    // different parameters.
    // vector<Rect> f;
    // faceCascade.detectMultiScale(frame, f, 1.1, 2, CASCADE_SCALE_IMAGE);
    // if (f.size() == 0)
    //     return false;

    // if (debug)
    //     cout << "Detected " << f.size() << " faces \n";

    // // Get only one face for the moment
    // bbox = Rect2d(f[0].x, f[0].y, f[0].width, f[0].height);

    // Input image needs to be 300x300 in this network
    // substract the mean (127.5)
    Mat blob = dnn::blobFromImage(frame, 1., Size(300,300), 127.5);
    net_.setInput(blob);

    // DetectionOutput layer produces one tensor with seven numbers for each
    // actual detection. Documentation from caffe:
    // Each row is a 7 dimension vector, which stores
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    Mat detections = net_.forward();
    cout << "detections =\n"  << detections << '\n';

    return false;
}

// Detects and tracks vehicles in video data, using a separate thread
struct TrackThread {
    TrackThread(void);
    ~TrackThread(void);

    struct Output {
        Rect2d bbox;
        bool tracking{false};
    };

    // If the tracking thread is busy, it does nothing. Otherwise, it pushes a
    // new frame to the thread and refreshes "out" with the new tracking data
    // (saying if we are tracking something and the bounding box in the frame if
    // that is the case).
    void process(const Mat& in, Output& out);

private:
    mutex dataMtx;
    condition_variable frameCondition;
    Mat frame;
    Output output;
    bool finish;
    // Keep this last as it uses the other members
    thread processThread;

    void threadMethod(void);
};

TrackThread::TrackThread(void) :
    finish{false},
    processThread{&TrackThread::threadMethod, this}
{
}

TrackThread::~TrackThread(void)
{
    {
        std::unique_lock<mutex> lock(dataMtx);
        finish = true;
        frameCondition.notify_one();
    }

    processThread.join();
}

void TrackThread::threadMethod(void)
{
    bool tracking = false;
    Rect2d bbox;
    Ptr<Tracker> tracker;
    VehicleDetector detector;

    while (true) {
        std::unique_lock<mutex> lock(dataMtx);
        frameCondition.wait(lock);

        if (finish)
            break;

        if (tracking)
            tracking = tracker->update(frame, bbox);

        if (!tracking) {
            tracking = detector.detectVehicle(frame, bbox);
            if (tracking) {
                // We have different options for the tracker:
                // TrackerBoosting: slow, does not adapt bounding box
                // TrackerMIL: slow, does not adapt bounding box
                // TrackerKCF: does not adapt bounding box
                // TrackerTLD: jumps between random parts of the image
                // TrackerMedianFlow: fast, good tracking, adapts box, can lose tracking
                // TrackerGOTURN: slow, jumps, large caffe model
                // TrackerMOSSE: fastest, does not adapt bounding box, can lose tracking
                // TrackerCSRT: a bit slow, best tracking, adapts box

                // At least for KFC, we need to re-create the tracker when
                // the tracked object changes. It looks like a repeated call
                // to init does not fully clean the state and the
                // performance of the tracker is greatly affected.
                tracker = TrackerMedianFlow::create();
                tracker->init(frame, bbox);
            }
        }

        output.bbox = bbox;
        output.tracking = tracking;

        cout << "Size is " << frame.cols << " x " << frame.rows
             << " . Tracking: " << tracking << '\n';
    }
}

void TrackThread::process(const Mat& in, TrackThread::Output& out)
{
    static const double scale_f = 2.;

    std::unique_lock<mutex> lock(dataMtx, defer_lock_t());
    if (lock.try_lock()) {
        // Take latest track result
        out.tracking = output.tracking;
        out.bbox = Rect2d(scale_f*output.bbox.x,
                          scale_f*output.bbox.y,
                          scale_f*output.bbox.width,
                          scale_f*output.bbox.height);
        // Makes a copy to the shared frame
        resize(in, frame, Size(), 1/scale_f, 1/scale_f);
        frameCondition.notify_one();
    }
}

int main(int argc, char **argv)
{
    static const char *windowTitle = "Tracking";
    // Read video from either camera of video file
    VideoCapture video;
    if (argc == 1) {
        video.open(0);
    } else if (argc == 2) {
        int videoSrc;
        istringstream arg1(argv[1]);
        arg1 >> videoSrc;
        video.open(videoSrc);
    } else if (argc == 3 && strcmp(argv[1], "-f") == 0) {
        video.open(argv[2]);
    } else {
        cout << "Usage: " << argv[0] << " [<dev_number> | -f <video_file>]\n";
        return 1;
    }

    if (!video.isOpened()) {
        cout << "Could not open video source\n";
        return 1;
    }

    namedWindow(windowTitle, WINDOW_NORMAL);
    //moveWindow(windowTitle, 0, 0);
    //resizeWindow(windowTitle, 960, 720);
    //setWindowProperty(windowTitle, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    TrackThread tt;
    Mat in;
    TrackThread::Output out;
    while (video.read(in))
    {
        tt.process(in, out);

        if (out.tracking)
            rectangle(in, out.bbox, Scalar(255, 0, 0), 8, 1);

        imshow(windowTitle, in);

        // Exit if ESC pressed
        if (waitKey(1) == 27)
            break;
    }
}