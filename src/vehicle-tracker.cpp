/*
 * Copyright (C) 2019 Alfonso Sanchez-Beato
 */
#include <memory>
#include <vector>

#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "util.h"
#include "mono-processor.hpp"

static const double detect_thresold_g = 0.5;

using namespace cv;
using namespace std;

struct DetectedObject {
    Rect2d bbox;
    const char *description;
};

// Detects vehicles on video frames
struct VehicleDetector : MonoProcessor<Mat, vector<DetectedObject>> {
    // threshold: [0-1] min confidence to consider something has been detected
    VehicleDetector(double threshold);
    ~VehicleDetector() { stop(); }

private:
    ENUM_WITH_STRINGS(ClassLabel,
                      (clBackground)(clAeroplane)(clBicycle)
                      (clBird)(clBoat)(clBottle)(clBus)(clCar)
                      (clCat)(clChair)(clCow)(clDiningtable)
                      (clDog)(clHorse)(clMotorbike)(clPerson)
                      (clPottedplant)(clSheep)(clSofa)
                      (clTrain)(clTvmonitor))

    double threshold_;
    dnn::Net net_;
    Mat frameBlob_;
    int frameWidth_, frameHeight_;
    vector<DetectedObject> detected_;

    void transactSafe(const Mat& in, vector<DetectedObject>& out);
    void process(void);
};

VehicleDetector::VehicleDetector(double threshold) : threshold_(threshold)
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

void VehicleDetector::transactSafe(const Mat& in, vector<DetectedObject>& out)
{
    // We substract 127,5 and apply a scaling factor of 1/127.5 = 0.007843 so
    // the image values are in the [-1,1] range. Also, the image size must be
    // 300x300 (this is a requirement for the method input data). This is needed
    // as the same transformation is applied when training, apparently (see
    // https://github.com/chuanqi305/MobileNet-SSD/blob/master/template/
    // MobileNetSSD_train_template.prototxt).
    frameBlob_ = dnn::blobFromImage(in, 0.007843, Size{300, 300}, 127.5);
    frameWidth_ = in.size[1];
    frameHeight_ = in.size[0];

    out = detected_;
}

void VehicleDetector::process(void)
{
    net_.setInput(frameBlob_);

    // DetectionOutput layer produces one tensor with seven numbers for each
    // actual detection. Documentation from caffe: Each row is a 7 dimension
    // vector, which stores [image_id, label, confidence, xmin, ymin, xmax,
    // ymax] The returned matrix is 4 dimensional. The size of the first two is
    // always one, the size of the third is the number of possibly detected
    // objects, the four is 7 (the size of DetectionOutput).
    // The coordinates are in the [0,1] range.
    Mat detections = net_.forward();
    vector<DetectedObject> detected;
    for (int i = 0; i < detections.size[2]; ++i) {
        enum ClassLabel label = static_cast<enum ClassLabel>(
            detections.at<float>(Vec<int, 4>{0, 0, i, 1}));
        if (   label != clBicycle && label != clBus
            && label != clCar && label != clMotorbike)
            continue;
        float confidence = detections.at<float>(Vec<int, 4>{0, 0, i, 2});
        if (confidence < threshold_)
            continue;

        BOOST_LOG_TRIVIAL(debug) << "detected " << ClassLabelStr(label)
                                 << " with confidence " << confidence;

        Rect2d bbox;
        bbox.x = frameWidth_*detections.at<float>(Vec<int, 4>{0, 0, i, 3});
        bbox.y = frameHeight_*detections.at<float>(Vec<int, 4>{0, 0, i, 4});
        double x_max, y_max;
        x_max = frameWidth_*detections.at<float>(Vec<int, 4>{0, 0, i, 5});
        y_max = frameHeight_*detections.at<float>(Vec<int, 4>{0, 0, i, 6});
        bbox.width = x_max - bbox.x;
        bbox.height = y_max - bbox.y;

        detected.push_back(DetectedObject{bbox, ClassLabelStr(label)});
    }

    detected_ = detected;
}

struct TrackingState {
    Rect2d bbox;
    bool tracking{false};
};

// Detects and tracks vehicles in video data, using a separate thread
struct TrackThread : MonoProcessor<Mat, TrackingState> {
    TrackThread(const Mat& frame, const Rect2d& bbox);
    ~TrackThread(void) { stop(); }

private:
    Mat frame_;
    bool tracking_;
    Rect2d bbox_;
    Ptr<Tracker> tracker_;

    void transactSafe(const Mat& in, TrackingState& out);
    void process(void);
};

TrackThread::TrackThread(const Mat& frame, const Rect2d& bbox) :
    tracking_{true},
    bbox_{bbox}
{
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
    //tracker_ = TrackerMedianFlow::create();
    tracker_ = TrackerMOSSE::create();
    tracker_->init(frame, bbox);
}

void TrackThread::process(void)
{
    if (tracking_)
        tracking_ = tracker_->update(frame_, bbox_);

    if (!tracking_)
        BOOST_LOG_TRIVIAL(debug) << "Tracking lost";
}

void TrackThread::transactSafe(const Mat& in, TrackingState& out)
{
    out.tracking = tracking_;
    out.bbox = bbox_;
    // TODO make this more efficient
    in.copyTo(frame_);

    // static const double scale_f = 2.;

    // // Take latest track result
    // out.tracking = tracking_;
    // out.bbox = Rect2d(scale_f*bbox_.x,
    //                   scale_f*bbox_.y,
    //                   scale_f*bbox_.width,
    //                   scale_f*bbox_.height);
    // // Makes a copy to the shared frame
    // resize(in, frame_, Size(), 1/scale_f, 1/scale_f);
}

int main(int argc, char **argv)
{
    static const char *windowTitle = "Tracking";
    int wait_ms = 1;

    // Set logging priority
    boost::log::core::get()->set_filter(
        boost::log::trivial::severity >= boost::log::trivial::debug);

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
        double fps = video.get(CAP_PROP_FPS);
        wait_ms = 1000/fps;
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

    VehicleDetector detect{detect_thresold_g};
    unique_ptr<TrackThread> tt;
    Mat in;
    TrackingState out;
    int numFrames = 0, numNotProcDetect = 0, numNotProcTrack = 0;

    while (video.read(in))
    {
        ++numFrames;

        vector<DetectedObject> detected;
        // TODO Continuous processing for the moment
        if (!detect.transact(in, detected))
            ++numNotProcDetect;
        if (!out.tracking && detected.size() > 0) {
            out.tracking = true;
            out.bbox = detected[0].bbox;
            tt = make_unique<TrackThread>(in, out.bbox);
        }

        if (tt && !tt->transact(in, out))
            ++numNotProcTrack;

        if (out.tracking)
            rectangle(in, out.bbox, Scalar(255, 0, 0), 8, 1);

        imshow(windowTitle, in);

        // Exit if ESC pressed
        if (waitKey(wait_ms) == 27)
            break;
    }

    BOOST_LOG_TRIVIAL(debug) << "Detection: " << numNotProcDetect << " frames ("
                             << 100*numNotProcDetect/numFrames
                             << "%) not processed";
    BOOST_LOG_TRIVIAL(debug) << "Tracking: " << numNotProcTrack << " frames ("
                             << 100*numNotProcTrack/numFrames
                             << "%) not processed";
}
