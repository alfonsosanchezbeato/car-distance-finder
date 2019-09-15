/*
 * Copyright (C) 2019 Alfonso Sanchez-Beato
 */
#include <chrono>
#include <list>
#include <memory>
#include <vector>

#include <boost/program_options.hpp>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "util.h"
#include "mono-processor.hpp"

static const double detect_thresold_g = 0.5;
static const double allowed_overlap_g = 0.5;
static const float min_similarity_hist_g = 0.5;

using namespace cv;
using namespace std;

namespace po = boost::program_options;
static constexpr char optHelp[] = "help";
static constexpr char optDevice[] = "device";
static constexpr char optVideo[] = "video";
static constexpr char optDebug[] = "debug";

inline double segmentOverlap(double a1, double a2, double b1, double b2);

struct DetectedObject {
    Rect2d bbox;
    const char *description;
};

// Detects vehicles on video frames
struct DetectTask {
    // threshold: [0-1] min confidence to consider something has been detected
    DetectTask(double threshold);
    ~DetectTask() {}

    void transactSafe(const Mat& in, vector<DetectedObject>& out);
    void process(void);

private:
    ENUM_WITH_STRINGS(ClassLabel,
                      (clBackground)(clAeroplane)(clBicycle)
                      (clBird)(clBoat)(clBottle)(clBus)(clCar)
                      (clCat)(clChair)(clCow)(clDiningtable)
                      (clDog)(clHorse)(clMotorbike)(clPerson)
                      (clPottedplant)(clSheep)(clSofa)
                      (clTrain)(clTvmonitor))

    bool crop_;
    double threshold_;
    dnn::Net net_;
    Mat frameBlob_;
    int frameWidth_, frameHeight_;
    vector<DetectedObject> detected_;
};

DetectTask::DetectTask(double threshold) : crop_{false}, threshold_{threshold}
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

void DetectTask::transactSafe(const Mat& in, vector<DetectedObject>& out)
{
    // We do a sort of alternate multi-scale detection to try to detect sooner
    // far objects.
    Mat inFrame;
    if (crop_) {
        frameWidth_ = in.size[1]/2;
        frameHeight_ = in.size[0]/2;
        inFrame = in(Rect2d(double(frameWidth_/2), double(frameHeight_/2),
                            double(frameWidth_), double(frameHeight_)));
    } else {
        frameWidth_ = in.size[1];
        frameHeight_ = in.size[0];
        inFrame = in;
    }

    // We substract 127,5 and apply a scaling factor of 1/127.5 = 0.007843 so
    // the image values are in the [-1,1] range. Also, the image size must be
    // 300x300 (this is a requirement for the method input data). This is needed
    // as the same transformation is applied when training, apparently (see
    // https://github.com/chuanqi305/MobileNet-SSD/blob/master/template/
    // MobileNetSSD_train_template.prototxt).
    frameBlob_ = dnn::blobFromImage(inFrame, 0.007843, Size{300, 300}, 127.5);

    out = detected_;
}

void DetectTask::process(void)
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

        LOG(debug) << "detected " << ClassLabelStr(label)
                   << " with confidence " << confidence;

        float x_dnn = detections.at<float>(Vec<int, 4>{0, 0, i, 3});
        float y_dnn = detections.at<float>(Vec<int, 4>{0, 0, i, 4});
        float x_max_dnn = detections.at<float>(Vec<int, 4>{0, 0, i, 5});
        float y_max_dnn = detections.at<float>(Vec<int, 4>{0, 0, i, 6});

        // If we are cropping, avoid detections too near to the border as we
        // might crop the vehicle when the frame is displayed.
        if (crop_) {
            static constexpr float perInside = 0.05;
            if (   x_dnn < 0.f + perInside || x_max_dnn > 1.f - perInside
                || y_dnn < 0.f + perInside || y_max_dnn > 1.f - perInside) {
                LOG(debug) << "Dropping cropped detection: "
                           << "too near to the border";
                continue;
            }
        }

        Rect2d bbox;
        bbox.x = frameWidth_*x_dnn;
        bbox.y = frameHeight_*y_dnn;
        bbox.width = frameWidth_*x_max_dnn - bbox.x;
        bbox.height = frameHeight_*y_max_dnn - bbox.y;
        if (crop_) {
            bbox.x += double(frameWidth_/2);
            bbox.y += double(frameHeight_/2);
        }

        detected.push_back(DetectedObject{bbox, ClassLabelStr(label)});
    }

    crop_ = !crop_;
    detected_ = detected;
}

struct TrackingState {
    Rect2d bbox;
    bool tracking{false};
    // Distance to the car in meters
    double distanceMet;
};

// Tracks vehicles in video data. Use with MonoProcessor.
struct TrackTask {
    TrackTask(const Mat& frame, const Rect2d& bbox);
    ~TrackTask(void) {}

    void transactSafe(const Mat& in, TrackingState& out);
    void process(void);

private:
    Mat frame_, frame0_;
    bool tracking_;
    Rect2d bbox_;
    double distanceMet_;
    Ptr<Tracker> tracker_;
    Mat firstHist_;

    void createTracker(void);
    Mat calcNormalizedHist3d(const Mat& frame, const Rect2d& bbox);
    double calcObjectDistance(const Rect2d& bbox);
};

TrackTask::TrackTask(const Mat& frame, const Rect2d& bbox) :
    frame0_{frame.clone()},
    tracking_{true},
    bbox_{bbox},
    distanceMet_{calcObjectDistance(bbox)},
    firstHist_{calcNormalizedHist3d(frame, bbox)}
{
}

// This functions returns a 3D histogram (the bins are cubes of R,G,B ranges)
// of a ROI of a frame.
Mat TrackTask::calcNormalizedHist3d(const Mat& frame, const Rect2d& bbox)
{
    // Return empty matrix if the bbox is out of the frame
    if (   bbox.x > frame.size[1] || bbox.x + bbox.width < 0
        || bbox.y > frame.size[0] || bbox.y + bbox.height < 0)
        return Mat{};

    double x, y, width, height;
    if (bbox.x < 0) {
        x = 0;
        width = bbox.width + bbox.x;
    } else {
        x = bbox.x;
        width = bbox.width;
    }
    if (bbox.y < 0) {
        y = 0;
        height = bbox.height + bbox.y;
    } else {
        y = bbox.y;
        height = bbox.height;
    }
    width = x + width > frame.size[1] ? frame.size[1] - x : width;
    height = y + height > frame.size[0] ? frame.size[0] - y : height;

    Mat roi{frame, Rect2d{x, y, width, height}};
    // Take channels 0, 1, and 2 from frame
    const int channels[] = {0, 1, 2};
    Mat hist;
    // 3-dimensional bins
    constexpr int binSide = 8;
    const int histSizes[] = {binSide, binSide, binSide};
    int dims = sizeof histSizes/sizeof histSizes[0];
    // 256/8 = 32 values in each bin
    float rRange[] = {0, 256};
    float gRange[] = {0, 256};
    float bRange[] = {0, 256};
    const float *ranges[] = {rRange, gRange, bRange};

    // The returned histogram is of CV_32F type
    calcHist(&roi, 1, channels, Mat{}, hist, dims, histSizes, ranges);

    int numPix = roi.size[0]*roi.size[1];
    hist /= numPix;

    return hist;
}

void TrackTask::createTracker(void)
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
    //tracker_ = TrackerMOSSE::create();
    tracker_ = TrackerCSRT::create();
    tracker_->init(frame0_, bbox_);
}

double TrackTask::calcObjectDistance(const Rect2d& bbox)
{
    // Focal lenght in pixels, vehicle back length in meters
    static constexpr double focalLength = 300.;
    static constexpr double vehicleBackLen = 2.;

    // Just do a very rough calculation for the moment
    return vehicleBackLen*focalLength/bbox.width;
}

void TrackTask::process(void)
{
    if (!tracker_)
        createTracker();

    if (tracking_)
        tracking_ = tracker_->update(frame_, bbox_);

    // Check that we have not skewed too much from the original target by
    // comparing histograms
    if (tracking_) {
        Mat hist = calcNormalizedHist3d(frame_, bbox_);
        if (hist.size[0] == 0 || hist.size[1] == 0) {
            LOG(warning) << "Tracking but no overlap?!";
            tracking_ = false;
        } else {
            Mat equalPer;
            // Percentage of equal pixels
            min(firstHist_, hist, equalPer);
            float simPer = sum(equalPer)[0];
            if (simPer < min_similarity_hist_g) {
                LOG(debug) << "Hist similarity below expected: " << simPer;
                tracking_ = false;
            }
        }
    }

    if (tracking_)
        distanceMet_ = calcObjectDistance(bbox_);

    if (!tracking_)
        LOG(debug) << "Tracking lost";
}

void TrackTask::transactSafe(const Mat& in, TrackingState& out)
{
    out.tracking = tracking_;
    out.bbox = bbox_;
    out.distanceMet = distanceMet_;
    // TODO make this more efficient by scaling
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

typedef MonoProcessor<Mat, vector<DetectedObject>,
                      DetectTask> DetectTaskProc;
typedef MonoProcessor<Mat, TrackingState, TrackTask> TrackerTaskProc;

struct TrackedObject {
    TrackingState state;
    unique_ptr<TrackerTaskProc> tt;
};

// Returns the overlap between two segments, a1a2 and b1b2. It assumes that
// a1 < a2 and b1 < b2.
inline double segmentOverlap(double a1, double a2, double b1, double b2)
{
    if (a1 <= b1 && a2 > b1) {
        if (a2 <= b2)
            return a2 - b1;
        else
            return b2 - b1;
    } else if (a1 >= b1 && a1 < b2) {
        if (a2 <= b2)
            return a2 - a1;
        else
            return b2 - a1;
    }

    return 0.;
}

bool squareOverlap(const Rect2d& a, const Rect2d& b)
{
    double overlap =
        segmentOverlap(a.x, a.x + a.width, b.x, b.x + b.width)*
        segmentOverlap(a.y, a.y + a.height, b.y, b.y + b.height);
    if (   overlap/(a.width*a.height) > allowed_overlap_g
        || overlap/(b.width*b.height) > allowed_overlap_g)
        return true;

    return false;
}

// We might be already tracking some of the detected objects. To avoid
// duplicated trackers, we look at the overlap between the bounding box of the
// tracker and the one coming from the detector. If the overlap is bigger than
// allowed_overlap_g for either of the boxes, we replace the older tracker with
// the new detection, which is considered more accurate.  If there is no
// overlap, we create a new tracker for it. The replaced trackers are appended
// to the garbage list.
void mergeTrackedObjects(const Mat& frame,
                         const vector<DetectedObject>& detected,
                         list<TrackedObject>& tracked,
                         list<TrackedObject>& garbage)
{
    list<TrackedObject> newTracked;
    for (const auto& detect : detected) {
        bool overlaps = false;
        for (auto& track : tracked) {
            if (squareOverlap(detect.bbox, track.state.bbox)) {
                overlaps = true;
                // We replace the current tracker with a new one that uses the
                // new bounding box. In principle, this should be more accurate
                // than latest bbox from the tracker, which might have skewed.
                LOG(debug) << "tracker replaced by newer detection";
                garbage.push_back(move(track));
                track = TrackedObject{{detect.bbox, true, 0.},
                                      make_unique<TrackerTaskProc>(
                                          TrackTask(frame, detect.bbox))};
                break;
            }
        }
        if (overlaps == false)
            newTracked.push_back(TrackedObject{{detect.bbox, true, 0.},
                        make_unique<TrackerTaskProc>(
                            TrackTask(frame, detect.bbox))});
    }

    tracked.splice(tracked.end(), newTracked);
}

// Occassionally two objects end up overlapping, track only one
void mergeOverlappingObjects(list<TrackedObject>& tracked,
                             list<TrackedObject>& garbage)
{
    for (auto tr = tracked.begin(), trEnd = tracked.end(); tr != trEnd; ++tr) {
        for (auto ot = next(tr); ot != trEnd; ) {
            if (squareOverlap(tr->state.bbox, ot->state.bbox)) {
                LOG(debug) << "Removing overlapped object";
                garbage.push_back(move(*ot));
                ot = tracked.erase(ot);
            } else {
                ++ot;
            }
        }
    }
}

void processStream(VideoCapture& video, chrono::steady_clock::duration period)
{
    static const char *windowTitle = "Tracking";
    DetectTaskProc detect{DetectTask(detect_thresold_g)};
    list<TrackedObject> tracked, garbage;
    Mat in;
    int numFrames = 0, numNotProcDetect = 0, numNotProcTrack = 0;
    chrono::steady_clock::duration waitDur = chrono::seconds(0);

    namedWindow(windowTitle, WINDOW_NORMAL);
    // To fill the display, uncomment the next two lines. For some weird reason,
    // resizing to a random size not related to the actual screen resolution is
    // needed in order to show the window in full screen mode.
    //resizeWindow(windowTitle, 960, 720);
    //setWindowProperty(windowTitle, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

    auto prev = chrono::steady_clock::now();
    while (video.read(in))
    {
        ++numFrames;

        // TODO Continuous detection for the moment
        vector<DetectedObject> detected;
        if (!detect.transact(in, detected))
            ++numNotProcDetect;
        TRACE_CODE(debug,
                   for (auto& res : detected)
                       rectangle(in, res.bbox, Scalar(0, 255, 0), 8, 1);)
        if (detected.size() > 0)
            mergeTrackedObjects(in, detected, tracked, garbage);

        for (auto tr = tracked.begin(), trEnd = tracked.end(); tr != trEnd; ) {
            tr->tt->transact(in, tr->state);
            if (tr->state.tracking) {
                rectangle(in, tr->state.bbox, Scalar(255, 0, 0), 8, 1);
                ostringstream tag;
                tag.precision(2);
                tag << tr->state.distanceMet << " m";
                putText(in, tag.str(),
                        Point{int(tr->state.bbox.x) - 20,
                                int(tr->state.bbox.y) - 20},
                        FONT_HERSHEY_SCRIPT_SIMPLEX, 1., Scalar(255, 0, 0), 3);
                ++tr;
            } else {
                garbage.push_back(move(*tr));
                tr = tracked.erase(tr);
            }
        }

        mergeOverlappingObjects(tracked, garbage);

        // Now do "garbage collection" of tracks. We do this so we do not get
        // stuck in the destructor, which needs to take the object lock.
        for (auto tr = garbage.begin(), trEnd = garbage.end(); tr != trEnd; ) {
            if (tr->tt->processorIdle())
                tr = tracked.erase(tr);
            else
                ++tr;
        }

        imshow(windowTitle, in);

        // Sleep for the time left until next frame should be displayed
        auto now = chrono::steady_clock::now();
        auto processDur = now - prev - waitDur;
        // We need waitDur >= 1, that's why we check 'process + 1 > period'
        waitDur = processDur + chrono::milliseconds(1) > period ?
            chrono::milliseconds(1) : period - processDur;
        prev = now;
        LOG(trace)
            << chrono::duration_cast<chrono::milliseconds>(period).count()
            << ' '
            << chrono::duration_cast<chrono::milliseconds>(processDur).count()
            << " wait ms: "
            << chrono::duration_cast<chrono::milliseconds>(waitDur).count();
        // Exit if ESC pressed
        if (waitKey(chrono::duration_cast
                    <chrono::milliseconds>(waitDur).count()) == 27)
            break;
    }

    LOG(debug) << "Detection: " << numNotProcDetect << " frames ("
               << 100*numNotProcDetect/numFrames << "%) not processed";
    LOG(debug) << "Tracking: " << numNotProcTrack << " frames ("
               << 100*numNotProcTrack/numFrames << "%) not processed";
}

// Check that 'opt1' and 'opt2' are not specified at the same time
void conflicting_options(const boost::program_options::variables_map& vm,
                         const char* opt1, const char* opt2)
{
    if (vm.count(opt1) && !vm[opt1].defaulted()
        && vm.count(opt2) && !vm[opt2].defaulted())
        throw logic_error(string("Conflicting options '")
                          + opt1 + "' and '" + opt2 + "'.");
}

boost::program_options::variables_map parseArguments(int argc, char **argv)
{
    po::options_description desc("Allowed options");
    desc.add_options()
        (optHelp, "produce help message")
        (optDevice, po::value<int>(),
         "get input from device number N - N in /dev/videoN")
        (optVideo, po::value<string>(), "get input from video file")
        (optDebug, po::value<string>()->default_value("info"),
         "set debug level to one of fatal, error, warning, info, debug, trace");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count(optHelp)) {
        cout << "Usage: " << argv[0] << " [--" << optHelp << " | --"
             << optDevice << " <device number> | --" << optVideo
             << " <video file>] [--" << optDebug << " <debug level>]\n";
        cout << desc << "\n";
        return vm;
    }

    // Read video from either camera of video file
    conflicting_options(vm, optDevice, optVideo);

    return vm;
}

int main(int argc, char **argv)
{
    chrono::steady_clock::duration period = chrono::milliseconds(1);

    // Handle arguments
    po::variables_map vm;
    try {
        vm = parseArguments(argc, argv);
    } catch(exception& e) {
        cerr << e.what() << '\n';
        cerr << "use --" << optHelp << " to see usage\n";
        return 1;
    }
    if (vm.count(optHelp))
        return 0;

    // Set logging priority
    initLog(vm[optDebug].as<string>());

    VideoCapture video;
    if (vm.count(optDevice)) {
        video.open(vm[optDevice].as<int>());
    } else if (vm.count(optVideo)) {
        video.open(vm[optVideo].as<string>());
        double fps = video.get(CAP_PROP_FPS);
        period = chrono::milliseconds(static_cast<int>(1000/fps));
    } else {
        video.open(0);
    }

    if (!video.isOpened()) {
        cout << "Could not open video source\n";
        return 1;
    }

    processStream(video, period);
    return 0;
}
