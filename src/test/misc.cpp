#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

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

struct Rectangle {
    double x, y;
    double width, height;
};

bool squareOverlap(const Rectangle& a, const Rectangle& b)
{
    double overlap =
        segmentOverlap(a.x, a.x + a.width, b.x, b.x + b.width)*
        segmentOverlap(a.y, a.y + a.height, b.y, b.y + b.height);
    cout << "Overlap is " << overlap << " pixels\n";
    cout << overlap/(a.width*a.height) << '\n';
    cout << overlap/(b.width*b.height) << '\n';
    if (   overlap/(a.width*a.height) > 0.5
        || overlap/(b.width*b.height) > 0.5)
        return true;

    return false;
}

// This functions returns a 3D histogram (the bins are cubes of R,G,B ranges)
// of a ROI of a frame.
Mat calcNormalizedHist3d(const Mat& frame, const Rect2d& bbox)
{
    // Check that the bbox is not out of the frame
    if (   segmentOverlap(bbox.x, bbox.x + bbox.width, 0, frame.size[1]) == 0.
        || segmentOverlap(bbox.y, bbox.y + bbox.height, 0, frame.size[0]) == 0.)
        return Mat{};

    double x, y, width, height;
    x = bbox.x < 0 ? 0 : bbox.x;
    y = bbox.y < 0 ? 0 : bbox.y;
    width = bbox.x + bbox.width > frame.size[1] ?
                 frame.size[1] - bbox.x : bbox.width;
    height = bbox.y + bbox.height > frame.size[0] ?
                 frame.size[0] - bbox.y : bbox.height;

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

    cout << "\nROI dims: " << roi.size << " Type: " << roi.type() << '\n';
    cout << "\nHist dims: " << hist.size << " Type: " << hist.type() << '\n';
    for (int i = 0; i < binSide; ++i) {
        for (int j = 0; j < binSide; ++j)
            cout << hist.at<float>(i, j, 4) << ' ';
        cout << '\n';
    }

    cout << "\nHist dims: " << hist.size << " Type: " << hist.type() << '\n';
    cout << "Num elem: " << sum(hist) << '\n';

    return hist;
}

// https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
void calcNormalizedHist(const Mat& frame, const Rect2d& bbox)
{
    Mat roi{frame, bbox};

    // Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split(roi, bgr_planes);
    // Establish the number of bins
    int histSize = 256;
    // Set the ranges (for B,G,R)
    float range[] = {0, 256} ;
    const float* histRange = {range};

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);


    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
            Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
            Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
            Scalar( 0, 0, 255), 2, 8, 0  );
    }

    /// Display
    namedWindow("calcHist Demo", WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );

    waitKey(0);
}

int main(void)
{
    cout << segmentOverlap(0.5, 1, 2, 3) << '\n';
    cout << segmentOverlap(0.5, 2.5, 2, 3) << '\n';
    cout << segmentOverlap(0.5, 3, 2, 3) << '\n';
    cout << segmentOverlap(0.5, 3.5, 2, 3) << '\n';
    cout << segmentOverlap(2.5, 2.75, 2, 3) << '\n';
    cout << segmentOverlap(2.5, 3.5, 2, 3) << '\n';
    cout << segmentOverlap(3, 4, 2, 3) << '\n';
    // 3
    cout << squareOverlap({0, 0, 3, 3}, {0, 0, 3, 1}) << '\n';
    // 1
    cout << squareOverlap({0, 0, 3, 3}, {2, 2, 3, 1}) << '\n';

    Mat img = imread("photo_2019-02-05_18-01-59.jpg");
    Mat img2 = imread("photo_2019-06-12_12-48-02.jpg");
    //Mat img2 = imread("niryo_one_keyshot_02.png");
    auto imgSz = img.size();
    cout << imgSz << '\n';
    cout << img.size << '\n';
    cout << img.size[1] << '\n';
    cout << img.size[0] << '\n';

    //calcNormalizedHist(img, Rect2d{0, 0, (double) img.size[1], (double) img.size[0]});

    Mat h1 = calcNormalizedHist3d(img, Rect2d{0, 0, (double) img.size[1],
                (double) img.size[0]});
    Mat h2 = calcNormalizedHist3d(img2, Rect2d{0, 0, (double) 2*img2.size[1],
                (double) img2.size[0]/2});
    if (h2.size[0] == 0 || h2.size[1] == 0) {
        cout << "XXXXXX\n";
        return 0;
    }
    Mat ad;
    //absdiff(h1, h2, ad); ad /= 2;
    // min gives us the number of elements that are "equal"
    min(h1, h2, ad);
    cout << "Color similarity: " << sum(ad)[0] << '\n';
}
