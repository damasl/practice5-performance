#include "retro_filter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <time.h>

using namespace std;
using namespace cv;

inline void alphaBlend(const Mat& src, Mat& dst, const Mat& alpha)
{
    Mat w, d;
    alpha.convertTo(w, CV_32S);
    //src.convertTo(s, CV_32S);
    dst.convertTo(d, CV_32S);
    d = (d*255 + src.mul(w)+ d.mul(-w))/255.0;
    d.convertTo(dst, CV_8U);
}


RetroFilter::RetroFilter(const Parameters& params) : rng_(time(0))
{
    params_ = params;

    resize(params_.fuzzyBorder, params_.fuzzyBorder, params_.frameSize);

    if (params_.scratches.rows < params_.frameSize.height ||
        params_.scratches.cols < params_.frameSize.width)
    {
        resize(params_.scratches, params_.scratches, params_.frameSize);
    }

    hsvScale_ = 1;
    hsvOffset_ = 20;
}

void RetroFilter::applyToVideo(const Mat& frame, Mat& retroFrame)
{
    int col, row;
    Mat luminance;
    cvtColor(frame, luminance, CV_BGR2GRAY);
    // Add scratches
    Scalar meanColor = mean(luminance.row(luminance.rows / 2));
    int x = rng_.uniform(0, params_.scratches.cols - luminance.cols);
    int y = rng_.uniform(0, params_.scratches.rows - luminance.rows);
    uchar mcolor = meanColor[0]*2.0;
    for (row = 0; row < luminance.rows; row += 1)
    {
        for (col = 0; col < luminance.cols; col += 1)
        {
            luminance.at<uchar>(row, col) = params_.scratches.at<uchar>(row + y, col + x) ? mcolor : luminance.at<uchar>(row, col);
        }
    }

    // Add fuzzy border
    Mat borderColor(params_.frameSize, CV_32S, Scalar::all(meanColor[0] * 1.5));
    alphaBlend(borderColor, luminance, params_.fuzzyBorder);

    // Apply sepia-effect
    retroFrame.create(luminance.size(), CV_8UC3);
    for (col = 0; col < luminance.cols; col += 1)
    {
        for (row = 0; row < luminance.rows; row += 1)
        {
            retroFrame.at<Vec3b>(row, col)[0] = 19;
            retroFrame.at<Vec3b>(row, col)[1] = 78;
            retroFrame.at<Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(luminance.at<uchar>(row, col) * hsvScale_ + hsvOffset_);
        }
    }
    cvtColor(retroFrame, retroFrame, COLOR_HSV2BGR);
}
