#ifndef _LEGO_FEATURE_H_
#define _LEGO_FEATURE_H_

#include <opencv2/opencv.hpp>

constexpr float segAlphaX = 0.2 / 180.0 * M_PI;
constexpr float segAlphaY = 2 / 180.0 * M_PI;
constexpr float segmentTheta = 30.0 / 180.0 * M_PI;
constexpr int LeastSegPoint = 30;
constexpr int LeastSegLine = 3;

cv::Mat1b legoFeature(const cv::Mat3f &imagery, const cv::Mat1b &mask, const cv::Mat1b &valid, float max_range);

#endif
