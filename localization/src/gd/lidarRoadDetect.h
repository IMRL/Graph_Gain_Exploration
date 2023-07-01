#ifndef _LIDAR_ROAD_DETECT_H_
#define _LIDAR_ROAD_DETECT_H_

#include "config.h"

#include <vector>
#include <array>
#include <algorithm>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class LidarRoadDetect
{
public:
#if defined(CONF_VLP16) || defined(CONF_VLP64)

    // lidar & block arguments
    static constexpr int Range_Num = 16;
    // static constexpr int Range_Num = 64;
    static constexpr int H_Ang_Num = 360 * 5;
    static constexpr int R_Piece = 3;
    static constexpr int H_Ang_Piece = 10 * H_Ang_Num / 360;
    // static constexpr int H_Ang_Piece = 20 * H_Ang_Num / 360;
    // static constexpr int RANSAC_N = 100;
    static constexpr int RANSAC_N = 20;
    // static constexpr float RANSAC_TH = 0.01;
    // static constexpr float RANSAC_TH = 0.008;
    static constexpr float RANSAC_TH = 0.01;
    // static constexpr float RANSAC_TH = 0.007;

#elif defined(CONF_VLP16_SIMULATOR)

    // simulator
    static constexpr int Range_Num = 16;
    static constexpr int H_Ang_Num = 350;
    static constexpr int R_Piece = 3;
    // static constexpr int H_Ang_Piece = 30 * H_Ang_Num / 360;
    static constexpr int H_Ang_Piece = 10 * H_Ang_Num / 360;
    static constexpr int RANSAC_N = 100;
    static constexpr float RANSAC_TH = 0.01;
#elif defined(CONF_VLP32)
    static constexpr int Range_Num = 32;
    static constexpr int H_Ang_Num = 360 * 5;
    static constexpr int R_Piece = 3;
    static constexpr int H_Ang_Piece = 10 * H_Ang_Num / 360;
    static constexpr int RANSAC_N = 20;
    static constexpr float RANSAC_TH = 0.01;
#endif

    // kitti
    // static constexpr int Range_Num = 64;
    // static constexpr int H_Ang_Num = 360 * 5;
    // static constexpr int R_Piece = 7;
    // static constexpr int H_Ang_Piece = 10 * H_Ang_Num / 360;
    // static constexpr int RANSAC_N = 100;
    // static constexpr float RANSAC_TH = 0.02;

    static constexpr int Down_sample = 4;
    // static constexpr int Down_sample = 1;


    // static constexpr std::array<int, 14> H_Ang_Piece_Arr{
    //     int(40.00 * H_Ang_Num / 360), 
    //     int(36.67 * H_Ang_Num / 360), 
    //     int(33.33 * H_Ang_Num / 360), 
    //     int(30.00 * H_Ang_Num / 360), 
    //     int(26.67 * H_Ang_Num / 360), 
    //     int(23.33 * H_Ang_Num / 360), 
    //     int(20.00 * H_Ang_Num / 360), 
    //     int(17.50 * H_Ang_Num / 360), 
    //     int(15.00 * H_Ang_Num / 360), 
    //     int(12.50 * H_Ang_Num / 360), 
    //     int(10.00 * H_Ang_Num / 360), 
    //     int( 7.50 * H_Ang_Num / 360), 
    //     int( 5.00 * H_Ang_Num / 360), 
    //     int( 0.00 * H_Ang_Num / 360),
    //     };


    // njust-bad
    // static constexpr int Range_Num = 64;
    // static constexpr int H_Ang_Num = 360 * 5;
    // static constexpr int R_Piece = 7;
    // static constexpr int H_Ang_Piece = 10 * H_Ang_Num / 360;
    // static constexpr int RANSAC_N = 100;
    // static constexpr float RANSAC_TH = 0.05;
#ifndef USE_FRACTURE
    std::tuple<cv::Mat3f, cv::Mat1b, cv::Mat1b> process(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
#else
    std::tuple<cv::Mat3f, cv::Mat1b, cv::Mat1b, cv::Mat1b> process(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
#endif

private:
    std::tuple<cv::Mat3f, cv::Mat1b> make_imagery(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
    std::tuple<cv::Mat3f, cv::Mat1b> make_imagery_downsample(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
};

#endif
