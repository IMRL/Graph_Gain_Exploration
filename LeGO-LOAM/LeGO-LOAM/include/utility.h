#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_


#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include "cloud_msgs/cloud_info.h"

// #include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
 
#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#define PI 3.14159265

using namespace std;
namespace pandar_pointcloud
{
struct PointXYZIT {
    PCL_ADD_POINT4D
    uint8_t intensity;
    double timestamp;
    uint16_t ring;                      ///< laser ring number
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;

}; // namespace pandar_pointcloud

POINT_CLOUD_REGISTER_POINT_STRUCT(pandar_pointcloud::PointXYZIT,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (uint8_t, intensity, intensity)(double, timestamp, timestamp)(uint16_t, ring, ring))

typedef pandar_pointcloud::PointXYZIT PPoint;

typedef pcl::PointXYZI  PointType;

// extern const string pointCloudTopic = "/kitti/velo/pointcloud";
// extern const string imageTopic = "/kitti/camera_color_left/image_raw";
extern const string imageTopic = "/camera/image_left_1";
// extern const string pointCloudTopic = "/pandora/sensor/pandora/hesai40/PointCloud2";
extern const string pointCloudTopic = "/velodyne_points";//"/rslidar_points";//"/velodyne_points";
// extern const string pointCloudTopic = "/kitti/velo/pointcloud";
// extern const string imuTopic = "/imu/data";
extern const string imuTopic = "/imu/data22";
extern const string gpsTopic = "/kitti/oxts/gps/fix22";

// Save pcd
// extern const string fileDirectory = "/home/yingwang/lio_map/";
extern const string fileDirectory = "/home/azurity/projects/octopoly/new-exp/lego-temp/";

// // VLP-16
// extern const int N_SCAN = 16;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 0.2;
// extern const float ang_res_y = 2;
// extern const float ang_bottom = 15.0+0.1;
// // extern const int groundScanInd = 1;
// extern const int groundScanInd = 7;

// HDL-32E
extern const int N_SCAN = 32;
extern const int Horizon_SCAN = 1800;
extern const float ang_res_x = 360.0/float(Horizon_SCAN);
extern const float ang_res_y = 41.33/float(N_SCAN-1);
extern const float ang_bottom = 30.67;
extern const int groundScanInd = 20;

// HDL-32E
// extern const int N_SCAN = 40;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 23/float(N_SCAN-1);
// extern const float ang_bottom = 16;
// extern const int groundScanInd = 20;//20

// Ouster users may need to uncomment line 159 in imageProjection.cpp
// Usage of Ouster imu data is not supported yet, please just publish point cloud data
// Ouster OS1-16
// extern const int N_SCAN = 16;
// extern const int Horizon_SCAN = 1024;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 7;

// Ouster OS1-64
// extern const int N_SCAN = 64;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 26.8/float(64-1);
// extern const float ang_bottom = 24.8; //24.8
// extern const int groundScanInd = 30;

// //Vel 64
// extern const int N_SCAN = 64;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 0.2;
// extern const float ang_res_y = 0.427;
// extern const float ang_bottom = 24.9;
// extern const int groundScanInd = 50;

extern const bool loopClosureEnableFlag = true;
extern const double mappingProcessInterval = 0.3;
// extern const double mappingProcessInterval = 0.1;

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;
extern const int imuQueLength = 200;
// extern const int imuQueLength = 100;

extern const float sensorMountAngle = 0.0;
extern const float segmentTheta = 60.0/180.0*M_PI; // decrese this value may improve accuracy
extern const int segmentValidPointNum = 5;//5
extern const int segmentValidLineNum = 3; //3
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI;


extern const int edgeFeatureNum = 2;//2
extern const int surfFeatureNum = 4;
extern const int sectionsTotal = 6;
extern const float edgeThreshold = 0.1;// 0.1
extern const float surfThreshold = 0.1;
extern const float nearestFeatureSearchSqDist = 25;

// Mapping Params//50
// extern const float surroundingKeyframeSearchRadius = 50.0; // key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
extern const float surroundingKeyframeSearchRadius = 100.0; // key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
extern const int   surroundingKeyframeSearchNum = 50; // submap size (when loop closure enabled)
// history key frames (history submap for loop closure)
extern const float historyKeyframeSearchRadius = 40; // key frame that is within n meters from current pose will be considerd for loop closure
// extern const float historyKeyframeSearchRadius = 150; // key frame that is within n meters from current pose will be considerd for loop closure
extern const int   historyKeyframeSearchNum = 25;//25 // 2n+1 number of hostory key frames will be fused into a submap for loop closure
// extern const float historyKeyframeFitnessScore = 0.3; // the smaller the better alignment //0.5
// extern const float historyKeyframeFitnessScore = 0.5; // the smaller the better alignment //0.5
extern const float historyKeyframeFitnessScore = 0.5; // the smaller the better alignment //0.5

extern const float globalMapVisualizationSearchRadius = 800.0; // key frames with in n meters will be visualized


struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time)
)

typedef PointXYZIRPYT  PointTypePose;

#endif
