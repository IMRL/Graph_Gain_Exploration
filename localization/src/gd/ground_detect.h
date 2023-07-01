#ifndef _GROUND_DETECT_H_
#define _GROUND_DETECT_H_

#include "config.h"
#include <chunkmap/chunk_map.h>
#include <chunkmap/simple_probability_values.h>

#if defined(USE_BIT8)
constexpr auto obstacle_segments = ObstacleFeature::Seg8;
#elif defined(USE_BIT32)
constexpr auto obstacle_segments = ObstacleFeature::Seg32;
#endif
#if defined(OBS_PROB) && defined(OBS_BIT4)
constexpr auto obstacle_bits = ObstacleFeature::Bit4;
#else
constexpr auto obstacle_bits = ObstacleFeature::Bit1;
#endif
using ObstacleTypeDesc = ChunkMap::ObstacleTypeDesc<obstacle_segments, obstacle_bits>;
constexpr ObstacleTypeDesc obstacle_type_desc;
using ObstacleProb = simple_prob_values<obstacle_bits>;

#ifdef OBS_PROB
const auto ObsProbMat = ObstacleProb::generate_value_mat();
const auto ObsProbHitTable = ObstacleProb::generate_step_table(0.9);
const auto ObsProbMissTable = ObstacleProb::generate_step_table(0.42);
// ObstacleTypeDesc::ElemType obstacle_init_data;
#endif

struct DetectInfo
{
    cv::Mat3b map;
    cv::Mat1f map_info;

    ObstacleTypeDesc::Type obstacle_info;

    ChunkMap::CompressedDesc feature;

    // cv::Mat3b landImage;
};

// struct MiniDetectInfo
// {
//     std::vector<Eigen::Vector3d> obstacle;
//     LidarIris::FeatureDesc feature;

//     cv::Mat3b map;
//     cv::Mat3b landImage;
// };

struct GroundDetectConfig
{
    float resolution;       // = 0.1;
    float hit_probability;  // = 0.45;
    float miss_probability; // = 0.51;
    float submap_radius;    // = 20;
    double robot_height;    // = 0.5;
};

// class ImageLandWrapper
// {
// protected:
//     cv::Mat1f unmapX, unmapY;
//     cv::Mat1f wrmapX, wrmapY;

// public:
//     cv::Mat1b observe;
//     cv::Mat1b front_obs;

//     virtual ~ImageLandWrapper() {}

//     virtual void init(float submap_radius, float resolution) = 0;
//     virtual cv::Mat1b proj(const pcl::PointCloud<pcl::PointXYZ> &cloud) = 0;

//     inline cv::Mat3b wrap(const cv::Mat3b image)
//     {
//         cv::Mat3b ret;
//         cv::remap(image, ret, unmapX, unmapY, cv::INTER_NEAREST);
//         cv::remap(ret, ret, wrmapX, wrmapY, cv::INTER_NEAREST);
//         return ret;
//     }

//     inline cv::Mat3b front(const cv::Mat3b &image)
//     {
//         cv::Mat ret;
//         cv::remap(image, ret, unmapX, unmapY, cv::INTER_NEAREST);
//         return ret;
//     }
// };

class GroundDetect
{
public:
    using Ptr = std::shared_ptr<GroundDetect>;
    virtual ~GroundDetect() {}

    // static Ptr create(GroundDetectConfig config, std::shared_ptr<ImageLandWrapper> wrapper);
    static Ptr create(GroundDetectConfig config);
    // virtual std::shared_ptr<DetectInfo> process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const cv::Mat3b &image) = 0;
    virtual std::shared_ptr<DetectInfo> process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) = 0;
    // virtual std::shared_ptr<MiniDetectInfo> mini_process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const cv::Mat3b &image) = 0;

    std::shared_ptr<LidarIris> iris;

protected:
    GroundDetect() {}
};

#endif
