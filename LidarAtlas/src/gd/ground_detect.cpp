#include "ground_detect.h"

#include <string>
#include <map>
#include <Eigen/Eigen>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <pcl/filters/conditional_removal.h>

#include "lidarRoadDetect.h"
#include "legoFeature.h"
#include "probability_values.h"

// #include <azurity-tools/utils/tictoc.hpp>

// #define REMOVE_FREE
// #define DEBUG_SHOW_VIZ
#define USE_CHUNK_MAP
// #define NO_MULTI_LAYER
#define OBS_INFO

// #define outfile

const float obstacle_resolution = 1.0; //0.2;
const float obstacle_base = 0;         //-1.4;

class GroundDetectImpl : public GroundDetect
{
    bool is_neighbor_unknown = true;
    static constexpr float padding = 1.0f;
    GroundDetectConfig config_;

    std::vector<bool> ang_mask;
    std::vector<uint16_t> hit_table;
    std::vector<uint16_t> miss_table;
    uint16_t inital_map_value;
    // std::shared_ptr<ImageLandWrapper> imageLandWrapper;

    cv::Point2i mapOffset;

    LidarRoadDetect rd_instance;

    /* #region */
    cv::Mat1b mask_filter(const cv::Mat1b &mask)
    {
        static cv::Mat1b mask_mat = cv::Mat1b::ones(3, 7);
        cv::Mat1b proc = (mask != 255) / 255;
        cv::filter2D(proc, proc, CV_8U, mask_mat);
        return mask | (proc <= 3);
    }

#ifndef BETTER
    cv::Mat1f make_hits(const cv::Mat3f &imagery, const cv::Mat1b &mask, const cv::Mat1b &valid, float max_range)
    {
        cv::Mat1f xyz[3];
        cv::split(imagery, xyz);
        cv::Mat1b z_thres = xyz[2] < config_.robot_height;
        cv::Mat1b obstacle = z_thres & valid & (mask != 255);
        cv::Mat1f dis;
        cv::sqrt(xyz[0].mul(xyz[0]) + xyz[1].mul(xyz[1]), dis);
        cv::Mat1f terminal_point = cv::Mat1f::ones(1, dis.cols) * -max_range;
        for (int i = 0; i < terminal_point.cols; i++)
        {
            double current_dis = 0;
            if (cv::sum(obstacle.col(i))[0] > 0)
            {
                cv::minMaxIdx(dis.col(i), &current_dis, nullptr, nullptr, nullptr, obstacle.col(i));
            }
            else
            {
                cv::minMaxIdx(dis.col(i), nullptr, &current_dis, nullptr, nullptr, obstacle.col(i));
                current_dis = -current_dis;
            }
            if (current_dis == 0 || std::abs(current_dis) > max_range)
                continue;
            terminal_point(0, i) = current_dis;
        }
        return terminal_point;
    }
#else
    cv::Mat1f make_hits(const cv::Mat3f &imagery, const cv::Mat1b &mask, const cv::Mat1b &valid, float max_range)
    {
        cv::Mat1f xyz[3];
        cv::split(imagery, xyz);
        cv::Mat1b z_thres = xyz[2] < config_.robot_height;
        cv::Mat1b obstacle = z_thres & valid & (mask != 255);
        cv::Mat1b ground = z_thres & valid & (mask == 255);
        cv::Mat1f dis;
        cv::sqrt(xyz[0].mul(xyz[0]) + xyz[1].mul(xyz[1]), dis);
        // cv::Mat1f terminal_point = cv::Mat1f::ones(1, dis.cols) * -max_range;
        cv::Mat1f terminal_point = cv::Mat1f::zeros(3, dis.cols);
#ifdef USE_FRACTURE
        cv::Mat1b defracture;
        {
            cv::Mat1b fracture_low = fracture & 0x0f;
            cv::Mat1b fracture_high = fracture & 0xf0;
            defracture = (((fracture_low.rowRange(0, 15) / 2) & fracture_low.rowRange(1, 16)) != 0) |
                         ((fracture_high.rowRange(0, 15) / 2) & fracture_high.rowRange(1, 16)) != 0;
        }
#endif
        for (int i = 0; i < terminal_point.cols; i++)
        {
            double current_dis = std::numeric_limits<float>::max();
            if (cv::sum(obstacle.col(i))[0] > 0)
            {
                cv::minMaxIdx(dis.col(i), &current_dis, nullptr, nullptr, nullptr, obstacle.col(i));
                if (current_dis < max_range)
                {
                    terminal_point(0, i) = current_dis;
                }
            }
            double dis_min = 0;
            double dis_max = 0;
            cv::Mat1b land_mask = ground.col(i) & (dis.col(i) < current_dis);
#ifndef USE_FRACTURE
            if (cv::sum(land_mask)[0] > 0)
            {
                cv::minMaxIdx(dis.col(i), &dis_min, &dis_max, nullptr, nullptr, land_mask);
                if (dis_min > max_range)
                    dis_min = max_range;
                if (dis_max > max_range)
                    dis_max = max_range;
            }
#else
            // if (dis(0, i) > /*4.6*/ 3.2)
            {
                for (int j = 1; j < imagery.rows; j++)
                {
                    if (!land_mask(j))
                        break;
                    // if (!defracture(j - 1, i))
                    //     break;
                    if (dis_min == 0)
                        dis_min = dis(j, i);
                    dis_max = std::max(dis_max, (double)dis(j, i));
                }
                if (dis_min > max_range)
                    dis_min = max_range;
                if (dis_max > max_range)
                    dis_max = max_range;
            }
#endif
            terminal_point(1, i) = dis_min;
            terminal_point(2, i) = dis_max;
        }
        return terminal_point;
    }
#endif

#ifndef BETTER
    cv::Mat3b prob_draw(const cv::Point2i &map_offset, const cv::Mat1f &hits, float resolution)
    {
        cv::Mat_<uint16_t> submap_value = cv::Mat_<uint16_t>::ones(std::round(2 * config_.submap_radius / resolution), std::round(2 * config_.submap_radius / resolution)) * inital_map_value;
        float k = 2 * M_PI / hits.cols;
        for (int i = 0; i < hits.cols; i++)
        {
            if (!ang_mask[i])
                continue;
            float ang = i * k;
            bool free = hits(0, i) < 0;
            float dis = std::abs(hits(0, i));
            Eigen::Vector2f begin = Eigen::Vector2f{std::cos(ang) * dis, std::sin(ang) * dis} / resolution;
            Eigen::Vector2f end{0, 0};
            {
                // bresenham line
                Eigen::Vector2f delta = end - begin;
                Eigen::Vector2i deltaSign{(delta.x() > 0 ? 1 : -1), (delta.y() > 0 ? 1 : -1)};
                delta = delta.cwiseAbs();
                cv::Point2i inital = cv::Point2i{int(begin.x()), int(begin.y())} + map_offset;
                bool is_swaped = false;
                if (delta.y() > delta.x())
                {
                    float temp = delta.x();
                    delta.x() = delta.y();
                    delta.y() = temp;
                    is_swaped = true;
                }
                float p = 2 * delta.y() - delta.x();
                if (!free)
                    submap_value(inital) = hit_table[submap_value(inital)];
                for (int j = 0; j <= delta.x(); j++)
                {
                    if (p < 0)
                    {
                        if (is_swaped == 0)
                        {
                            inital.x += deltaSign.x();
                        }
                        else
                        {
                            inital.y += deltaSign.y();
                        }
                        p = p + 2 * delta.y();
                    }
                    else
                    {
                        inital.x += deltaSign.x();
                        inital.y += deltaSign.y();
                        p = p + 2 * delta.y() - 2 * delta.x();
                    }
                    submap_value(inital) = miss_table[submap_value(inital)];
                }
            }
        }
        cv::Mat3b result(submap_value.size());
        for (int i = 0; i < submap_value.rows; i++)
            for (int j = 0; j < submap_value.cols; j++)
            {
                const int delta = 128 - cartographer::mapping::ProbabilityToLogOddsInteger(cartographer::mapping::ValueToProbability(submap_value(i, j)));
                const uint8_t alpha = delta > 0 ? 0 : -delta;
                const uint8_t value = delta > 0 ? delta : 0;
                result(i, j) = cv::Vec3b{value, ((value || alpha) ? alpha : (uint8_t)1), (submap_value(i, j) == inital_map_value ? (uint8_t)0 : (uint8_t)255)};
            }
        return result;
    }
#else
    cv::Mat3b prob_draw(const cv::Point2i &map_offset, const cv::Mat1f &hits, float resolution)
    {
        static int initial_safe = 0;
        initial_safe+=1;
        cv::Mat_<uint16_t> submap_value = cv::Mat_<uint16_t>::ones(std::round(2 * config_.submap_radius / resolution), std::round(2 * config_.submap_radius / resolution)) * inital_map_value;
        float k = 2 * M_PI / hits.cols;
        for (int i = 0; i < hits.cols; i++)
        {
            if (!ang_mask[i])
                continue;
            float ang = i * k;
            // bool free = hits(0, i) < 0;
            // float dis = std::abs(hits(0, i));
            Eigen::Vector2f begin{0, 0};
            if (is_neighbor_unknown && initial_safe > 0) {
                begin = Eigen::Vector2f{std::cos(ang) * hits(1, i), std::sin(ang) * hits(1, i)} / resolution;
            }
            // Eigen::Vector2f begin = Eigen::Vector2f{std::cos(ang) * hits(1, i), std::sin(ang) * hits(1, i)} / resolution;


            // furthest obs
            Eigen::Vector2f end = Eigen::Vector2f{std::cos(ang) * hits(2, i), std::sin(ang) * hits(2, i)} / resolution;

            // furthest available ground
            // float real_dis = std::max(hits(0, i), hits(2, i));
            // Eigen::Vector2f end = Eigen::Vector2f{std::cos(ang) * real_dis, std::sin(ang) * real_dis} / resolution;
            if (hits(0, i) > 0)
            {
                Eigen::Vector2f hitf = Eigen::Vector2f{std::cos(ang) * hits(0, i), std::sin(ang) * hits(0, i)} / resolution;
                cv::Point2i hit = cv::Point2i{int(hitf.x()), int(hitf.y())} + map_offset;
                submap_value(hit) = hit_table[submap_value(hit)];
            }
            // Eigen::Vector2f end{0, 0};
            if (hits(1, i) > 0 && hits(2, i) > 0)
            // if (hits(0, i) > 0)
            {
                // bresenham line
                Eigen::Vector2f delta = end - begin;
                Eigen::Vector2i deltaSign{(delta.x() > 0 ? 1 : -1), (delta.y() > 0 ? 1 : -1)};
                delta = delta.cwiseAbs();
                cv::Point2i inital = cv::Point2i{int(begin.x()), int(begin.y())} + map_offset;
                bool is_swaped = false;
                if (delta.y() > delta.x())
                {
                    float temp = delta.x();
                    delta.x() = delta.y();
                    delta.y() = temp;
                    is_swaped = true;
                }
                float p = 2 * delta.y() - delta.x();
                for (int j = 0; j <= delta.x(); j++)
                {
                    if (p < 0)
                    {
                        if (is_swaped == 0)
                        {
                            inital.x += deltaSign.x();
                        }
                        else
                        {
                            inital.y += deltaSign.y();
                        }
                        p = p + 2 * delta.y();
                    }
                    else
                    {
                        inital.x += deltaSign.x();
                        inital.y += deltaSign.y();
                        p = p + 2 * delta.y() - 2 * delta.x();
                    }
                    submap_value(inital) = miss_table[submap_value(inital)];
                }
            }
            }
        uint8_t px_w = (uint8_t)(config_.submap_radius / resolution);
        for (int i = px_w-2; i < px_w+2; i++)
            for (int j = px_w-2; j < px_w+2; j++)
            {
                submap_value(cv::Point2i(i, j)) = miss_table[submap_value(cv::Point2i(i, j))];
        }
        cv::Mat3b result(submap_value.size());
        for (int i = 0; i < submap_value.rows; i++)
            for (int j = 0; j < submap_value.cols; j++)
            {
                const int delta = 128 - cartographer::mapping::ProbabilityToLogOddsInteger(cartographer::mapping::ValueToProbability(submap_value(i, j)));  // Fig.5(c)
                const uint8_t alpha = delta > 0 ? 0 : -delta;
                const uint8_t value = delta > 0 ? delta : 0;
                result(i, j) = cv::Vec3b{value, ((value || alpha) ? alpha : (uint8_t)1), (submap_value(i, j) == inital_map_value ? (uint8_t)0 : (uint8_t)255)};
            }
        return result;
    }
#endif

    ObstacleTypeDesc::Type obstacle_info_draw(const cv::Point2i &map_offset, const cv::Mat3f &imagery, const cv::Mat1b &mask, const cv::Mat1b &valid, float resolution)
    {
        ObstacleTypeDesc::Type submap_value = ObstacleTypeDesc::Type::zeros(std::round(2 * config_.submap_radius / config_.resolution), std::round(2 * config_.submap_radius / config_.resolution));
        {
            cv::Mat1f xyz[3];
            cv::split(imagery, xyz);
            cv::Mat1b z_thres = xyz[2] >= obstacle_base; //0;
            cv::Mat1b obstacle = z_thres & valid & (mask != 255);
#ifdef OBS_PROB
            cv::Mat1b valid_ray = z_thres & valid;
#endif
            for (int r = 0; r < imagery.rows; r++)
                for (int c = 0; c < imagery.cols; c++)
                {
                    // debug
                    if (!ang_mask[c])
                        continue;
#ifndef OBS_PROB
                    if (!obstacle(r, c))
                        continue;
                    // if (!ang_mask[c])
                    //     continue;
                    cv::Vec3f pos = imagery(r, c);
                    float dis = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    if (dis >= config_.submap_radius)
                        continue;
                    Eigen::Vector2f point = Eigen::Vector2f{pos[0], pos[1]} / resolution;
                    cv::Point2i inital = cv::Point2i{int(point.x()), int(point.y())} + map_offset;
                    int height = std::floor((pos[2] - obstacle_base) / obstacle_resolution);
                    if (height > obstacle_type_desc.max_height)
                        height = obstacle_type_desc.max_height;
                    ObstacleTypeDesc::ElemType value;
                    value.set(height);
                    ObstacleTypeDesc::wrapper(submap_value(inital)) |= value;
#else
                    if (!valid_ray(r, c))
                        continue;
                    cv::Vec3f pos = imagery(r, c);
                    float dis = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
                    if (dis >= config_.submap_radius)
                        continue;
                    Eigen::Vector2f point = Eigen::Vector2f{pos[0], pos[1]} / resolution;
                    Eigen::Vector2f center{0, 0};
                    {
                        cv::Rect rect{0, 0, submap_value.cols, submap_value.rows};
                        // bresenham line
                        Eigen::Vector2f delta = center - point;
                        Eigen::Vector2i deltaSign{(delta.x() > 0 ? 1 : -1), (delta.y() > 0 ? 1 : -1)};
                        delta = delta.cwiseAbs();
                        cv::Point2i inital = cv::Point2i{int(point.x()), int(point.y())} + map_offset;
                        bool is_swaped = false;
                        if (delta.y() > delta.x())
                        {
                            float temp = delta.x();
                            delta.x() = delta.y();
                            delta.y() = temp;
                            is_swaped = true;
                        }
                        float height_step = pos[2] / delta.x();
                        float p = 2 * delta.y() - delta.x();
                        for (int j = 0; j <= delta.x(); j++)
                        {
                            if (rect.contains(inital))
                            {
                                // TODO: calc the z in right way, such as resolution and boundary
                                // auto accessor = ObstacleTypeDesc::wrapper(submap_value(inital))((uint16_t)std::floor(pos[2] - height_step * j) * obstacle_bits, obstacle_bits);
                                auto accessor = ObstacleTypeDesc::wrapper(submap_value(inital))(std::min((uint16_t)std::floor((pos[2] - height_step * j - obstacle_base) / obstacle_resolution), (uint16_t)obstacle_type_desc.max_height) * obstacle_bits, obstacle_bits);
                                if (j == 0 && obstacle(r, c)) // hit
                                {
                                    accessor = ObsProbHitTable[(uint16_t)accessor];
                                }
                                else // miss
                                {
                                    accessor = ObsProbMissTable[(uint16_t)accessor];
                                }
                            }
                            if (p < 0)
                            {
                                if (is_swaped == 0)
                                {
                                    inital.x += deltaSign.x();
                                }
                                else
                                {
                                    inital.y += deltaSign.y();
                                }
                                p = p + 2 * delta.y();
                            }
                            else
                            {
                                inital.x += deltaSign.x();
                                inital.y += deltaSign.y();
                                p = p + 2 * delta.y() - 2 * delta.x();
                            }
                        }
                    }
#endif
                }
        }
        return submap_value;
    }
    /* #endregion */

    pcl::PointCloud<pcl::PointXYZ>::Ptr xyzToCloud(const cv::Mat3f &imagery, const cv::Mat1b &mask, const cv::Mat1b &valid)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr ret(new pcl::PointCloud<pcl::PointXYZ>);
        cv::Mat1f xyz[3];
        cv::split(imagery, xyz);
        cv::Mat1b z_thres = xyz[2] >= obstacle_base;
        cv::Mat1b obstacle = /*z_thres &*/ valid & (mask != 255);
        for (int r = 0; r < imagery.rows; r++)
            for (int c = 0; c < imagery.cols; c++)
            {
                // debug
                if (!ang_mask[c])
                    continue;
                if (obstacle(r, c))
                {
                    auto v = imagery(r,c);
                    pcl::PointXYZ p;
                    p.x = v[0];
                    p.y = v[1];
                    p.z = v[2];
                    ret->push_back(p);
                }
            }
        return ret;
    }

public:
    // GroundDetectImpl(GroundDetectConfig config, std::shared_ptr<ImageLandWrapper> wrapper) : config_(config), imageLandWrapper(wrapper)
    GroundDetectImpl(GroundDetectConfig config) : config_(config)
    {
        {
            // int th1 = 135 * LidarRoadDetect::H_Ang_Num / 360;
            // int th2 = 225 * LidarRoadDetect::H_Ang_Num / 360;
            // for (int i = 0; i < LidarRoadDetect::H_Ang_Num; i++)
            // {
            //     if (((i >= 247) && (i <= 308)) || ((i >= 687) && (i <= 748)) ||((i >= 1147) && (i <= 1648)))
            //         ang_mask.push_back(false);
            //     else
            //     // ang_mask.push_back(i < th1 || i > th2);
            //         ang_mask.push_back(true);
            // }


            // 0~
            for (int i = 0; i < LidarRoadDetect::H_Ang_Num; i++)
            {
                if (((i >= 247) && (i <= 308)) || ((i >= 687) && (i <= 748)) || (i > 900))
                    ang_mask.push_back(false);
                else
                // ang_mask.push_back(i < th1 || i > th2);
                    ang_mask.push_back(true);
            }
        }
        iris = std::make_shared<LidarIris>(4, 18, 1.6, 0.75, 2);
        hit_table = cartographer::mapping::ComputeLookupTableToApplyCorrespondenceCostOdds(cartographer::mapping::Odds(config_.hit_probability));
        miss_table = cartographer::mapping::ComputeLookupTableToApplyCorrespondenceCostOdds(cartographer::mapping::Odds(config_.miss_probability));
        inital_map_value = cartographer::mapping::CorrespondenceCostToValue(cartographer::mapping::ProbabilityToCorrespondenceCost(0.5));
        // imageLandWrapper->init(config_.submap_radius, config_.resolution);
        mapOffset = {std::round(config_.submap_radius / config_.resolution), std::round(config_.submap_radius / config_.resolution)};
    }
    virtual ~GroundDetectImpl() {}

    // virtual std::shared_ptr<DetectInfo> process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const cv::Mat3b &image) override
    virtual std::shared_ptr<DetectInfo> process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) override
    {
        auto info = std::make_shared<DetectInfo>();

        // lidar part
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
        if (cloud->size() == 0)
            return nullptr;

        auto result = rd_instance.process(cloud);
        auto map_info = make_hits(
            std::get<0>(result),  // xyz fig4(a) 
            mask_filter(std::get<1>(result)),  // Fig4(d)
            std::get<2>(result), // valid
            config_.submap_radius - 2 * config_.resolution);

        info->map = prob_draw(mapOffset, map_info, config_.resolution);

        auto lego_mask = legoFeature(std::get<0>(result), std::get<1>(result), std::get<2>(result), config_.submap_radius);
        info->obstacle_info = obstacle_info_draw(mapOffset, std::get<0>(result), mask_filter(std::get<1>(result)), std::get<2>(result) & lego_mask, config_.resolution);

        info->map_info = map_info.row(0);
        // info.feature = iris->GetFeature(iris->GetIris(*cloud));
        {
            auto feature = iris->GetFeature(iris->GetIris(*cloud));
            cv::imencode(".png", feature.img, info->feature.img);
            cv::imencode(".png", feature.T, info->feature.T);
            cv::imencode(".png", feature.M, info->feature.M);
        }

        #ifdef outfile
            cv::imwrite("/home/szz/Documents/chunkmap_backup_szz_20221122/results/fig4a.pfm", std::get<0>(result));
            cv::imwrite("/home/szz/Documents/chunkmap_backup_szz_20221122/results/fig4d.png", std::get<1>(result));
            cv::imwrite("/home/szz/Documents/chunkmap_backup_szz_20221122/results/fig4mask.png", std::get<2>(result));
            cv::imwrite("/home/szz/Documents/chunkmap_backup_szz_20221122/results/fig5c.png", info->map);
            std::ofstream laserFile;
            laserFile.open("/home/szz/Documents/chunkmap_backup_szz_20221122/results/laserFile.txt");
            for(int i = 0; i < info->map_info.cols; i++) {
                laserFile << info->map_info.at<float>(0, i) << " "  ;
            }
        #endif

        // camera part
        // info->landImage = imageLandWrapper->wrap(image);

        return info;
    }

#if 0
    // virtual std::shared_ptr<MiniDetectInfo> mini_process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const cv::Mat3b &image) override
    virtual std::shared_ptr<MiniDetectInfo> mini_process(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const cv::Mat3b &image) override
    {
        auto info = std::make_shared<MiniDetectInfo>();

        // lidar part
        // std::vector<int> indices;
        // pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
        // if (cloud->size() == 0)
        //     return nullptr;

        // aztools::utils::tictoc<std::chrono::duration<double>> tt;

        // tt.tic("rd");
        auto result = rd_instance.process(cloud);
        // tt.toc_and_log("rd");

        auto map_info = make_hits(
            std::get<0>(result),
            mask_filter(std::get<1>(result)),
            std::get<2>(result),
            config_.submap_radius - 2 * config_.resolution);
        // std::get<3>(result));
        info->map = prob_draw(mapOffset, map_info, config_.resolution);

        // tt.tic("lego");
        auto lego_mask = legoFeature(std::get<0>(result), std::get<1>(result), std::get<2>(result), config_.submap_radius);
        // tt.toc_and_log("lego");

        {
            auto imagery = std::get<0>(result);
            cv::Mat1f xyz[3];
            cv::split(imagery, xyz);
            cv::Mat1b z_thres = xyz[2] >= obstacle_base; //0;
            cv::Mat1b obsMask = z_thres & (std::get<2>(result) & lego_mask) & (mask_filter(std::get<1>(result)) != 255);
            for (int r = 0; r < obsMask.rows; r++)
                for (int c = 0; c < obsMask.cols; c++)
                    if (obsMask(r, c))
                    {
                        auto p = imagery(r, c);
                        info->obstacle.push_back({p[0], p[1], std::floor(p[2])});
                    }
        }

        // info->map_info = map_info.row(0);
        info->feature = iris->GetFeature(iris->GetIris(*cloud));
        // {
        //     auto feature = iris->GetFeature(iris->GetIris(*cloud));
        //     cv::imencode(".png", feature.img, info->feature.img);
        //     cv::imencode(".png", feature.T, info->feature.T);
        //     cv::imencode(".png", feature.M, info->feature.M);
        // }

        // camera part
        // info->landImage = imageLandWrapper->wrap(image);

        return info;
    }
#endif
};

// GroundDetect::Ptr GroundDetect::create(GroundDetectConfig config, std::shared_ptr<ImageLandWrapper> wrapper)
GroundDetect::Ptr GroundDetect::create(GroundDetectConfig config)
{
    // return std::make_shared<GroundDetectImpl>(config, wrapper);
    return std::make_shared<GroundDetectImpl>(config);
}
