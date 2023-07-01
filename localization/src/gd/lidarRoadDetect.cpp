#include "lidarRoadDetect.h"
#include "commonRansac.h"
#include <pcl/visualization/cloud_viewer.h>

struct PlaneModel
{
    using PointType = cv::Point3f;
    using Extra = int;
    cv::Point3f normal;
    float distance;
    bool reject = false;
    PlaneModel() : normal{0, 0, 0}, distance(0) {}
    PlaneModel(const PlaneModel &model) : normal(model.normal), distance(model.distance) {}
    PlaneModel(const std::vector<cv::Point3f> &points, const std::vector<int> &extra)
    {
        if (extra[0] == extra[1] && extra[0] == extra[2])
        {
            reject = true;
            return;
        }
        normal = (points[1] - points[0]).cross(points[2] - points[0]);
        float norm = cv::norm(normal);
        normal.x /= norm;
        normal.y /= norm;
        normal.z /= norm;
        if (normal.z < 0)
        {
            normal = -normal;
        }
        distance = -normal.dot(points[0]);
    }
    float operator()(const cv::Point3f &point)
    {
        return std::abs(normal.dot(point) + distance);
    }
};

cv::Mat1b calcNegObs(const cv::Mat3f &imagery, const cv::Mat1b &valid)
{
    cv::Mat1f xyz[3];
    cv::split(imagery, xyz);
    cv::Mat1f distance = xyz[0].mul(xyz[0]) + xyz[1].mul(xyz[1]);
    cv::sqrt(distance, distance);
    cv::Mat1f gridX = cv::Mat1f::zeros(imagery.size());
    cv::Mat1f gridY = cv::Mat1f::zeros(imagery.size());
    // int cols = distance.cols;
    // for (int c = 0; c < cols; c++)
    // {
    //     auto z0 = xyz[2].col((c + cols - 1) % cols);
    //     auto z1 = xyz[2].col(c);
    //     auto z2 = xyz[2].col((c + 1) % cols);
    //     auto x0 = distance.col((c + cols - 1) % cols);
    //     auto x1 = distance.col(c);
    //     auto x2 = distance.col((c + 1) % cols);
    //     // gridX.col(c) = cv::abs((z1 - z0).mul(1 / (x1 - x0)) - (z2 - z1).mul(1 / (x2 - x1)));
    //     gridX.col(c) = -((z1 - z0) - (z2 - z1));
    //     // gridY.col(c) = cv::abs((x1 - x0) - (x2 - x1));
    //     // gridX.col(c) = cv::abs((z1 - z0) - (z2 - z1));
    // }
    int cols = distance.cols;
    int rows = distance.rows;
    for (int r = 1; r < rows / 4 * 3 - 1; r++)
    {
        auto z0 = xyz[2].row(r - 1);
        auto z1 = xyz[2].row(r);
        auto z2 = xyz[2].row(r + 1);
        auto x0 = distance.row(r - 1);
        auto x1 = distance.row(r);
        auto x2 = distance.row(r + 1);
        // gridX.row(r) = cv::abs((z1 - z0).mul(1 / (x1 - x0)) - (z2 - z1).mul(1 / (x2 - x1)));
        gridY.row(r) = -((z1 - z0) - (z2 - z1));
        for (int c = 0; c < cols; c++)
        {
            if (valid(r - 1, c) == 0 || valid(r, c) == 0 || valid(r + 1, c) == 0)
                gridY(r, c) = 0;
        }
    }
    cv::Mat1b output = gridY > 0.08;
    return output;
}

#ifndef USE_FRACTURE
std::tuple<cv::Mat3f, cv::Mat1b, cv::Mat1b> LidarRoadDetect::process(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
#else
std::tuple<cv::Mat3f, cv::Mat1b, cv::Mat1b, cv::Mat1b> LidarRoadDetect::process(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
#endif
{
    cv::Mat3f imagery;
    cv::Mat1b valid;
#if defined(CONF_VLP16) || defined(CONF_VLP16_SIMULATOR) || defined(CONF_VLP32)
    std::tie(imagery, valid) = make_imagery(cloud);
#elif defined(CONF_VLP64)
    std::tie(imagery, valid) = make_imagery_downsample(cloud);
#endif
    cv::Mat1b mask = cv::Mat1b::zeros(imagery.size());
#ifdef USE_FRACTURE
    cv::Mat1b fracture = cv::Mat1b::zeros(imagery.size());
    cv::Mat1b basic_patch = cv::Mat1b::zeros(R_Piece, H_Ang_Piece);
    basic_patch.row(0) = 0x44;
    basic_patch.row(1) = 0x22;
    basic_patch.row(2) = 0x11;
#endif
    for (int R = 0; R < Range_Num - R_Piece; R++)
    {
        // const int H_Ang_Piece = H_Ang_Piece_Arr[R];
        // if (H_Ang_Piece == 0)
        //     continue;
        for (int C = 0; C < H_Ang_Num; C += H_Ang_Piece / 2)
        {
            std::vector<cv::Point3f> points;
            std::vector<int> extra;
            for (int i = 0; i < R_Piece; i++)
            {
                for (int j = 0; j < H_Ang_Piece; j++)
                {
                    const auto &p = imagery((R + i) % Range_Num, (C + j) % H_Ang_Num); // TODO: remove "%Range_Num"
                    if (p[0] == 0 && p[1] == 0 && p[2] == 0)
                        continue;
                    points.push_back({p[0], p[1], p[2]});
                    extra.push_back((R + i) % Range_Num);
                }
            }
            if (points.size() < 3)
                continue;
            int max_count;
            PlaneModel model;
            std::tie(max_count, model) = ransac<cv::Point3f, int, 3, PlaneModel>(RANSAC_N, RANSAC_TH, points, extra, [](const PlaneModel &m)
                                                                                 {
                                                                                     // return m.normal.z > 0.9;
                                                                                     // return true;
                                                                                     return !m.reject;
                                                                                 });
            if (max_count >= points.size() / 2)
            {
                int value = 0; // OBS_INFO
                if (model.normal.z > 0.9)
                    value = 255;
                else if (model.normal.z < 0.5)
                    value = 128;
                for (int i = 0; i < R_Piece; i++)
                {
                    for (int j = 0; j < H_Ang_Piece; j++)
                    {
                        const auto &p = imagery((R + i) % Range_Num, (C + j) % H_Ang_Num);
                        if (model(cv::Point3f{p[0], p[1], p[2]}) <= RANSAC_TH)
                        {
                            mask((R + i) % Range_Num, (C + j) % H_Ang_Num) |= value;
#ifdef USE_FRACTURE
                            fracture((R + i) % Range_Num, (C + j) % H_Ang_Num) |= basic_patch(i, j) & value;
#endif
                        }
                    }
                }
            }
        }
    }
    return {imagery, mask & ~calcNegObs(imagery, valid), valid};
}

std::tuple<cv::Mat3f, cv::Mat1b> LidarRoadDetect::make_imagery(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    std::vector<float> ang_list;
    for (const auto &p : *cloud)
    {
        float ang = std::atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y));
        ang_list.push_back(ang);
    }
    Eigen::Map<Eigen::ArrayXf> ang_array(&ang_list.front(), ang_list.size());
    float min_ang = ang_array.minCoeff();
    float max_ang = ang_array.maxCoeff();

    float delta_ang = (max_ang - min_ang) / (Range_Num - 1);
    cv::Mat3f imagery = cv::Mat3f::zeros(Range_Num, H_Ang_Num);
    cv::Mat1b valid = cv::Mat1b::zeros(Range_Num, H_Ang_Num);
    for (size_t i = 0; i < cloud->size(); i++)
    {
        const auto &p = cloud->at(i);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue; // debug
        float theta = std::atan2(p.y, p.x) * H_Ang_Num / M_PI / 2.0;
        int index = std::floor((ang_list[i] - min_ang) / delta_ang + 0.5);
        int column = std::floor(theta + 0.5);
        column = (column + H_Ang_Num) % H_Ang_Num;
        imagery(index, column) = {p.x, p.y, p.z};
        valid(index, column) = 255;
    }
    return {imagery, valid};
}

std::tuple<cv::Mat3f, cv::Mat1b> LidarRoadDetect::make_imagery_downsample(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    std::vector<float> ang_list;
    for (const auto &p : *cloud)
    {
        float ang = std::atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y));
        ang_list.push_back(ang);
    }
    Eigen::Map<Eigen::ArrayXf> ang_array(&ang_list.front(), ang_list.size());
    float min_ang = ang_array.minCoeff();
    float max_ang = ang_array.maxCoeff();

    float delta_ang = (max_ang - min_ang) / (Range_Num * Down_sample - 1);
    cv::Mat3f imagery = cv::Mat3f::zeros(Range_Num * Down_sample, H_Ang_Num);
    cv::Mat1b valid = cv::Mat1b::zeros(Range_Num * Down_sample, H_Ang_Num);
    for (size_t i = 0; i < cloud->size(); i++)
    {
        const auto &p = cloud->at(i);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) /* || p.intensity == 0*/)
            continue; // debug
        if ((p.x * p.x + p.y * p.y) < 3 * 3)
            continue;
        float theta = std::atan2(p.y, p.x) * H_Ang_Num / M_PI / 2.0;
        int index = std::floor((ang_list[i] - min_ang) / delta_ang + 0.5);
        int column = std::floor(theta + 0.5);
        column = (column + H_Ang_Num) % H_Ang_Num;
        imagery(index, column) = {p.x, p.y, p.z};
        valid(index, column) = 255;
    }
    cv::Mat3f imagery_down = cv::Mat3f::zeros(Range_Num, H_Ang_Num);
    cv::Mat1b valid_down = cv::Mat1b::zeros(Range_Num, H_Ang_Num);
    for (int i = 0; i < Range_Num; i++)
    {
        imagery.row(i * Down_sample + 3).copyTo(imagery_down.row(i));
        valid.row(i * Down_sample + 3).copyTo(valid_down.row(i));
    }
    return {imagery_down, valid_down};
}
