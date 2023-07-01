#ifndef _COMMON_RANSAC_H_
#define _COMMON_RANSAC_H_

#include <functional>
#include <vector>
#include <array>
#include <set>
#include <algorithm>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

template <typename PointType, typename Extra, size_t PointNeedN, typename ModelArgments>
std::tuple<size_t, ModelArgments> ransac(size_t RANSAC_N, float RANSAC_TH, const std::vector<PointType> &points, const std::vector<Extra> &extra, const std::function<bool(const ModelArgments &)> &checker)
{
    size_t max_count = 0;
    ModelArgments best_model;
    for (size_t n = 0; n < RANSAC_N; n++)
    {
        int length = points.size();
        if (length < PointNeedN)
            break;
        std::set<int> indices;
        while (indices.size() < PointNeedN)
        {
            indices.insert(rand() % length);
        }
        std::vector<int> indices_vec;
        std::vector<Extra> extra_vec;
        std::copy(indices.begin(), indices.end(), std::back_inserter(indices_vec));
        std::vector<PointType> points_vec;
        for (const auto &index : indices_vec)
        {
            points_vec.push_back(points[index]);
            extra_vec.push_back(extra[index]);
        }
        ModelArgments model(points_vec, extra_vec);
        if (!checker(model))
            continue;
        //
        size_t count = 0;
        for (const auto &p : points)
        {
            if (model(p) <= RANSAC_TH)
                count++;
        }
        if (count > max_count)
        {
            max_count = count;
            best_model = model;
        }
    }
    return {max_count, best_model};
}

#endif
