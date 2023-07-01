#ifndef _MAP_BUILDER_H_
#define _MAP_BUILDER_H_

#include "ground_detect.h"
#include "mutli_execute.h"

struct MapBuilderConfig
{
    float resolution;
    float submap_radius;
};

struct SubmapInfo
{
    std::shared_ptr<DetectInfo> detect;

    Eigen::Isometry3d pose;
    Eigen::Isometry3d old_pose;
    bool has_old_pose;
    bool inited;

    cv::Rect global_rect;
};

class MapBuilder
{
public:
    using Ptr = std::shared_ptr<MapBuilder>;

    virtual ~MapBuilder() {}

    // static Ptr create(MapBuilderConfig config, std::shared_ptr<ImageLandWrapper> wrapper, ChunkMap::Ptr chunk_map);
    static Ptr create(MapBuilderConfig config, ChunkMap::Ptr chunk_map);

    template <size_t N1, size_t N2>
    std::set<ChunkMap::Index> draw_multi(std::vector<size_t> &need_update_submaps)
    {
        // tt.tic("draw");
        auto [rotated_cache, redraw_chunk_list] = draw_prepare(need_update_submaps);
        std::vector<std::function<void()>> rotate_tasks;
        for (auto &item : rotated_cache)
        {
            rotate_tasks.emplace_back(std::bind(&MapBuilder::rotated_generator, this, submaps[item.first].get(), &item.second));
        }
        multi_execute<N1>(rotate_tasks);
        //
        std::vector<std::function<void()>> draw_tasks;
        draw_tasks.reserve(redraw_chunk_list.size());
        for (const auto &chunk_index : redraw_chunk_list)
        {
            draw_tasks.emplace_back(std::bind(&MapBuilder::layer_generator, this, chunk_index, &rotated_cache));
        }
        multi_execute<N2>(draw_tasks);
        // tt.toc("draw");
        // chunk_map_->update(redraw_chunk_list);
        return redraw_chunk_list;
    }

    virtual void saveMap(const std::string& path) = 0;

    std::unordered_map<size_t, std::shared_ptr<SubmapInfo>> submaps;

// protected:
    ChunkMap::Ptr chunk_map_;

    struct RotatedCache
    {
        cv::Mat1b observe;
        cv::Mat1f value;
        cv::Mat1f alpha;
        cv::Mat1f elevation;
        cv::Mat1f elevation_sigma;
        // cv::Mat_<int32_t> obstacle_info;
        ObstacleTypeDesc::Type obstacle_info;
        // cv::Mat3b image;
        // cv::Mat1b image_observe;
    };

    MapBuilder() {}
    virtual void rotated_generator(const SubmapInfo *const submap, RotatedCache *const cache) = 0;
    virtual void layer_generator(const ChunkMap::Index &chunk_index, std::map<size_t, RotatedCache> const *const rotated_cache) = 0;
    virtual std::tuple<std::map<size_t, RotatedCache>, std::set<ChunkMap::Index>> draw_prepare(std::vector<size_t> &need_update_submaps) = 0;
};

#endif
