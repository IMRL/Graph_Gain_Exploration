#include "globalmap_builder.h"

#ifdef OBS_PROB
extern void obstacle_prob_merge(ObstacleTypeDesc::Type &&dst, const ObstacleTypeDesc::Type &src);
#endif

extern cv::Mat1b obstacle_unzero_mask(ObstacleTypeDesc::Type &&dst);

class GlobalmapBuilderImpl : public GlobalmapBuilder
{
    MapBuilderConfig config_;
    Eigen::Matrix4f base_rect;

    cv::Mat1f precalced_x_offset, precalced_y_offset;

    std::set<ChunkMap::Index> draw_prepare(const std::map<size_t, std::set<ChunkMap::Index>> &submap_update_area) override
    {
        std::set<ChunkMap::Index> ret;
        const float chunk_base = chunk_map_->chunkBase();
        const float resolution = chunk_map_->resolution();

        for (const auto &it : submap_update_area)
        {
            auto pose = submaps_.at(it.first).base_pose;
            if (pose == Eigen::Matrix4f::Zero())
                continue;
            auto oldPose = submaps_.at(it.first).old_pose;
            bool base_move = oldPose != pose;
            for (const auto &index : it.second)
            {
                Eigen::Matrix4f points = base_rect;
                points.row(0).array() += index.x * chunk_base;
                points.row(1).array() += index.y * chunk_base;
                Eigen::Matrix4f global_pose = pose * points;
                for (int64_t x = std::floor(global_pose.row(0).minCoeff() / chunk_base); x <= std::floor(global_pose.row(0).maxCoeff() / chunk_base); x++)
                    for (int64_t y = std::floor(global_pose.row(1).minCoeff() / chunk_base); y <= std::floor(global_pose.row(1).maxCoeff() / chunk_base); y++)
                    {
                        ret.insert({x, y});
                    }
                if (base_move)
                {
                    Eigen::Matrix4f global_pose = oldPose * points;
                    for (int64_t x = std::floor(global_pose.row(0).minCoeff() / chunk_base); x <= std::floor(global_pose.row(0).maxCoeff() / chunk_base); x++)
                        for (int64_t y = std::floor(global_pose.row(1).minCoeff() / chunk_base); y <= std::floor(global_pose.row(1).maxCoeff() / chunk_base); y++)
                        {
                            ret.insert({x, y});
                        }
                }
            }
        }
        for (const auto &it : ret)
        {
            (*chunk_map_)[it].meta = std::set<size_t>{};
            for (const auto &submap : submaps_)
            {
                auto pose = submap.second.base_pose;
                if (pose == Eigen::Matrix4f::Zero())
                    continue;
                Eigen::Matrix4f points = base_rect;
                points.row(0).array() += it.x * chunk_base;
                points.row(1).array() += it.y * chunk_base;
                Eigen::Matrix4f global_pose = pose.inverse() * points;
                bool tested = false;
                for (int64_t x = std::floor(global_pose.row(0).minCoeff() / chunk_base); x <= std::floor(global_pose.row(0).maxCoeff() / chunk_base); x++)
                {
                    for (int64_t y = std::floor(global_pose.row(1).minCoeff() / chunk_base); y <= std::floor(global_pose.row(1).maxCoeff() / chunk_base); y++)
                    {
                        if (submap.second.map->has({x, y}))
                        {
                            std::any_cast<std::set<size_t>>(&(*chunk_map_)[it].meta)->insert(submap.first);
                            tested = true;
                            break;
                        }
                    }
                    if (tested)
                        break;
                }
            }
        }
        return ret;
    }

    std::vector<ChunkLayer> makeSubmapLayers(ChunkMap::Index tl, ChunkMap::Index br, const SubmapInfo &submap)
    {
        const size_t chunk_size = chunk_map_->chunkSize();
        std::vector<ChunkLayer> ret;
        const auto createLayer = [&]()
        {
            cv::Size size{chunk_size * (br.x - tl.x + 1), chunk_size * (br.y - tl.y + 1)};
            ChunkLayer layer;
            layer.observe = cv::Mat1b::zeros(size);
            layer.occupancy = cv::Mat1f::ones(size) * 128;
            layer.elevation = cv::Mat1f::zeros(size);
            layer.elevation_sigma = -cv::Mat1f::ones(size);
            layer.obstacle_info = ObstacleTypeDesc_create(chunk_map_->obstacle_segments, chunk_map_->obstacle_bits, size);
            return layer;
        };
        for (int64_t x = tl.x; x <= br.x; x++)
            for (int64_t y = tl.y; y <= br.y; y++)
            {
                if (!submap.map->has({x, y}))
                    continue;
                const auto &chunk = submap.map->at({x, y});
                int base_layer = 0;
                for (const auto &l : chunk.getLayers())
                {
                    int selected = -1;
                    for (int i = base_layer; i < ret.size(); i++)
                    {
                        if (x - tl.x > 0)
                        {
                            bool failed = false;
                            for (int step = 0; step < chunk_size; step++)
                            {
                                if (ret[i].observe(step, chunk_size - 1) && l.observe(step, 0))
                                {
                                    float offset = ret[i].elevation(step, chunk_size - 1) - l.elevation(step, 0);
                                    if (std::abs(offset) > 0.2)
                                    {
                                        failed = true;
                                        break;
                                    }
                                }
                            }
                            if (failed)
                                continue;
                        }
                        if (y - tl.y > 0)
                        {
                            bool failed = false;
                            for (int step = 0; step < chunk_size; step++)
                            {
                                if (ret[i].observe(chunk_size - 1, step) && l.observe(0, step))
                                {
                                    float offset = ret[i].elevation(chunk_size - 1, step) - l.elevation(0, step);
                                    if (std::abs(offset) > 0.2)
                                    {
                                        failed = true;
                                        break;
                                    }
                                }
                            }
                            if (failed)
                                continue;
                        }
                        selected = i;
                        break;
                    }
                    if (selected < 0)
                    {
                        selected = ret.size();
                        ret.push_back(createLayer());
                    }
                    cv::Rect rect{(x - tl.x) * chunk_size, (y - tl.y) * chunk_size, chunk_size, chunk_size};
                    l.observe.copyTo(ret[selected].observe(rect));
                    l.occupancy.copyTo(ret[selected].occupancy(rect));
                    l.elevation.copyTo(ret[selected].elevation(rect));
                    l.elevation_sigma.copyTo(ret[selected].elevation_sigma(rect));
                    l.obstacle_info.copyTo(ret[selected].obstacle_info(rect));
                }
            }
        return ret;
    }

    std::vector<ChunkLayer> remap(const ChunkMap::Index &curIndex, const std::vector<ChunkLayer> &concatedData, const Eigen::Matrix4f &transform, const ChunkMap::Index &tl)
    {
        const size_t chunk_size = chunk_map_->chunkSize();
        const float chunk_base = chunk_map_->chunkBase();
        const float resolution = chunk_map_->resolution();

        Eigen::Matrix4f transformInv = transform.inverse();
        Eigen::Matrix3f rinv = transformInv.block<3, 3>(0, 0);
        cv::Mat1f z = -(precalced_x_offset * rinv(2, 0) + precalced_y_offset * rinv(2, 1)) / rinv(2, 2);
        cv::Mat1f map_x = precalced_x_offset * rinv(0, 0) + precalced_y_offset * rinv(0, 1) + z * rinv(0, 2); // + config_.submap_radius;
        cv::Mat1f map_y = precalced_x_offset * rinv(1, 0) + precalced_y_offset * rinv(1, 1) + z * rinv(1, 2); // + config_.submap_radius;

        Eigen::Vector4f curPose = transformInv * Eigen::Vector4f{curIndex.x * chunk_base, curIndex.y * chunk_base, 0, 1};
        map_x -= tl.x * chunk_base - curPose.x();
        map_y -= tl.y * chunk_base - curPose.y();

        map_x /= resolution;
        map_y /= resolution;

        std::vector<ChunkLayer> ret;
        const auto createLayer = [&]()
        {
            cv::Size size{chunk_size, chunk_size};
            ChunkLayer layer;
            layer.observe = cv::Mat1b::zeros(size);
            layer.occupancy = cv::Mat1f::ones(size) * 128;
            layer.elevation = cv::Mat1f::zeros(size);
            layer.elevation_sigma = -cv::Mat1f::ones(size);
            layer.obstacle_info = ObstacleTypeDesc_create(chunk_map_->obstacle_segments, chunk_map_->obstacle_bits, size);
            return layer;
        };
        for (const auto &l : concatedData)
        {
            auto newLayer = createLayer();
            cv::remap(l.observe, newLayer.observe, map_x, map_y, cv::INTER_NEAREST);
            if (cv::sum(newLayer.observe)[0] == 0)
                continue;
            cv::remap(l.occupancy, newLayer.occupancy, map_x, map_y, cv::INTER_NEAREST);

            auto elev = l.elevation.clone();
            for (int r = 0; r < l.elevation.rows; r++)
                for (int c = 0; c < l.elevation.cols; c++)
                {
                    elev(r, c) += transform(2, 0) * (tl.x * chunk_base + c * resolution) + transform(2, 1) * (tl.y * chunk_base + r * resolution) + transform(2, 3);
                }
            cv::remap(elev, newLayer.elevation, map_x, map_y, cv::INTER_NEAREST);
            cv::remap(l.elevation_sigma, newLayer.elevation_sigma, map_x, map_y, cv::INTER_NEAREST);
            cv::remap(l.occupancy, newLayer.occupancy, map_x, map_y, cv::INTER_NEAREST);
            ret.push_back(newLayer);
        }
        return ret;
    }

    virtual void layer_generator(const ChunkMap::Index &chunk_index) override
    {
        const float chunk_base = chunk_map_->chunkBase();
        const size_t chunk_size = chunk_map_->chunkSize();

        const auto &chunk = chunk_map_->at(chunk_index);
        std::vector<size_t> submap_list;
        const std::set<size_t> *meta = std::any_cast<const std::set<size_t>>(&chunk_map_->at(chunk_index).meta);
        submap_list.assign(meta->begin(), meta->end());
        std::sort(submap_list.begin(), submap_list.end());

        std::vector<ChunkLayer> preparedLayers;

        // step 1: from each submap, construct rotated patchs
        for (const auto &index : submap_list)
        {
            Eigen::Matrix4f points = base_rect;
            points.row(0).array() += chunk_index.x * chunk_base;
            points.row(1).array() += chunk_index.y * chunk_base;
            Eigen::Matrix4f submap_pose = submaps_[index].base_pose.inverse() * points;
            ChunkMap::Index tl{std::floor(submap_pose.row(0).minCoeff() / chunk_base), std::floor(submap_pose.row(1).minCoeff() / chunk_base)};
            ChunkMap::Index br{std::ceil(submap_pose.row(0).maxCoeff() / chunk_base), std::ceil(submap_pose.row(1).maxCoeff() / chunk_base)};

            auto layers = makeSubmapLayers(tl, br, submaps_[index]);
            auto cuttedLayers = remap(chunk_index, layers, submaps_[index].base_pose, tl);
            std::copy(cuttedLayers.begin(), cuttedLayers.end(), std::back_inserter(preparedLayers));
        }
        if (preparedLayers.size() == 0)
        {
            std::vector<ChunkLayer> layerData;
            (*chunk_map_)[chunk_index].setLayers(layerData);
            return;
        }
        // step 2: elevation analyse
        std::vector<std::tuple<size_t, float, float>> layers;
        for (int i = 0; i < preparedLayers.size(); i++)
        {
            const auto &it = preparedLayers[i];
            int count = cv::sum(it.observe / 255)[0];
            cv::Mat1f temp = cv::Mat1f::zeros(it.observe.size());
            it.elevation_sigma.copyTo(temp, it.observe);
            double avg_sigma = cv::sum(temp)[0] / count;
            temp.setTo(0);
            it.elevation.copyTo(temp, it.observe);
            double avg_height = cv::sum(temp)[0] / count;
            layers.push_back({i, avg_height, avg_sigma});
        }
        std::sort(layers.begin(), layers.end(), [&](auto &a, auto &b)
                  { return std::get<1>(a) < std::get<1>(b); });
        std::vector<std::vector<size_t>> peaks;
        peaks.push_back(std::vector<size_t>{std::get<0>(layers[0])});
        // check the peaks need to be merged or not
        for (size_t i = 1; i < layers.size(); i++)
        {
            const auto &a = layers[i];
            const auto &b = layers[i - 1];
            constexpr int N = 30;
            Eigen::Array<double, N + 1, 1> p_array = Eigen::Array<double, N + 1, 1>::LinSpaced(N + 1, std::min(std::get<1>(a), std::get<1>(b)), std::max(std::get<1>(a), std::get<1>(b)));
            p_array = (-(p_array - std::get<1>(a)).square() / std::get<2>(a) / 2).exp() / std::sqrt(std::get<2>(a)) + (-(p_array - std::get<1>(b)).square() / std::get<2>(b) / 2).exp() / std::sqrt(std::get<2>(b));
            double p_saddle = p_array.minCoeff();
            double p_submax = std::min(p_array(0, 0), p_array(N, 0));
#ifndef NO_MULTI_LAYER
            if (p_saddle / p_submax < 0.6)
            {
                peaks.push_back({});
            }
#endif
            peaks.back().push_back(std::get<0>(a));
        }
        // step 3: merge
        std::vector<ChunkLayer> layerData;
        for (auto &peak : peaks)
        {
            std::sort(peak.begin(), peak.end());
            layerData.push_back(chunk_map_->at(chunk_index).createLayer());
            for (const auto &it : peak)
            {
                auto &layer = layerData.back();
                auto &submap = preparedLayers[it];
                {
                    layer.occupancy = layer.occupancy + submap.occupancy - 128.0f;
                    layer.occupancy.setTo(1.0f, layer.occupancy < 1.0f);
                    layer.occupancy.setTo(255.0f, layer.occupancy > 255.0f);
                }
                layer.observe |= submap.observe;

                const auto obs_unzero = obstacle_unzero_mask(submap.obstacle_info);

                cv::Mat1f new_sigma = layer.elevation_sigma.mul(submap.elevation_sigma).mul(1 / (layer.elevation_sigma + submap.elevation_sigma));
                cv::Mat1f new_elevation = (layer.elevation.mul(submap.elevation_sigma) + submap.elevation.mul(layer.elevation_sigma)).mul(1 / (layer.elevation_sigma + submap.elevation_sigma));
                cv::Mat mix_mask = (layer.elevation_sigma > 0) & (submap.observe | obs_unzero);
                cv::Mat copy_mask = (layer.elevation_sigma < 0) & (submap.observe | obs_unzero);
                new_elevation.copyTo(layer.elevation, mix_mask);
                submap.elevation.copyTo(layer.elevation, copy_mask);
                new_sigma.copyTo(layer.elevation_sigma, mix_mask);
                submap.elevation_sigma.copyTo(layer.elevation_sigma, copy_mask);
#ifndef OBS_PROB
                layer.obstacle_info |= submap.obstacle_info;
#else
                obstacle_prob_merge(layer.obstacle_info, submap.obstacle_info);
#endif
            }
        }
        (*chunk_map_)[chunk_index].setLayers(layerData);
    }

public:
    GlobalmapBuilderImpl(MapBuilderConfig config, ChunkMap::Ptr chunk_map)
        : config_(config)
    {
        chunk_map_ = chunk_map;

        float chunk_base = chunk_map_->chunkBase();
        float resolution = chunk_map_->resolution();
        size_t chunk_size = chunk_map_->chunkSize();
        base_rect = Eigen::Matrix4f::Zero();
        base_rect.row(3).array() = 1.0f;
        base_rect(0, 1) = base_rect(0, 3) = base_rect(1, 2) = base_rect(1, 3) = chunk_base - resolution;

        cv::Size size{chunk_size, chunk_size};
        precalced_x_offset = cv::Mat1f::zeros(size);
        precalced_y_offset = cv::Mat1f::zeros(size);
        for (int i = 0; i < precalced_x_offset.rows; i++)
        {
            for (int j = 0; j < precalced_x_offset.cols; j++)
            {
                Eigen::Vector2f pos{j * config_.resolution, i * config_.resolution};
                precalced_x_offset(i, j) = pos.x();
                precalced_y_offset(i, j) = pos.y();
            }
        }
    }

    virtual void saveMap(const std::string &path) override
    {
        std::ofstream ofs(path, std::ios::binary);
        ChunkMap::save(ofs, chunk_map_);
        ofs.close();
    }

    virtual void addLocalmap(LocalmapIndex index, std::shared_ptr<LocalmapInfo> localmap) override
    {
        if (submaps_.find(index.submap_index) == submaps_.end())
        {
            auto sub_chunk = std::make_shared<ChunkMap>(ChunkMap::Config{
                .resolution = chunk_map_->resolution(),
                .chunk_base = chunk_map_->chunkBase(),
            });
            auto sub_builder = MapBuilder::create(config_, sub_chunk);
            submaps_[index.submap_index] = SubmapInfo{
                .map = sub_chunk,
                .builder = sub_builder,
                .base_pose = Eigen::Matrix4f::Zero(),
            };
        }
        submaps_[index.submap_index].builder->submaps[index.localmap_index] = localmap;
    }
};

GlobalmapBuilder::Ptr GlobalmapBuilder::create(MapBuilderConfig config, ChunkMap::Ptr chunk_map)
{
    // return std::make_shared<MapBuilderImpl>(config, wrapper, chunk_map);
    return std::make_shared<GlobalmapBuilderImpl>(config, chunk_map);
}
