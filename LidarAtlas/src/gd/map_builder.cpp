#include "map_builder.h"

#include <any>
#include <set>
#include <fstream>

struct ChunkLayerBlockInfo
{
    cv::Rect rect_chunk_local, rect_submap_local;
    size_t submap_index;
    // judge info
    double avg_sigma;
    double avg_height;
};

bool rect_intersect(const cv::Rect &srcA, const cv::Rect &srcB, cv::Rect &dstA, cv::Rect &dstB)
{
    cv::Point tl{cv::max(srcA.tl().x, srcB.tl().x), cv::max(srcA.tl().y, srcB.tl().y)};
    cv::Point br{cv::min(srcA.br().x, srcB.br().x), cv::min(srcA.br().y, srcB.br().y)};
    if (tl.x >= br.x || tl.y >= br.y)
        return false;
    dstA = cv::Rect(tl - srcA.tl(), br - srcA.tl());
    dstB = cv::Rect(tl - srcB.tl(), br - srcB.tl());
    return true;
}

#ifdef OBS_PROB
void obstacle_prob_merge(ObstacleTypeDesc::Type &&dst, const ObstacleTypeDesc::Type &src)
{
    for (int r = 0; r < dst.rows; r++)
        for (int c = 0; c < dst.cols; c++)
        {
            auto &src_elem = ObstacleTypeDesc::wrapper(src(r, c));
            auto &dst_elem = ObstacleTypeDesc::wrapper(dst(r, c));
            for (int n = 0; n < obstacle_segments; n++)
            {
                dst_elem(n * obstacle_bits, obstacle_bits) = ObsProbMat[(uint16_t)dst_elem(n * obstacle_bits, obstacle_bits)][(uint16_t)src_elem(n * obstacle_bits, obstacle_bits)];
            }
        }
}
#endif

cv::Mat1b obstacle_unzero_mask(ObstacleTypeDesc::Type &&dst)
{
    std::vector<cv::Mat1b> layers;
    cv::split(dst, layers);
    cv::Mat1b ret(dst.size(), 0);
    for (const auto &l : layers)
        ret |= (l != 0);
    return ret;
}

class MapBuilderImpl : public MapBuilder
{
    MapBuilderConfig config_;
    cv::Mat1f precalced_sigma;
    cv::Mat1f precalced_x_offset, precalced_y_offset;
    // std::shared_ptr<ImageLandWrapper> imageLandWrapper;
    // ChunkMap::Ptr chunk_map_;
    uint32_t chunk_size;

    std::tuple<cv::Mat1f, cv::Mat1f, cv::Mat1f> makeRemap(const Eigen::Matrix3f &rotation)
    {
        Eigen::Matrix3f rinv = rotation.inverse();
        cv::Mat1f z = -(precalced_x_offset * rinv(2, 0) + precalced_y_offset * rinv(2, 1)) / rinv(2, 2);
        cv::Mat1f map_x = precalced_x_offset * rinv(0, 0) + precalced_y_offset * rinv(0, 1) + z * rinv(0, 2) + config_.submap_radius;
        cv::Mat1f map_y = precalced_x_offset * rinv(1, 0) + precalced_y_offset * rinv(1, 1) + z * rinv(1, 2) + config_.submap_radius;
        return {map_x / config_.resolution, map_y / config_.resolution, z};
    }

    // std::vector<std::shared_ptr<LocalmapInfo>> submaps;

    virtual void rotated_generator(const LocalmapInfo *const submap, RotatedCache *const cache) override
    {
        auto &detect = submap->detect;
        auto rotation = submap->pose.rotation();
#if 0
        // auto rotation = Eigen::AngleAxisd(submap->pose.rotation().eulerAngles(0, 1, 2).z(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Vector3d center = rotation * (Eigen::Vector3d{-1, -1, 0} * config_.submap_radius);
        Eigen::Vector3d x_axis = rotation * (Eigen::Vector3d{1, -1, 0} * config_.submap_radius);
        Eigen::Vector3d y_axis = rotation * (Eigen::Vector3d{-1, 1, 0} * config_.submap_radius);
        // Eigen::Vector3d c_axis = rotation * (Eigen::Vector3d{1, 1, 0} * config_.submap_radius);
        std::array<cv::Point2f, 3> from{cv::Point2f{-config_.submap_radius, -config_.submap_radius}, cv::Point2f{config_.submap_radius, -config_.submap_radius}, cv::Point2f{-config_.submap_radius, config_.submap_radius}}; //, cv::Point2f{config_.submap_radius, config_.submap_radius}};
        std::array<cv::Point2f, 3> to{cv::Point2f{(float)center.x(), (float)center.y()}, cv::Point2f{(float)x_axis.x(), (float)x_axis.y()}, cv::Point2f{(float)y_axis.x(), (float)y_axis.y()}};                               //, cv::Point2f{(float)c_axis.x(), (float)c_axis.y()}};
        for (int i = 0; i < 3; i++)
        {
            from[i] = (from[i] + cv::Point2f{config_.submap_radius + 0.5, config_.submap_radius + 0.5}) / config_.resolution;
            to[i] = (to[i] + cv::Point2f{config_.submap_radius + 0.5, config_.submap_radius + 0.5}) / config_.resolution;
        }
        cv::Mat affine_mat = cv::getAffineTransform(from, to);
        // cv::Mat affine_mat = cv::getPerspectiveTransform(from, to);
        //
        cv::Mat temp;
        cv::warpAffine(detect->map, temp, affine_mat, detect->map.size(), cv::INTER_NEAREST);
        temp.convertTo(temp, CV_32F);
        cv::Mat1f channels[3];
        cv::split(temp, channels);
        cache->observe = channels[2] != 0;
        cache->value = channels[0];
        cache->alpha = channels[1] / 255;
        Eigen::Matrix4d pose_mat = submap->pose.matrix();
        cv::Mat1f elevation = pose_mat(2, 0) * precalced_x_offset + pose_mat(2, 1) * precalced_y_offset + pose_mat(2, 3);
        cv::warpAffine(elevation, cache->elevation, affine_mat, elevation.size());
        float cos_theta = (submap->pose.rotation() * Eigen::Vector3d::UnitZ()).cwiseAbs().z();
        cv::warpAffine(precalced_sigma * cos_theta, cache->elevation_sigma, affine_mat, precalced_sigma.size());
        cv::warpAffine(detect->obstacle_info, cache->obstacle_info, affine_mat, detect->obstacle_info.size(), cv::INTER_NEAREST);
#endif
        auto remaps = makeRemap(rotation.cast<float>());
        cv::Mat temp;
        cv::remap(detect->map, temp, std::get<0>(remaps), std::get<1>(remaps), cv::INTER_NEAREST);
        temp.convertTo(temp, CV_32F);
        cv::Mat1f channels[3];
        cv::split(temp, channels);
        cache->observe = channels[2] != 0;
        cache->value = channels[0];
        cache->alpha = channels[1] / 255;
        Eigen::Matrix4d pose_mat = submap->pose.matrix();
        // cache->elevation = std::get<2>(remaps) + pose_mat(2, 3);
        cv::Mat1f elevation = pose_mat(2, 0) * precalced_x_offset + pose_mat(2, 1) * precalced_y_offset + pose_mat(2, 3);
        cv::remap(elevation, cache->elevation, std::get<0>(remaps), std::get<1>(remaps), cv::INTER_NEAREST);
        float cos_theta = (submap->pose.rotation() * Eigen::Vector3d::UnitZ()).cwiseAbs().z();
        cv::remap(precalced_sigma * cos_theta, cache->elevation_sigma, std::get<0>(remaps), std::get<1>(remaps), cv::INTER_NEAREST);
        cv::remap(detect->obstacle_info, cache->obstacle_info, std::get<0>(remaps), std::get<1>(remaps), cv::INTER_NEAREST);
#if 0
        cv::warpPerspective(detect->landImage, cache->image, affine_mat, detect->landImage.size(), cv::INTER_NEAREST);
        cv::warpPerspective(imageLandWrapper->observe, cache->image_observe, affine_mat, imageLandWrapper->observe.size(), cv::INTER_NEAREST);
#endif
    }

    virtual void layer_generator(const ChunkMap::Index &chunk_index, std::map<size_t, RotatedCache> const *const rotated_cache) override
    {
        const auto &chunk = chunk_map_->at(chunk_index);
        std::vector<size_t> submap_list;
        const std::set<size_t> *meta = std::any_cast<const std::set<size_t>>(&chunk_map_->at(chunk_index).meta);
        submap_list.assign(meta->begin(), meta->end());
        std::sort(submap_list.begin(), submap_list.end());
        //
        cv::Rect rect_chunk{int(chunk_index.x * chunk_size), int(chunk_index.y * chunk_size), int(chunk_size), int(chunk_size)};

        std::vector<ChunkLayer> layers;
        std::vector<ChunkLayerBlockInfo> infos;
        for (const auto &submap_index : submap_list)
        {
            const auto &submap = submaps.at(submap_index);
            cv::Rect rect_chunk_local, rect_submap_local;
            if (!rect_intersect(rect_chunk, submap->global_rect, rect_chunk_local, rect_submap_local))
                continue;
            const auto &cache = rotated_cache->at(submap_index);

            int count = cv::sum(cache.observe(rect_submap_local) / 255)[0];
            // WARN:
            if (count == 0)
            {
                count = rect_submap_local.area();
                if (cv::sum(obstacle_unzero_mask(cache.obstacle_info(rect_submap_local)))[0] == 0) // && cv::sum(cache.image_observe(rect_submap_local))[0] == 0)
                    continue;
                // continue;
            }
            cv::Mat1f temp = cv::Mat1f::zeros(rect_submap_local.size());
            cache.elevation_sigma(rect_submap_local).copyTo(temp, cache.observe(rect_submap_local));
            double avg_sigma = cv::sum(temp)[0] / count;
            // double avg_sigma;
            // cv::minMaxIdx(temp, &avg_sigma,nullptr,nullptr,nullptr,cache.observe(rect_submap_local));
            temp.setTo(0);
            cache.elevation(rect_submap_local).copyTo(temp, cache.observe(rect_submap_local));
            double avg_height = cv::sum(temp)[0] / count;

            // calc judge value
            ChunkLayerBlockInfo info;
            info.rect_chunk_local = rect_chunk_local;
            info.rect_submap_local = rect_submap_local;
            info.submap_index = submap_index;
            info.avg_sigma = avg_sigma;
            info.avg_height = avg_height;
            infos.push_back(info);
        }
        if (infos.size() > 0)
        {
            std::vector<size_t> peak_list; // save the index of peak's submap_index
            bool pre_down = true;
            for (size_t i = 0; i < infos.size() - 1; i++)
            {
                if (infos[i + 1].avg_sigma - infos[i].avg_sigma > 0)
                {
                    if (pre_down)
                    {
                        peak_list.push_back(i);
                    }
                    pre_down = false;
                }
                else
                {
                    pre_down = true;
                }
            }
            if (pre_down)
            {
                peak_list.push_back(infos.size() - 1);
            }
            std::sort(peak_list.begin(), peak_list.end(), [&](auto &a, auto &b)
                      { return infos[a].avg_height < infos[b].avg_height; });
            size_t peak_id = 0;
            std::vector<std::pair<size_t, size_t>> peak_check; // alloc peak index, (submap_index, peak_index)
            peak_check.push_back({peak_list[0], peak_id});
            // check the peaks need to be merged or not
            for (size_t i = 1; i < peak_list.size(); i++)
            {
                const auto &a = infos[peak_list[i]];
                const auto &b = infos[peak_list[i - 1]];
                constexpr int N = 30;
                Eigen::Array<double, N + 1, 1> p_array = Eigen::Array<double, N + 1, 1>::LinSpaced(N + 1, std::min(a.avg_height, b.avg_height), std::max(a.avg_height, b.avg_height));
                p_array = (-(p_array - a.avg_height).square() / a.avg_sigma / 2).exp() / std::sqrt(a.avg_sigma) + (-(p_array - b.avg_height).square() / b.avg_sigma / 2).exp() / std::sqrt(b.avg_sigma);
                double p_saddle = p_array.minCoeff();
                double p_submax = std::min(p_array(0, 0), p_array(N, 0));
#ifndef NO_MULTI_LAYER
                if (p_saddle / p_submax < 0.6)
                {
                    peak_id += 1;
                }
#endif
                peak_check.push_back({peak_list[i], peak_id});
            }
            std::sort(peak_check.begin(), peak_check.end(), [](auto &a, auto &b)
                      { return a.first < b.first; });
            for (size_t i = 0; i <= peak_id; i++)
            {
                layers.push_back(chunk.createLayer());
            }
            size_t current_peak = 0;
            for (size_t i = 0; i < infos.size(); i++)
            {
                if (current_peak < peak_check.size() && i >= peak_check[current_peak].first)
                {
                    current_peak++;
                }
                size_t peak = 0;
                if (current_peak == 0)
                {
                    peak = peak_check.front().second;
                }
                else if (current_peak >= peak_check.size())
                {
                    peak = peak_check.back().second;
                }
                else
                {
                    const auto &a = infos[peak_check[current_peak - 1].first];
                    const auto &b = infos[peak_check[current_peak].first];
                    if (std::abs(infos[i].avg_height - a.avg_height) < std::abs(infos[i].avg_height - b.avg_height))
                    {
                        peak = peak_check[current_peak - 1].second;
                    }
                    else
                    {
                        peak = peak_check[current_peak].second;
                    }
                }
                //
                auto &layer = layers[peak];
                const auto &rect_chunk_local = infos[i].rect_chunk_local;
                const auto &rect_submap_local = infos[i].rect_submap_local;
                const auto &cache = rotated_cache->at(infos[i].submap_index);

                cv::Mat1f(cv::min(layer.occupancy(rect_chunk_local).mul(1.0f - cache.alpha(rect_submap_local)) + cache.value(rect_submap_local), 255.0)).copyTo(layer.occupancy(rect_chunk_local), cache.observe(rect_submap_local));
                layer.observe(rect_chunk_local) |= cache.observe(rect_submap_local);

                // const auto obs_unzero = obstacle_unzero_mask(cache.obstacle_info(rect_submap_local));
                const auto obs_unzero = obstacle_unzero_mask(cache.obstacle_info(rect_submap_local)); // | cache.image_observe(rect_submap_local) != 0;

                cv::Mat1f new_sigma = layer.elevation_sigma(rect_chunk_local).mul(cache.elevation_sigma(rect_submap_local)).mul(1 / (layer.elevation_sigma(rect_chunk_local) + cache.elevation_sigma(rect_submap_local)));
                cv::Mat1f new_elevation = (layer.elevation(rect_chunk_local).mul(cache.elevation_sigma(rect_submap_local)) + cache.elevation(rect_submap_local).mul(layer.elevation_sigma(rect_chunk_local))).mul(1 / (layer.elevation_sigma(rect_chunk_local) + cache.elevation_sigma(rect_submap_local)));
                cv::Mat mix_mask = (layer.elevation_sigma(rect_chunk_local) > 0) & (cache.observe(rect_submap_local) | obs_unzero);
                cv::Mat copy_mask = (layer.elevation_sigma(rect_chunk_local) < 0) & (cache.observe(rect_submap_local) | obs_unzero);
                new_elevation.copyTo(layer.elevation(rect_chunk_local), mix_mask);
                cache.elevation(rect_submap_local).copyTo(layer.elevation(rect_chunk_local), copy_mask);
                new_sigma.copyTo(layer.elevation_sigma(rect_chunk_local), mix_mask);
                cache.elevation_sigma(rect_submap_local).copyTo(layer.elevation_sigma(rect_chunk_local), copy_mask);
#ifndef OBS_PROB
                layer.obstacle_info(rect_chunk_local) |= cache.obstacle_info(rect_submap_local);
#else
                obstacle_prob_merge(layer.obstacle_info(rect_chunk_local), cache.obstacle_info(rect_submap_local));
#endif
#if 0
                // cam-layer
                layer.image_observe(rect_chunk_local) |= cache.image_observe(rect_submap_local);
                cv::Mat1f new_img_sigma = layer.image_sigma(rect_chunk_local).mul(cache.elevation_sigma(rect_submap_local)).mul(1 / (layer.image_sigma(rect_chunk_local) + cache.elevation_sigma(rect_submap_local)));
                // TODO: for each layer
                cv::Mat3f ltemp, ctemp;
                cv::Mat1f lchans[3], cchans[3];
                cv::Mat1b ochans[3];
                layer.image.convertTo(ltemp, CV_32FC3);
                cache.image.convertTo(ctemp, CV_32FC3);
                cv::split(ltemp, lchans);
                cv::split(ctemp, cchans);
                for (int c = 0; c < 3; c++)
                {
                    cv::Mat1f temp = (lchans[c](rect_chunk_local).mul(cache.elevation_sigma(rect_submap_local)) + cchans[c](rect_submap_local).mul(layer.image_sigma(rect_chunk_local))).mul(1 / (layer.image_sigma(rect_chunk_local) + cache.elevation_sigma(rect_submap_local)));
                    temp.convertTo(ochans[c], CV_8U);
                    // cv::Mat1f new_image = (layer.image(rect_chunk_local).mul(cache.elevation_sigma(rect_submap_local)) + cache.image(rect_submap_local).mul(layer.image_sigma(rect_chunk_local))).mul(1 / (layer.image_sigma(rect_chunk_local) + cache.elevation_sigma(rect_submap_local)));
                }
                cv::Mat3b new_image;
                cv::merge(ochans, 3, new_image);
                cv::Mat img_mix_mask = (layer.image_sigma(rect_chunk_local) > 0) & (cache.image_observe(rect_submap_local));
                cv::Mat img_copy_mask = (layer.image_sigma(rect_chunk_local) < 0) & (cache.image_observe(rect_submap_local));
                new_image.copyTo(layer.image(rect_chunk_local), img_mix_mask);
                cache.image(rect_submap_local).copyTo(layer.image(rect_chunk_local), img_copy_mask);
                new_img_sigma.copyTo(layer.image_sigma(rect_chunk_local), img_mix_mask);
                cache.elevation_sigma(rect_submap_local).copyTo(layer.image_sigma(rect_chunk_local), img_copy_mask);
#endif
            }
        }
        (*chunk_map_)[chunk_index].setLayers(layers);
    }

    // template <size_t N1, size_t N2>
    // void drawCallback_multi(std::vector<size_t> &need_update_submaps)


    virtual std::tuple<std::map<size_t, RotatedCache>, std::set<ChunkMap::Index>> draw_prepare(std::vector<size_t> &need_update_submaps) override
    {
        // ROS_INFO("drawCallback_multi");
        if (submaps.empty())
            return {};
        std::vector<size_t> submap_index_list;
        // ROS_INFO_STREAM(need_update_submaps.size());
        need_update_submaps.swap(submap_index_list);
        // ROS_INFO_STREAM(submap_index_list.size());
        if (submap_index_list.empty())
            return {};
        std::set<ChunkMap::Index> redraw_chunk_list;
        for (const auto &submap_index : submap_index_list)
        {
            auto &submap = submaps.at(submap_index);
            auto &detect = submap->detect;
            if (!submap->inited)
                continue;
            if (submap->has_old_pose)
            {
                Eigen::Vector2d position_eigen = submap->old_pose.translation().block<2, 1>(0, 0);
                cv::Point base_position = cv::Point{std::round((position_eigen.x() - config_.submap_radius) / config_.resolution), std::round((position_eigen.y() - config_.submap_radius) / config_.resolution)};
                cv::Rect local_rect(base_position.x, base_position.y, detect->map.cols, detect->map.rows);
                cv::Point tl = local_rect.tl();
                cv::Point br = local_rect.br();
                ChunkMap::Index chunk_tl{int64_t(std::floor(tl.x / (float)chunk_size)), int64_t(std::floor(tl.y / (float)chunk_size))};
                ChunkMap::Index chunk_br{int64_t(std::ceil(br.x / (float)chunk_size)), int64_t(std::ceil(br.y / (float)chunk_size))};
                for (int y = chunk_tl.y; y <= chunk_br.y; y++)
                {
                    for (int x = chunk_tl.x; x <= chunk_br.x; x++)
                    {
                        ChunkMap::Index chunk_index{x, y};
                        redraw_chunk_list.insert(chunk_index);
                        std::set<size_t> *meta = std::any_cast<std::set<size_t>>(&(*chunk_map_)[chunk_index].meta);
                        if (meta != nullptr)
                        {
                            meta->erase(submap_index);
                        }
                    }
                }
            }
            Eigen::Vector2d position_eigen = submap->pose.translation().block<2, 1>(0, 0);
            cv::Point base_position = cv::Point{std::round((position_eigen.x() - config_.submap_radius) / config_.resolution), std::round((position_eigen.y() - config_.submap_radius) / config_.resolution)};
            cv::Rect local_rect(base_position.x, base_position.y, detect->map.cols, detect->map.rows);
            cv::Point tl = local_rect.tl();
            cv::Point br = local_rect.br();
            ChunkMap::Index chunk_tl{int64_t(std::floor(tl.x / (float)chunk_size)), int64_t(std::floor(tl.y / (float)chunk_size))};
            ChunkMap::Index chunk_br{int64_t(std::ceil(br.x / (float)chunk_size)), int64_t(std::ceil(br.y / (float)chunk_size))};
            submap->global_rect = local_rect;
            for (int y = chunk_tl.y; y <= chunk_br.y; y++)
            {
                for (int x = chunk_tl.x; x <= chunk_br.x; x++)
                {
                    ChunkMap::Index chunk_index{x, y};
                    redraw_chunk_list.insert(chunk_index);
                    std::set<size_t> *meta = std::any_cast<std::set<size_t>>(&(*chunk_map_)[chunk_index].meta);
                    if (meta == nullptr)
                    {
                        std::set<size_t> set;
                        set.insert(submap_index);
                        (*chunk_map_)[chunk_index].meta = set;
                    }
                    else
                    {
                        meta->insert(submap_index);
                    }
                }
            }
            submap->old_pose = submap->pose;
            submap->has_old_pose = true;
        }
        std::map<size_t, RotatedCache> rotated_cache;
        for (const auto &chunk_index : redraw_chunk_list)
        {
            std::vector<size_t> submap_list;
            std::set<size_t> *meta = std::any_cast<std::set<size_t>>(&(*chunk_map_)[chunk_index].meta);
            submap_list.assign(meta->begin(), meta->end());
            std::sort(submap_list.begin(), submap_list.end());
            //
            cv::Rect rect_chunk{int(chunk_index.x * chunk_size), int(chunk_index.y * chunk_size), int(chunk_size), int(chunk_size)};

            std::vector<ChunkLayer> layers;
            std::vector<ChunkLayerBlockInfo> infos;
            for (const auto &submap_index : submap_list)
            {
                const auto &submap = submaps.at(submap_index);
                cv::Rect rect_chunk_local, rect_submap_local;
                if (!rect_intersect(rect_chunk, submap->global_rect, rect_chunk_local, rect_submap_local))
                    continue;
                if (rotated_cache.find(submap_index) == rotated_cache.end())
                    rotated_cache[submap_index] = {};
            }
        }
        //
        return {std::move(rotated_cache), std::move(redraw_chunk_list)};
    }

public:
    // MapBuilderImpl(MapBuilderConfig config, std::shared_ptr<ImageLandWrapper> wrapper, ChunkMap::Ptr chunk_map)
    MapBuilderImpl(MapBuilderConfig config, ChunkMap::Ptr chunk_map)
        : config_(config)//, imageLandWrapper(wrapper)
    {
        chunk_map_ = chunk_map;
        constexpr float sigma_k = 1;
        constexpr float sigma_alpha = 5 * M_PI / 180.0;
        constexpr float sigma_base = 0.1;
        // constexpr float sigma_alpha = 2 * M_PI / 180.0;
        // constexpr float sigma_base = 0.03;
        // constexpr float sigma_alpha = 0.5 * M_PI / 180.0;
        // constexpr float sigma_base = 0.001;
        cv::Size size{std::round(config_.submap_radius * 2 / config_.resolution), std::round(config_.submap_radius * 2 / config_.resolution)};
        precalced_sigma = cv::Mat1f::zeros(size);
        precalced_x_offset = cv::Mat1f::zeros(size);
        precalced_y_offset = cv::Mat1f::zeros(size);
        Eigen::Vector2f center{config_.submap_radius, config_.submap_radius};
        for (int i = 0; i < precalced_sigma.rows; i++)
        {
            for (int j = 0; j < precalced_sigma.cols; j++)
            {
                Eigen::Vector2f pos{j * config_.resolution, i * config_.resolution};
                pos -= center;
                precalced_x_offset(i, j) = pos.x();
                precalced_y_offset(i, j) = pos.y();
                float sigma = sigma_k * pos.norm() * std::tan(sigma_alpha) + sigma_base;
                precalced_sigma(i, j) = sigma * sigma;
            }
        }
        chunk_size = chunk_map_->chunkSize();
    };

    virtual void saveMap(const std::string & path) override
    {
        for (const auto &submap : submaps)
        {
            if (!submap.second->inited)
                continue;
            ChunkMap::KeyFrameData frame;
            frame.feature = submap.second->detect->feature;
            frame.pose = submap.second->pose.matrix().cast<float>();
            chunk_map_->key_frames.push_back(frame);
        }

        std::ofstream ofs(path, std::ios::binary);
        ChunkMap::save(ofs, chunk_map_);
        ofs.close();
    }
};

// MapBuilder::Ptr MapBuilder::create(MapBuilderConfig config, std::shared_ptr<ImageLandWrapper> wrapper, ChunkMap::Ptr chunk_map)
MapBuilder::Ptr MapBuilder::create(MapBuilderConfig config, ChunkMap::Ptr chunk_map)
{
    // return std::make_shared<MapBuilderImpl>(config, wrapper, chunk_map);
    return std::make_shared<MapBuilderImpl>(config, chunk_map);
}
