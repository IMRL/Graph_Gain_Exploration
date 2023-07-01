#ifndef _GLOBALMAP_BUILDER_H_
#define _GLOBALMAP_BUILDER_H_

#include "map_builder.h"

class GlobalmapBuilder
{
public:
    using Ptr = std::shared_ptr<GlobalmapBuilder>;

    virtual ~GlobalmapBuilder() {}

    // static Ptr create(MapBuilderConfig config, std::shared_ptr<ImageLandWrapper> wrapper, ChunkMap::Ptr chunk_map);
    static Ptr create(MapBuilderConfig config, ChunkMap::Ptr chunk_map);

    struct LocalmapIndex
    {
        size_t submap_index;
        int64_t localmap_index;
    };

    template <size_t N1, size_t N2>
    std::set<ChunkMap::Index> draw_multi(std::vector<LocalmapIndex> &need_update_localmaps)
    {
        std::map<size_t, std::vector<size_t>> need_update_lists;
        std::set<size_t> full_update;
        for (const auto &it : need_update_localmaps)
        {
            if (it.localmap_index < 0)
            {
                full_update.insert(it.submap_index);
                continue;
            }
            if (need_update_lists.find(it.submap_index) == need_update_lists.end())
            {
                need_update_lists[it.submap_index] = {};
            }
            need_update_lists[it.submap_index].push_back(it.localmap_index);
        }
        need_update_localmaps.clear();
        std::map<size_t, std::set<ChunkMap::Index>> submap_update_area;
        for (auto &it : need_update_lists)
        {
            submap_update_area[it.first] = submaps_.at(it.first).builder->draw_multi<N1, N2>(it.second);
        }
        for (auto &it : full_update)
        {
            if (submap_update_area.find(it) == submap_update_area.end())
            {
                submap_update_area[it] = std::set<ChunkMap::Index>{};
            }
            for (const auto &key : submaps_.at(it).map->keys())
            {
                submap_update_area[it].insert(key);
            }
        }

        auto redraw_chunk_list = draw_prepare(submap_update_area);

        for (auto &it : full_update)
        {
            submaps_.at(it).old_pose = submaps_.at(it).base_pose;
        }

        std::vector<std::function<void()>> draw_tasks;
        draw_tasks.reserve(redraw_chunk_list.size());
        for (const auto &chunk_index : redraw_chunk_list)
        {
            draw_tasks.emplace_back(std::bind(&GlobalmapBuilder::layer_generator, this, chunk_index));
        }
        multi_execute<N2>(draw_tasks);
        return redraw_chunk_list;
    }

    virtual void saveMap(const std::string &path) = 0;

    void freezeSubmap(size_t submap)
    {
        if (submaps_.find(submap) != submaps_.end())
        {
            submaps_.at(submap).builder->submaps.clear();
        }
    }

    virtual void addLocalmap(LocalmapIndex index, std::shared_ptr<LocalmapInfo> localmap) = 0;
    std::shared_ptr<LocalmapInfo> getLocalmap(LocalmapIndex index) const
    {
        if (submaps_.find(index.submap_index) == submaps_.end())
            return nullptr;
        const auto &builder = submaps_.at(index.submap_index).builder;
        if (builder->submaps.find(index.localmap_index) == builder->submaps.end())
            return nullptr;
        return builder->submaps.at(index.localmap_index);
    }

    MapBuilder::Ptr getSubmap(size_t index) const
    {
        if (submaps_.find(index) == submaps_.end())
            return nullptr;
        return submaps_.at(index).builder;
    }
    bool hasSubmap(size_t index) const
    {
        return submaps_.find(index) != submaps_.end();
    }
    Eigen::Matrix4f &submapPose(size_t index)
    {
        return submaps_.at(index).base_pose;
    }
    Eigen::Matrix4f &submapOldPose(size_t index)
    {
        return submaps_.at(index).old_pose;
    }

protected:
    struct SubmapInfo
    {
        ChunkMap::Ptr map;
        MapBuilder::Ptr builder;
        Eigen::Matrix4f base_pose;
        Eigen::Matrix4f old_pose;
    };

    ChunkMap::Ptr chunk_map_;
    std::unordered_map<size_t, SubmapInfo> submaps_;

    virtual std::set<ChunkMap::Index> draw_prepare(const std::map<size_t, std::set<ChunkMap::Index>> &submap_update_area) = 0;
    virtual void layer_generator(const ChunkMap::Index &chunk_index) = 0;
};

#endif
