#ifndef _CHUNK_MAP_CHUNKMAP_DISPLAY_H_
#define _CHUNK_MAP_CHUNKMAP_DISPLAY_H_

#include <mutex>
#include <map>
#include <string>

#include <ros/ros.h>
#include <ros/package.h>
// #include <rviz/properties/property.h>
#include <rviz/message_filter_display.h>
#include <chunkmap_msgs/UpdateList.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreMesh.h>
#include <chunkmap/chunk_map.h>
#include "chunk_visual.h"

namespace IMRL
{
    struct ChunkItem
    {
        std::vector<ChunkVisual::Ptr> layers;
    };

    class ChunkMapDisplay : public rviz::MessageFilterDisplay<chunkmap_msgs::UpdateList>
    {
        Q_OBJECT

    public:
        ChunkMapDisplay();
        virtual ~ChunkMapDisplay();

        ChunkMapDisplay(const ChunkMapDisplay &) = delete;
        ChunkMapDisplay &operator=(const ChunkMapDisplay &) = delete;

    private:
        void onInitialize() override;
        void reset() override;
        void processMessage(const chunkmap_msgs::UpdateList::ConstPtr &msg) override;
        void update(float wall_dt, float ros_dt) override;
        void updateTopic() override;

        void addUpdate(const std::vector<ChunkMap::Index> &index_list);
        void loadChunks(const std::vector<ChunkMap::Index> &index_list);

        std::mutex mutex_;
        std::map<ChunkMap::Index, ChunkItem> chunks_;
        Ogre::SceneNode *scene_node_ = nullptr;
        Ogre::MeshPtr mesh_;
        std::string base_topic_;
        float resolution_;
        uint32_t chunk_size_;
        ros::NodeHandle private_nh_;
        ros::ServiceClient load_chunk_client_;

        std::mutex list_mutex_;
        std::set<ChunkMap::Index> need_update_;
    };

}

#endif // _CHUNK_MAP_CHUNKMAP_DISPLAY_H_
