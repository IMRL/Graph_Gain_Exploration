#include "chunkmap_display.h"
#include <chunkmap_msgs/GetChunkMapInfo.h>
#include <chunkmap_msgs/GetChunkVizData.h>
#include <OGRE/OgreManualObject.h>

// #define ROS_PACKAGE_NAME "chunkmap_msgs"

namespace IMRL
{
    constexpr char kMaterialsDirectory[] = "/ogre_media/materials";
    constexpr char kGlsl120Directory[] = "/glsl120";
    constexpr char kScriptsDirectory[] = "/scripts";

    ChunkMapDisplay::ChunkMapDisplay() : chunk_size_(0)
    {
        // local_map_list_property_ = new rviz::Property(QString("local maps"), QVariant(), QString(), this);
        const std::string package_path = ros::package::getPath(ROS_PACKAGE_NAME);
        Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
            package_path + kMaterialsDirectory, "FileSystem", ROS_PACKAGE_NAME);
        Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
            package_path + kMaterialsDirectory + kGlsl120Directory, "FileSystem",
            ROS_PACKAGE_NAME);
        Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
            package_path + kMaterialsDirectory + kScriptsDirectory, "FileSystem",
            ROS_PACKAGE_NAME);
        Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();
    }

    ChunkMapDisplay::~ChunkMapDisplay()
    {
        chunks_.clear();
        scene_manager_->destroySceneNode(scene_node_);
    }

    void ChunkMapDisplay::onInitialize()
    {
        MFDClass::onInitialize();
        scene_node_ = scene_manager_->getRootSceneNode()->createChildSceneNode();
    }

    void ChunkMapDisplay::reset()
    {
        MFDClass::reset();
        std::lock_guard<std::mutex> lock(mutex_);
        chunks_.clear();
    }

    void ChunkMapDisplay::processMessage(const chunkmap_msgs::UpdateList::ConstPtr &msg)
    {
        std::vector<ChunkMap::Index> index_list;
        for (const auto &it : msg->chunks)
        {
            index_list.push_back({it.x, it.y});
        }
        addUpdate(index_list);
    }

    void ChunkMapDisplay::update(const float wall_dt, const float ros_dt)
    {
        static int MAX_UPDATE_ONCE = 100;
        //
        std::vector<ChunkMap::Index> index_list;
        {
            std::lock_guard<std::mutex> guard(list_mutex_);
            for (const auto &i : need_update_)
            {
                index_list.push_back(i);
                if (index_list.size() >= MAX_UPDATE_ONCE)
                    break;
            }
            for (const auto &i : index_list)
            {
                need_update_.erase(i);
            }
        }
        if (index_list.size() > 0)
            loadChunks(index_list);
    }

    void ChunkMapDisplay::updateTopic()
    {
        MFDClass::updateTopic();
        std::string update_topic = topic_property_->getTopicStd();
        base_topic_ = update_topic.substr(0, update_topic.length() - 12);
        load_chunk_client_ = private_nh_.serviceClient<chunkmap_msgs::GetChunkVizData>(base_topic_ + "/get_chunk_viz_data");
        chunkmap_msgs::GetChunkMapInfo info_msg;
        if (ros::service::call(base_topic_ + "/get_info", info_msg.request, info_msg.response))
        {
            resolution_ = info_msg.response.ChunkMapInfo.resolution;
            chunk_size_ = info_msg.response.ChunkMapInfo.chunk_size;
            {
                float base_offset = resolution_ / 2;
                Ogre::ManualObject *manual_object_ = scene_manager_->createManualObject("ChunkManual");
                manual_object_->begin("chunk_map/ChunkCell", Ogre::RenderOperation::OT_TRIANGLE_LIST);
                for (int y = 0; y < chunk_size_; y++)
                {
                    for (int x = 0; x < chunk_size_; x++)
                    {
                        float tex_x = (x + 0.5) / (float)chunk_size_;
                        float tex_y = (y + 0.5) / (float)chunk_size_;

                        manual_object_->position((x * resolution_) - base_offset, (y * resolution_) - base_offset, 0.0f);
                        manual_object_->textureCoord(tex_x, tex_y);
                        manual_object_->normal(0.0f, 0.0f, 1.0f);

                        manual_object_->position((x * resolution_) + base_offset, (y * resolution_) - base_offset, 0.0f);
                        manual_object_->textureCoord(tex_x, tex_y);
                        manual_object_->normal(0.0f, 0.0f, 1.0f);

                        manual_object_->position((x * resolution_) - base_offset, (y * resolution_) + base_offset, 0.0f);
                        manual_object_->textureCoord(tex_x, tex_y);
                        manual_object_->normal(0.0f, 0.0f, 1.0f);

                        manual_object_->position((x * resolution_) + base_offset, (y * resolution_) + base_offset, 0.0f);
                        manual_object_->textureCoord(tex_x, tex_y);
                        manual_object_->normal(0.0f, 0.0f, 1.0f);

                        int id = (y * chunk_size_ + x) * 4;
                        manual_object_->quad(id, id + 1, id + 3, id + 2);
                    }
                }
                manual_object_->end();
                mesh_ = manual_object_->convertToMesh("ChunkMesh");
                scene_manager_->destroyManualObject(manual_object_);
            }
            // ROS_INFO_STREAM(resolution_ << " " << chunk_size_);
            std::vector<ChunkMap::Index> index_list;
            for (const auto &it : info_msg.response.ChunkMapInfo.chunk_index)
            {
                index_list.push_back({it.x, it.y});
            }
            addUpdate(index_list);
        }
    }

    void ChunkMapDisplay::addUpdate(const std::vector<ChunkMap::Index> &index_list)
    {
        std::lock_guard<std::mutex> guard(list_mutex_);
        for (const auto &i : index_list)
        {
            need_update_.insert(i);
        }
    }

    void ChunkMapDisplay::loadChunks(const std::vector<ChunkMap::Index> &index_list)
    {
        // ROS_WARN("load chunk");
        if (chunk_size_ == 0)
            return;
        // {
        //     std::lock_guard<std::mutex> lock(mutex_);
        //     for (const auto &index : index_list)
        //     {
        //         if (chunks_.find(index) == chunks_.end())
        //         {
        //             ChunkItem item;
        //             // item.chunk.reset(new ChunkVisual(scene_manager_, scene_node_, resolution_, chunk_size_, index));
        //             chunks_[index] = item;
        //         }
        //     }
        // }
        chunkmap_msgs::GetChunkVizData msg;
        for (const auto &index : index_list)
        {
            chunkmap_msgs::ChunkIndex req_index;
            req_index.x = index.x;
            req_index.y = index.y;
            msg.request.index.push_back(req_index);
        }
        if (!mutex_.try_lock())
        {
            ROS_WARN("delay chunk_map refresh.");
            return;
        }
        // ROS_WARN("lock success");
        // if (ros::service::call(base_topic_ + "/get_chunk_viz_data", msg.request, msg.response))
        if (load_chunk_client_.call(msg))
        {
            // ROS_WARN("query success");
            // TODO: where is the faster lock place?
            // std::lock_guard<std::mutex> lock(mutex_);
            for (int i = 0; i < index_list.size(); i++)
            {
                if (msg.response.has_chunk[i])
                {
                    // TODO: another lock place
                    const auto &index = index_list[i];
                    const auto &data = msg.response.data[i];
                    // if (chunks_.find(index) == chunks_.end())
                    //     chunks_.insert(index, {});
                    auto &item = chunks_[index];
                    int more = data.layers.size() - item.layers.size();
                    // TODO: could faster?
                    for (int j = 0; j < more; j++)
                    {
                        // ROS_WARN("more:(");
                        item.layers.push_back(std::make_shared<ChunkVisual>(scene_manager_, scene_node_, mesh_, resolution_, chunk_size_, index, item.layers.size()));
                    }
                    int less = item.layers.size() - data.layers.size();
                    for (int j = 0; j < less; j++)
                    {
                        // ROS_WARN("less:(");
                        item.layers.pop_back();
                    }
                    // ROS_WARN("before update");
                    for (int j = 0; j < data.layers.size(); j++)
                    {
                        item.layers[j]->updateTexture(data.layers[j].elevation, data.layers[j].occupancy, data.layers[j].elevation_alpha, data.layers[j].elevation_beta);
                    }
                    // ROS_WARN("update finish");
                    // chunks_[index_list[i]].chunk->updateTexture(data.elevation, data.occupancy, data.elevation_alpha, data.elevation_beta);
                }
                else
                {
                    ROS_WARN_STREAM("no chunk: (" << index_list[i].x << "," << index_list[i].y << ")");
                }
            }
        }
        mutex_.unlock();
    }
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(IMRL::ChunkMapDisplay, rviz::Display)
