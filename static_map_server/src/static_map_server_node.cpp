#include <ros/ros.h>
#include <chunkmap_ros/chunk_map_service.h>

class Node
{
    ChunkMap::Ptr chunk_map;

    std::string save_path;

public:
    Node(ros::NodeHandle &nh)
    {
        float resolution;
        float chunk_base;
        nh.param<float>("resolution", resolution, 0.1f);
        nh.param<float>("chunk_base", chunk_base, 3.0f);
        nh.param<std::string>("save_path", save_path, "");
        if(save_path != "") ROS_INFO_STREAM("will be saved to " << save_path);
        // init chunk map
        chunk_map = std::make_shared<ChunkMapService>(ChunkMap::Config{
            .resolution = resolution,
            .chunk_base = chunk_base,
        }, nh);
        chunk_map->desc_type = ChunkMap::DescType::compressed;

        std::ifstream ifs(save_path, std::ios::binary);
        ChunkMap::load(ifs, chunk_map);
        ifs.close();
    }
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "static_map_server");
    ros::NodeHandle nh("~");

    Node node{nh};

    ros::spin();

    return 0;
}
