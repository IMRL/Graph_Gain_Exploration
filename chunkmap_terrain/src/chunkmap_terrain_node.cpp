#include <ros/ros.h>
#include <chunkmap_ros/chunk_map_client.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

static constexpr float VHeight = 0.6;

class Node
{
    ros::Subscriber subOdom;
    ros::Publisher pubTerrain;
    ChunkMap::Ptr chunk_map;
    ros::Timer timer;

    bool inited = false;
    Eigen::Vector3f pose;
    float range;

public:
    Node(ros::NodeHandle &nh)
    {
        nh.param<float>("range", range, 15);
        subOdom = nh.subscribe<nav_msgs::Odometry>("/state_estimation", 5, &Node::odomHandler, this);
        pubTerrain = nh.advertise<sensor_msgs::PointCloud2>("/chunkmap_terrain_map", 1);
        chunk_map = std::make_shared<ChunkMapClient>(nh, "/chunkmap", [&](const std::set<ChunkMap::Index> &arg) {});
        timer = nh.createTimer(ros::Duration(0.2), &Node::handler, this);
        timer.start();
    }

    void odomHandler(const nav_msgs::OdometryConstPtr &msg)
    {
        pose = Eigen::Vector3f{msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z};
        inited = true;
    }

    void handler(const ros::TimerEvent &)
    {
        if (!inited)
            return;
        ChunkMap::CellIndex base_index;
        if (!chunk_map->query(pose, base_index))
            return;
        int offset = std::ceil(range / chunk_map->chunkBase());
        const auto cbase = chunk_map->chunkBase();
        const auto step = chunk_map->chunkSize();
        const auto resolution = chunk_map->resolution();
        pcl::PointCloud<pcl::PointXYZI>::Ptr terrain(new pcl::PointCloud<pcl::PointXYZI>);
        for (int x = -offset; x <= offset; x++)
            for (int y = -offset; y <= offset; y++)
            {
                ChunkMap::Index index = base_index.chunk;
                index.x += x;
                index.y += y;
                if (chunk_map->has(index))
                {
                    const auto &layers = chunk_map->at(index).getLayers();
                    for (const auto &layer : layers)
                    {
                        for (int r = 0; r < step; r++)
                            for (int c = 0; c < step; c++)
                                if (layer.observe(r, c) == 255 /*&& std::abs(layer.elevation(r, c) - pose.z()) < 2.0*/)
                                {
                                    pcl::PointXYZI p;
                                    p.x = index.x * cbase + c * resolution;
                                    p.y = index.y * cbase + r * resolution;
                                    p.z = layer.elevation(r, c) - VHeight;
                                    if (layer.occupancy(r, c) <= 127)
                                    {
                                        for (int i = 0; i < 15; i++)
                                        {
                                            p.z += 0.1;
                                            p.intensity = p.z - pose.z() + VHeight;
                                            terrain->push_back(p);
                                        }
                                    }
                                    else if (layer.occupancy(r, c) < 130)
                                        continue;
                                    else
                                    {
                                        p.intensity = 0;//p.z - pose.z() + VHeight;
                                        terrain->push_back(p);
                                    }
                                    // p.intensity = p.z;
                                    // p.z -= VHeight;
                                    // terrain->push_back(p);
                                }
                    }
                }
            }
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*terrain, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();
        pubTerrain.publish(msg);
    }
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "chunkmap_terrain");
    ros::NodeHandle nh("~");

    Node node{nh};

    ros::spin();

    return 0;
}
