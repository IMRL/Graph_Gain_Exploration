// sys
#include <queue>
#include <deque>
#include <mutex>
#include <fstream>
#include <cmath>

// ros
#include <ros/ros.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/MapMetaData.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <chunkmap_ros/chunk_map_client.h>
#include <chunkmap_msgs/GetChunkMapInfo.h>
#include <chunkmap_msgs/GetChunkData.h>
#include <chunkmap_msgs/GetKeyFrameInfo.h>
#include <chunkmap_msgs/GetKeyFrameData.h>

// eigen
// #include <Eigen/Eigen>

class OgmNode
{
    ChunkMap::Ptr chunk_map;
    int cnt;
    std::vector<int> ops;
    std::vector<int> ops2;
    double free_bound;
    double occu_bound;
    ros::Subscriber subLidar;
    ros::Publisher ogmPub;
    ros::Publisher metaPub;

    Eigen::Matrix4f last_pose;

public:
    OgmNode(ros::NodeHandle &nh)
    {
        cnt = 0;
        std::string odom_topic;
        std::string rbtlc_topic = "/odometry/filtered";
        nh.param<std::string>("odom_topic", odom_topic, rbtlc_topic);
        nh.param<double>("free_bound", free_bound, 40);
        nh.param<double>("occu_bound", occu_bound, 55);
        nh.getParam("a_list", ops);
        nh.getParam("b_list", ops2);
        ROS_INFO_STREAM("free:\t" << free_bound << "\toccu:\t" << occu_bound);

        //  https://github.com/ros-planning/navigation/blob/noetic-devel/map_server/src/main.cpp
        ogmPub = nh.advertise<nav_msgs::OccupancyGrid>("/map", 1, true);
        metaPub = nh.advertise<nav_msgs::MapMetaData>("/map_metadata", 1, true);

        last_pose = Eigen::Matrix4f::Identity();

        chunk_map = std::make_shared<ChunkMapClient>(nh, "/chunkmap", [&](const std::set<ChunkMap::Index> &arg)
                                                     {   // things to be done when updated
                                                        this->sendFullmap();
                                                     });
        chunk_map->desc_type = ChunkMap::DescType::compressed;
        std::static_pointer_cast<ChunkMapClient>(chunk_map)->init();

        // subLidar = nh.subscribe<sensor_msgs::PointCloud2>(points_topic, /*2*/ 1, &OgmNode::pointHandler, this);
        // subOdom = nh.subscribe<nav_msgs::Odometry>(odom_topic, /*2*/ 1, &OgmNode::odomHandler, this);
    }

    // void pointHandler(const sensor_msgs::PointCloud2ConstPtr &msg) { }

    std::vector<signed char> MatrixToVector(cv::Mat1f mat)
    {
        // cv::Mat flat = mat.reshape(1, mat.total()*mat.channels());
        // std::vector<signed char> vec = mat.isContinuous()? flat : flat.clone();
        // return vec;
        unsigned int rows = mat.rows;
        unsigned int cols = mat.cols;
        size_t size = rows*cols;
        std::vector<signed char> Vector(size);
        for(unsigned int i = 0;i < rows; i++)
        {
            for(unsigned int j = 0; j < cols; j++)
            {
                if (mat[i][j] == -1) {
                    Vector[j + i*cols ] = mat[i][j];
                } else if (mat[i][j] <= free_bound) {
                    Vector[j + i*cols ] = 0;  // free
                } else if (mat[i][j] < occu_bound) {
                    Vector[j + i*cols ] = -1;  // unknown
                } else {
                    Vector[j + i*cols ] = 100;  // obstacle
                }
                // Vector[j + i*cols ] = mat[i][j];
                // ROS_WARN_STREAM(((int)Vector[j + i*cols ]) << '\t' << mat[i][j]);
            }
        }
        return Vector;
    }

    void sendFullmap() {
        std::vector<int64_t> indice_x, indice_y;
        for (const auto key : chunk_map->keys())
        {
            indice_x.push_back(key.x);
            indice_y.push_back(key.y);
        }
        if (indice_x.size() == 0) return;
        std::sort(indice_x.begin(), indice_x.end());
        std::sort(indice_y.begin(), indice_y.end());
        double minx = indice_x[0], miny = indice_y[0], maxx = indice_x[indice_x.size()-1], maxy = indice_y[indice_y.size()-1];

        int w = maxx - minx+1, h = maxy - miny+1, step = chunk_map->chunkSize();
        cv::Mat1f full_map(h*step, w*step);
        cv::Mat1f tile(step, step);
        full_map.setTo(-1);
        // std::cout << h << ' ' << w << ' '  << step << std::endl;
        for (const auto key : chunk_map->keys())
        {
            const auto& chunk = chunk_map->at(key);
            for (const auto &layer : chunk.getLayers())
            {   
                // std::cout << key.x << ' ' << key.y << ' ' << layer.occupancy.size() << std::endl;
                // ROS_INFO_STREAM(layer.occupancy);
                tile = (layer.occupancy / (-2.55) + 100);
                // ROS_INFO_STREAM(tile);
                tile.copyTo(full_map(cv::Rect((key.x-((int)minx)) * step, (key.y - ((int)miny)) * step, step, step)));
            }
        }

        // https://github.com/SunZezhou/Active-SLAM-with-Cartographer/blob/master/map_after_process/src/map_after_process_node.cpp
        cv::Mat wall_mask = (full_map >= occu_bound) & (full_map != -1);
        cv::Mat free_mask = (full_map <= free_bound) & (full_map != -1);

        // cv::dilate(wall_mask, wall_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * DILATE_RADIUS + 1, 2 * DILATE_RADIUS + 1), cv::Point(DILATE_RADIUS, DILATE_RADIUS)));
    

        for (auto i : ops) {
            std::cout << i;
            if (i < 0) {
                auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(-i, -i));
                // auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(-i, -i), cv::Point(-i/2, -i/2));
                cv::erode(free_mask, free_mask, kernel);
                // cv::morphologyEx(free_mask, free_mask, cv::MORPH_CLOSE, kernel);
            } else if (i > 0) {
                auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(i, i));
                // auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(i/2, i/2));
                cv::dilate(free_mask, free_mask, kernel);
                // cv::morphologyEx(free_mask, free_mask, cv::MORPH_OPEN, kernel);
            }
        }

        for (auto i : ops2) {
            std::cout << i;
            if (i < 0) {
                auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(-i, -i));
                // auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(-i, -i), cv::Point(-i/2, -i/2));
                cv::erode(wall_mask, wall_mask, kernel);
                // cv::morphologyEx(wall_mask, wall_mask, cv::MORPH_CLOSE, kernel);
            } else if (i > 0) {
                auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(i, i));
                // auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(i, i), cv::Point(i/2, i/2));
                cv::dilate(wall_mask, wall_mask, kernel);
                // cv::morphologyEx(wall_mask, wall_mask, cv::MORPH_OPEN, kernel);
            }
        }

        full_map.setTo(-1);
        full_map.setTo(0, free_mask);
        full_map.setTo(100, wall_mask);

        nav_msgs::OccupancyGrid ogm;
        ogm.info.width = w*step;
        ogm.info.height = h*step;
        ogm.info.origin.position.x = minx*chunk_map->chunkBase();
        ogm.info.origin.position.y = miny*chunk_map->chunkBase();
        ogm.info.origin.position.z = 0.0;
        ogm.info.origin.orientation.w = 1.0;
        ogm.info.origin.orientation.x = 0.0;
        ogm.info.origin.orientation.y = 0.0;
        ogm.info.origin.orientation.z = 0.0;
        ogm.info.resolution = chunk_map->resolution();
        // ogm.info.map_load_time
        ogm.header.stamp = ros::Time::now();
        ogm.header.frame_id = "map";
        ogm.header.seq = cnt;
        ogm.data = MatrixToVector(full_map);

        nav_msgs::MapMetaData meta;
        // meta.header.stamp = ogm.header.stamp;
        // meta.header.frame_id = "map";
        meta.origin = ogm.info.origin;
        meta.resolution = ogm.info.resolution;
        meta.width  = ogm.info.width ;
        meta.height = ogm.info.height;
        metaPub.publish(meta);
        ogmPub.publish(ogm);
        ROS_INFO_STREAM("\033[1;32m----> pub ogm the " << cnt << "th time.\n" << meta <<".\033[0m");
        cnt += 1;
        /*
map_load_time: 
  secs: 1666762396
  nsecs: 170736392
resolution: 0.20000000298
width: 800
height: 800
origin: 
  position: 
    x: -40.0
    y: -40.0
    z: 0.0
  orientation: 
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
        */
    }
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "ogm_sender");
    ros::NodeHandle nh("~");

    OgmNode node{nh};

    ROS_INFO("\r\n\033[1;32m----> subs cm and pubs ogm.\033[0m");
    // while ()
    ros::spin();

    return 0;
}
