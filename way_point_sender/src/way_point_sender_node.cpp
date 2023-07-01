// sys
#include <queue>
#include <deque>
#include <mutex>
#include <fstream>

// ros
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <cmath>

#include <yaml-cpp/yaml.h>

// eigen
// #include <Eigen/Eigen>

class WaypointSender {
    public:
        ros::NodeHandle nh;

        ros::Publisher  waypointPub;
        ros::Subscriber lidarOdomSub;

        double scale = 0.1;
        double ox = -0, oy = -0;
        int curIdx = 0;

        std::vector<geometry_msgs::PointStamped> way_points;

        WaypointSender() : nh("~") {
            lidarOdomSub = nh.subscribe("/odometry/filtered", 1, &WaypointSender::odomHandler, this, ros::TransportHints().tcpNoDelay());

            waypointPub = nh.advertise<geometry_msgs::PointStamped>("/way_point", 10);


            YAML::Node node = YAML::LoadFile("/ros_ws/chunkmap.yml");
            //todo
            
            double xx, yy;

            std::ifstream configFile("/ros_ws/way_point_config.txt");
            configFile >> ox >> oy >> scale;
            configFile.close();

            std::ifstream waypointsFile("/ros_ws/way_point.txt");
            // Eigen::Vector3d waypoints;
            while(!waypointsFile.eof()) {
                waypointsFile >> yy >> xx;
                geometry_msgs::PointStamped point;
                point.point.x = (xx + ox)*scale;
                point.point.y = (yy + oy)*scale;
                ROS_INFO_STREAM(point.point);
                // z
                way_points.push_back(point);
            }
            waypointsFile.close();
            ROS_INFO_STREAM("read size of " << way_points.size());
        }
        //geometry_msgs::PointStamped
        void odomHandler(const nav_msgs::OdometryConstPtr &msg) {
            // if (std::isnan(msg->latitude + msg->longitude + msg->altitude)) {
            //     ROS_ERROR("POS LLA NAN...");
            //     return;
            // }
            // ROS_INFO("odom!");
        
            // if (timestamp % 1000)  // down frequency
            double px = msg->pose.pose.position.x;
            double py = msg->pose.pose.position.y;
            double pz = msg->pose.pose.position.z;

            int window_size = 30;
            int searchEnd = curIdx + window_size, tmp, nxtIdx;
            if (curIdx == way_points.size()-1) {
                // nxtIdx = curIdx;  // last way point; dont update
                return;
            } else {
                if (way_points.size() < searchEnd) {
                    searchEnd = way_points.size();
                }
                // find next waypoint 6-meter or 30 points away, but should not get closer
                double dis = 0, last_dis = -1;
                for (tmp = curIdx; tmp < searchEnd; tmp++) {
                    double ax = way_points[tmp].point.x, ay = way_points[tmp].point.y;
                    dis = std::sqrt((ax-px)*(ax-px)+(ay-py)*(ay-py));
                    if (dis > 6 && dis > last_dis) {
                        break;
                    } else {
                        last_dis = dis;
                    }
                }
                nxtIdx = tmp;
            }

            // TODO: add orientation
            geometry_msgs::PointStamped waypoint = way_points[nxtIdx];
            waypoint.header.stamp = msg->header.stamp;
            waypoint.header.frame_id = "map";
            waypoint.point.z = pz;  // 
            waypointPub.publish(waypoint);
            ROS_INFO_STREAM("send: " << nxtIdx << " from " << curIdx);
            curIdx = nxtIdx;
        }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "waypoint_sender");
  WaypointSender sender;
  ROS_INFO("\033[1;32m----> Simple waypoint_sender Started.\033[0m");
  ros::spin();
  return 0;
}