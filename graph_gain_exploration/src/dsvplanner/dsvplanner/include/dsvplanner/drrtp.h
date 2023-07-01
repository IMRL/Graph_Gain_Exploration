/**************************************************************************
drrtp.h
Header of the drrtp (dynamic_rrtplanner) class

Hongbiao Zhu(hongbiaz@andrew.cmu.edu)
05/25/2020

**************************************************************************/

#ifndef DRRTP_H_
#define DRRTP_H_

#include "dsvplanner/clean_frontier_srv.h"
#include "dsvplanner/drrt.h"
#include "dsvplanner/drrt_base.h"
#include "dsvplanner/dsvplanner_srv.h"
#include "dsvplanner/dual_state_frontier.h"
#include "dsvplanner/dual_state_graph.h"
#include "dsvplanner/grid.h"
// #include "octomap_world/octomap_manager.h"
#include "chunkmap_ros/chunk_map_client.h"
#include "laser_geometry/laser_geometry.h"

namespace dsvplanner_ns
{
class drrtPlanner
{
public:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // ros::Subscriber odomSub_;
  ros::Subscriber boundarySub_;
  ros::Subscriber pathSub_;

  ros::ServiceServer plannerService_;
  ros::ServiceServer cleanFrontierService_;
  dsvplanner::dsvplanner_srv::Response preRes;
  int preId;
  int waitLoop = 5;;

  Params params_;
  // volumetric_mapping::OctomapManager* manager_;
  ChunkMap::Ptr chunk_map_;
  DualStateGraph* dual_state_graph_;
  DualStateFrontier* dual_state_frontier_;
  Drrt* drrt_;
  OccupancyGrid* grid_;

  message_filters::Subscriber<nav_msgs::Odometry> odomSub_;
  message_filters::Subscriber<sensor_msgs::LaserScan> scanSub_;
  typedef message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, sensor_msgs::LaserScan> syncPolicy;
  typedef message_filters::Synchronizer<syncPolicy> Sync;
  boost::shared_ptr<Sync> sync_;
 

  bool init();
  bool setParams();
  // bool setPublisherPointer();
  void posecovCallback(const sensor_msgs::PointCloud2& msg);
  void odomCallback(const nav_msgs::Odometry& pose);
  void scanCallback(const sensor_msgs::LaserScan::Ptr& scan);
  void scanAndOdomSyncHandler(const nav_msgs::Odometry::ConstPtr& pose, const sensor_msgs::LaserScan::ConstPtr& scan);
  laser_geometry::LaserProjection projector_;
  void boundaryCallback(const geometry_msgs::PolygonStamped& boundary);
  bool plannerServiceCallback(dsvplanner::dsvplanner_srv::Request& req, dsvplanner::dsvplanner_srv::Response& res);
  bool cleanFrontierServiceCallback(dsvplanner::clean_frontier_srv::Request& req,
                                    dsvplanner::clean_frontier_srv::Response& res);
  void cleanLastSelectedGlobalFrontier();

  drrtPlanner(ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  ~drrtPlanner();

private:
  std::string odomSubTopic;
  std::string scanSubTopic;
  std::string boundarySubTopic;
  std::string newTreePathPubTopic;
  std::string remainingTreePathPubTopic;
  std::string boundaryPubTopic;
  std::string globalSelectedFrontierPubTopic;
  std::string localSelectedFrontierPubTopic;
  std::string plantimePubTopic;
  std::string nextGoalPubTopic;
  std::string randomSampledPointsPubTopic;
  std::string shutDownTopic;
  std::string plannerServiceName;
  std::string cleanFrontierServiceName;

  std::chrono::steady_clock::time_point plan_start_;
  std::chrono::steady_clock::time_point RRT_generate_over_;
  std::chrono::steady_clock::time_point gain_computation_over_;
  std::chrono::steady_clock::time_point plan_computation_over_;;
  std::chrono::steady_clock::time_point plan_over_;
  std::chrono::steady_clock::duration time_span;
};
}

#endif  // DRRTP_H
