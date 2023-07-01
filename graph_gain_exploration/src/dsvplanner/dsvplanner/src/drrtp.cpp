/*
drrtp.cpp
Implementation of drrt_planner class

Created by Hongbiao Zhu (hongbiaz@andrew.cmu.edu)
05/25/2020
*/

#include <eigen3/Eigen/Dense>

#include <visualization_msgs/Marker.h>
#include <misc_utils/misc_utils.h>
#include <dsvplanner/drrtp.h>

#include <graph_utils.h>

using namespace Eigen;

dsvplanner_ns::drrtPlanner::drrtPlanner(ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
  : nh_(nh), nh_private_(nh_private)
{
  chunk_map_ = std::make_shared<ChunkMapClient>(nh, "/chunkmap", [](const std::set<ChunkMap::Index> &arg){});
  chunk_map_->desc_type = ChunkMap::DescType::compressed;
  std::static_pointer_cast<ChunkMapClient>(chunk_map_)->init();
  // manager_ = new volumetric_mapping::OctomapManager(nh_, nh_private_);
  grid_ = new OccupancyGrid(nh_, nh_private_);
  dual_state_graph_ = new DualStateGraph(nh_, nh_private_, chunk_map_, grid_);
  dual_state_frontier_ = new DualStateFrontier(nh_, nh_private_, chunk_map_, grid_);
  drrt_ = new Drrt(chunk_map_, dual_state_graph_, dual_state_frontier_, grid_);

  init();
  drrt_->setParams(params_);
  drrt_->init();

  ROS_INFO("Successfully launched DSVP node");
}

dsvplanner_ns::drrtPlanner::~drrtPlanner()
{
  // if (manager_)
  // {
  //   delete manager_;
  // }
  if (dual_state_graph_)
  {
    delete dual_state_graph_;
  }
  if (dual_state_frontier_)
  {
    delete dual_state_frontier_;
  }
  if (grid_)
  {
    delete grid_;
  }
  if (drrt_)
  {
    delete drrt_;
  }
}

// void dsvplanner_ns::drrtPlanner::odomCallback(const nav_msgs::Odometry& pose)
// {
//   drrt_->setRootWithOdom(pose);
//   // Planner is now ready to plan.
//   drrt_->plannerReady_ = true;
// }

void dsvplanner_ns::drrtPlanner::scanCallback(const sensor_msgs::LaserScan::Ptr& scan)
{
  //Polar coordinate method
  if(scan->ranges.size() <= 0)
    return;
  drrt_->scanCloud_->clear();
  sensor_msgs::PointCloud2::Ptr cloud;
  projector_.projectLaser(*scan, *cloud);
  pcl::fromROSMsg(*cloud, *drrt_->scanCloud_);

  //scale
  // for(int i = 0; i < drrt_->scanPolar.size(); i++) {
  //   drrt_->scanPolar[i] *= 2.5;
  // }

//for rviz
  // std::ofstream rangeFile;
  // rangeFile.open("/home/szz/Documents/dsv_planner_sim/results/rangeFile.txt");
  // drrt_->scanPolarPoint_.clear();
  // for(int i = 0; i < drrt_->scanPolar.size(); i++) {
  //   tf::Vector3 initPoint(drrt_->scanPolar[i] * cos(i * M_PI / 180),
  //                         drrt_->scanPolar[i] * sin(i * M_PI / 180), drrt_->root_[2]);
  //   tf::Vector3 rotPoint = drrt_->transformToMap * initPoint;
  //   rangeFile << rotPoint[0] << " " << rotPoint[1] << " " << rotPoint[2] << std::endl;
  //   pcl::PointXYZ tempPoint;
  //   tempPoint.x = rotPoint.getX();
  //   tempPoint.y = rotPoint.getY();
  //   tempPoint.z = rotPoint.getZ();
  //   tempPoint.intensity = 0;
  //   drrt_->scanPolarPoint_->points.push_back(tempPoint);
  // }
  // rangeFile.close();
}

void dsvplanner_ns::drrtPlanner::scanAndOdomSyncHandler(const nav_msgs::Odometry::ConstPtr& pose, const sensor_msgs::LaserScan::ConstPtr& scan) {
  drrt_->setRootWithOdom(*pose);
  // Planner is now ready to plan.
  drrt_->plannerReady_ = true;

  if(scan->ranges.size() <= 0)
    return;
  drrt_->scanCloud_->clear();
  sensor_msgs::PointCloud2 cloud;
  projector_.projectLaser(*scan, cloud);
  pcl::fromROSMsg(cloud, *drrt_->scanCloud_);
}

void dsvplanner_ns::drrtPlanner::boundaryCallback(const geometry_msgs::PolygonStamped& boundary)
{
  drrt_->setBoundary(boundary);
  dual_state_frontier_->setBoundary(boundary);
}

void dsvplanner_ns::drrtPlanner::posecovCallback(const sensor_msgs::PointCloud2& msg) {
  drrt_->poseCov_->clear();
  pcl::fromROSMsg(msg, *drrt_->poseCov_);
  ROS_WARN_STREAM(drrt_->poseCov_->points[drrt_->poseCov_->points.size()-1].intensity);
  // if((drrt_->loopflag == true && drrt_->poseCov_->points[drrt_->poseCov_->points.size()-1].intensity < 3e-5) || drrt_->poseCov_->points[drrt_->poseCov_->points.size()-1].intensity > 9e-4) {
  if(0) {
    drrt_->loopflag = false; 
    drrt_->loopInit = false;
    drrt_->loopContinue = false;
  }
  if(drrt_->loopflag == false) {
    if(0) {
    // if(drrt_->poseCov_->points[drrt_->poseCov_->points.size()-1].intensity > 1e-4) {
        drrt_->local_plan_ = true;
        drrt_->loopflag = true;
        drrt_->loopInit = false;
        drrt_->loopContinue = false;
        ROS_WARN_STREAM("Active LOOP!");

        geometry_msgs::Point point;
        point.x = drrt_->root_[0];
        point.y = drrt_->root_[1];
        point.z = drrt_->root_[2];
        drrt_->loopPoint.push_back(point);
        point.x = drrt_->poseCov_->points[drrt_->poseCov_->points.size()/2].y;
        point.y = drrt_->poseCov_->points[drrt_->poseCov_->points.size()/2].x;
        point.z = drrt_->poseCov_->points[drrt_->poseCov_->points.size()/2].z;
        drrt_->loopPoint.push_back(point);
        
      }
  }
}

bool dsvplanner_ns::drrtPlanner::plannerServiceCallback(dsvplanner::dsvplanner_srv::Request& req,
                                                        dsvplanner::dsvplanner_srv::Response& res)
{
  plan_start_ = std::chrono::steady_clock::now();
  // drrt_->gotoxy(0, 10);  // Go to the specific line on the screen
  // Check if the planner is ready.
  if (!drrt_->plannerReady_)
  {
    std::cout << "No odometry. Planner is not ready!" << std::endl;
    return true;
  }
  // if (manager_ == NULL)
  if (chunk_map_ == nullptr)
  {
    std::cout << "No chunkmap. Planner is not ready!" << std::endl;
    return true;
  }
  // if (manager_->getMapSize().norm() <= 0.0)
  if (chunk_map_->keys().size() == 0)
  {
    std::cout << "Chunkmap is empty. Planner is not set up!" << std::endl;
    return true;
  }

  // set terrain points and terrain voxel elevation
  drrt_->setTerrainVoxelElev();

  using namespace std::chrono_literals;

  // Clear old tree and the last global frontier.
  cleanLastSelectedGlobalFrontier();
  drrt_->clear();
  // Reinitialize.
  // drrt_->plannerInit();
  drrt_->plannerInit_szz();

  // Iterate the tree construction method.
  int loopCount = 0;
  bool haveSampled = false;
  while (ros::ok() && drrt_->remainingFrontier_ && (drrt_->getNodeCounter() < params_.kCuttoffIterations || drrt_->loopflag == true) &&
         !(drrt_->normal_local_iteration_ && (drrt_->getNodeCounter() >= params_.kVertexSize && drrt_->gainFound())))
  {
    if (loopCount > drrt_->loopCount_ * (drrt_->getNodeCounter() + 1))
    {
      break;
    }
    drrt_->plannerIterate();
    loopCount++;
    haveSampled = true;
  }
  if(dual_state_graph_->local_graph_.vertices.size() == 0) {
    std::cout << "local graph empty!" << std::endl;
    return false;
  }
  if(haveSampled == true) {
    RRT_generate_over_ = std::chrono::steady_clock::now();
    time_span = RRT_generate_over_ - plan_start_;
    double rrtExpansionTime = 
        double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

    drrt_->getConcaveHull();
    RRT_generate_over_ = std::chrono::steady_clock::now();
    time_span = RRT_generate_over_ - plan_start_;
    double concaveTime = 
        double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den - rrtExpansionTime;

    drrt_->updateGraphGain();
    RRT_generate_over_ = std::chrono::steady_clock::now();
    time_span = RRT_generate_over_ - plan_start_;
    double graphgainTime = 
        double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den - concaveTime - rrtExpansionTime;
  }
  
  // Publish rrt
  drrt_->publishNode();
  std::cout << "     New node number is " << drrt_->getNodeCounter() << "\n"
            << "     Current local RRT size is " << dual_state_graph_->getLocalVertexSize() << "\n"
            << "     Current global graph size is " << dual_state_graph_->getGlobalVertexSize() << std::endl;
            // << "     RRT Expansion time is  " << rrtExpansionTime << std::endl 
            // << "     Concave Hull time is  " << concaveTime << std::endl
            // << "     Graph Gain time is  " << graphgainTime << std::endl;
  // infoFile << drrt_->getNodeCounter() << " ";
  RRT_generate_over_ = std::chrono::steady_clock::now();
  time_span = RRT_generate_over_ - plan_start_;
  double rrtGenerateTime =
      double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
  // if(drrt_->activeloop_ == true)
  //   res.mode.data = 4; // mode 4 means active loop
  // else if(drrt_->global_plan_ == true)
  //   res.mode.data = 3;  // mode 3 means global exploration
  // else if (drrt_->local_plan_ == true)
  //   res.mode.data = 1;  // mode 1 means local planning
  
  // Reset planner state
  drrt_->global_plan_pre_ = drrt_->global_plan_;
  drrt_->global_plan_ = false;
  drrt_->local_plan_ = false;

  // update planner state of last iteration
  dual_state_frontier_->setPlannerStatus(drrt_->global_plan_pre_);

  // Update planner state of next iteration
  geometry_msgs::Point robot_position;
  robot_position.x = drrt_->root_[0];
  robot_position.y = drrt_->root_[1];
  robot_position.z = drrt_->root_[2];
	
  float dist = 0;
  if (preRes.goal.size() != 0)
    dist = sqrt(pow((robot_position.x - preRes.goal[0].x), 2) + pow((robot_position.y - preRes.goal[0].y), 2)
               + pow((robot_position.z - preRes.goal[0].z), 2));
  
  double bestExplorationGain = dual_state_graph_->getGain_szz(robot_position, drrt_->twist_, dist);
  // double bestExplorationGain = dual_state_graph_->getGain(robot_position);
  
  if(drrt_->loopflag == true && drrt_->loopInit == false && drrt_->global_plan_pre_) {
        drrt_->loopInit = true;
        drrt_->loopPoint[0] = robot_position;
        geometry_msgs::Point bestLocalPoint = drrt_->loopPoint[drrt_->loopPoint.size()-1];
        drrt_->loopPoint.pop_back();
        // construct Asymmetric TSP Matrix
        drrt_->loopPoint = dual_state_graph_->getAsymmetricTSPMatrix(drrt_->loopPoint);      
        for(int ind = 2; ind < drrt_->loopPoint.size(); ind++) {
        if(misc_utils_ns::PointXYDist(drrt_->loopPoint[ind], bestLocalPoint) <= 2.5) {
            // std::swap(drrt_->loopPoint[ind], drrt_->loopPoint[drrt_->loopPoint.size()-1]);
            drrt_->loopPoint.erase(drrt_->loopPoint.begin()+ind);
            ind--;
          }
        }
        double seqOrder = 0, reverseOrder = 0;
        Eigen::Vector3d A,B(drrt_->loopPoint[0].x, drrt_->loopPoint[0].y, drrt_->loopPoint[0].z), C(drrt_->loopPoint[1].x, drrt_->loopPoint[1].y, drrt_->loopPoint[1].z);
        drrt_->tspTour = dual_state_graph_->solveTSP(drrt_->loopPoint);
        for(int ind = 0; ind < drrt_->tspTour.size(); ind++) {
          A[0] = drrt_->tspTour[ind].x; A[1] = drrt_->tspTour[ind].y; A[2] = drrt_->tspTour[ind].z; 
          double dis = ((A-B).cross(A-C)).norm() / (B-C).norm();
          seqOrder += dis * (1 - (double)ind / drrt_->tspTour.size());
          reverseOrder += dis * ind / drrt_->tspTour.size();
        }
        // drrt_->publishTSPPath();
        if(reverseOrder < seqOrder) reverse(drrt_->tspTour.begin(), drrt_->tspTour.end());
        drrt_->tspTour.insert(drrt_->tspTour.begin(), bestLocalPoint);
        drrt_->publishTSPPath();
  }
  else if(drrt_->loopflag == true && drrt_->loopInit == false && drrt_->global_plan_pre_ == false) {
    drrt_->local_plan_ = false;
    drrt_->global_plan_ = true;
    drrt_->loopPoint.push_back(dual_state_graph_->getBestLocalVertexPosition());
    dual_state_graph_->updateGlobalGraph();
    dual_state_graph_->updateExploreDirection();
    return true;
  }
  
  if(drrt_->loopflag != true) {
    if (!drrt_->nextNodeFound_ && drrt_->global_plan_pre_ && drrt_->gainFound() <= 0)
    {
      drrt_->return_home_ = true;
      geometry_msgs::Point home_position;
      home_position.x = 0;
      home_position.y = 0;
      home_position.z = 0;
      res.goal.push_back(home_position);
      res.mode.data = 2;  // mode 2 means returning home

      dual_state_frontier_->cleanAllUselessFrontiers();
      return true;
    }
    // else if (!drrt_->nextNodeFound_ && !drrt_->global_plan_pre_ && dual_state_graph_->getGain(robot_position) <= 0)
    else if (!drrt_->nextNodeFound_ && !drrt_->global_plan_pre_ &&  bestExplorationGain <= 0)
    {
      #if 1
      if (dist > 1 && waitLoop > 0) {
      #else
      std::vector<int> path;
      graph_utils_ns::ShortestPathBtwVertex(path, dual_state_graph_->local_graph_, 0, preId);
      bool path_exists = true;
      float dist_path = 0;
      // Check if there is an existing path
      if (path.empty())
      {
        // No path exists
        dist_path = 0;
      }
      else
      {
        // Compute path length
        dist_path = graph_utils_ns::PathLength(path, dual_state_graph_->local_graph_);
      }
      if (dist_path > 0.5) {
      #endif
        waitLoop--;
        drrt_->local_plan_ = true;
        res = preRes;
        // std::cout << "robot_position: " << robot_position.x << " " << robot_position.y << " " << robot_position.z << " ";
        // std::cout << "preRes: " << preRes.goal[0].x << " " << preRes.goal[0].y << " " << preRes.goal[0].z << " " << std::endl;
        return true;
      }
      waitLoop = 5;
      if(drrt_->localPlanOnceMore_) {
        drrt_->localPlanOnceMore_ = false;
        drrt_->local_plan_ = true;
        std::cout << "localPlanOnceMore!" << std::endl;
        return true;
      }
      drrt_->global_plan_ = true;
      drrt_->localPlanOnceMore_ = true;
      std::cout << "     No Remaining local frontiers  "
                << "\n"
                << "     Switch to relocation stage "
                << "\n"
                << "     Total plan lasted " << 0 << std::endl;
      return true;
    }
    else
    {
      drrt_->local_plan_ = true;
      drrt_->localPlanOnceMore_ = true;
    }
  }
  waitLoop = 5;
  gain_computation_over_ = std::chrono::steady_clock::now();
  time_span = gain_computation_over_ - RRT_generate_over_;
  double getGainTime =
      double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

  // Extract next goal.
  geometry_msgs::Point next_goal_position;
  if(drrt_->loopflag == true && drrt_->loopInit == true) {
      // if(drrt_->loopContinue == false){
      //   drrt_->loopContinue = true;
      //   dual_state_graph_->getGain_szz(robot_position, drrt_->twist_, dist);
      //   for(int ind = 2; ind < drrt_->loopPoint.size(); ind++) {
      //     if(misc_utils_ns::PointXYDist(drrt_->loopPoint[ind], dual_state_graph_->getBestLocalVertexPosition()) <= 2.5) {
      //       // std::swap(drrt_->loopPoint[ind], drrt_->loopPoint[drrt_->loopPoint.size()-1]);
      //       drrt_->loopPoint.erase(drrt_->loopPoint.begin()+ind);
      //       ind--;
      //     }
      //   }
      //   drrt_->tspTour = dual_state_graph_->solveTSP(drrt_->loopPoint);
      //   drrt_->tspTour.insert(drrt_->tspTour.begin(), dual_state_graph_->getBestLocalVertexPosition());
      //   drrt_->publishTSPPath();
        
      // }
      if(drrt_->tspTour.size() > 1) {
        dual_state_graph_->best_vertex_id_ = graph_utils_ns::GetClosestVertexIdxToPoint(dual_state_graph_->local_graph_, drrt_->tspTour[drrt_->tspTour.size()-1]);
        res.mode.data = 1;
        drrt_->local_plan_ = false;
        drrt_->global_plan_ = true;
        drrt_->tspTour.pop_back();
        dual_state_graph_->updateGlobalGraph();
        dual_state_graph_->updateExploreDirection();
        next_goal_position = dual_state_graph_->getBestGlobalVertexPosition();
      }
      else {
        dual_state_graph_->best_vertex_id_ = graph_utils_ns::GetClosestVertexIdxToPoint(dual_state_graph_->local_graph_, drrt_->tspTour[0]);
        res.mode.data = 4;
        drrt_->local_plan_ = true;
        drrt_->global_plan_ = false;
        drrt_->tspTour.clear();
        drrt_->loopPoint.clear();
        drrt_->publishTSPPath();
        dual_state_graph_->updateGlobalGraph();
        dual_state_graph_->updateExploreDirection();
        next_goal_position = dual_state_graph_->getBestGlobalVertexPosition();
      }
  }
  else if(drrt_->nextNodeFound_) {
    dual_state_graph_->best_vertex_id_ = drrt_->NextBestNodeIdx_;
    dual_state_graph_->updateExploreDirection();
    next_goal_position = dual_state_graph_->getBestGlobalVertexPosition();
  } 

  else if (drrt_->global_plan_pre_ == true && drrt_->gainFound())
  {
    dual_state_graph_->best_vertex_id_ = drrt_->bestNodeId_;
    dual_state_graph_->updateExploreDirection();
    next_goal_position = dual_state_graph_->getBestLocalVertexPosition();
    res.mode.data = 1;
  }
  else
  {
    dual_state_graph_->updateGlobalGraph();
    dual_state_graph_->updateExploreDirection();
    next_goal_position = dual_state_graph_->getBestLocalVertexPosition();
    res.mode.data = 1;
  }
  dual_state_graph_->setCurrentPlannerStatus(drrt_->global_plan_pre_);
  res.goal.push_back(next_goal_position);
  preId = dual_state_graph_->best_vertex_id_;
  preRes = res;

  // plan_computation_over_ = std::chrono::steady_clock::now();
  // time_span = plan_computation_over_ - gain_computation_over_;
  // double getPlanTime =
  //     double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

  geometry_msgs::PointStamped next_goal_point;
  next_goal_point.header.frame_id = "map";
  next_goal_point.point = next_goal_position;
  params_.nextGoalPub_.publish(next_goal_point);

  plan_over_ = std::chrono::steady_clock::now();
  time_span = plan_over_ - plan_start_;
  double plantime =
      double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
  std::cout << "     RRT generation lasted  " << rrtGenerateTime << "\n"
            << "     Computiong gain lasted " << getGainTime << "\n"
            << "     Total plan lasted " << plantime << std::endl;
  return true;
}

bool dsvplanner_ns::drrtPlanner::cleanFrontierServiceCallback(dsvplanner::clean_frontier_srv::Request& req,
                                                              dsvplanner::clean_frontier_srv::Response& res)
{
  // if (drrt_->nextNodeFound_)
  // {
  //   dual_state_frontier_->updateToCleanFrontier(drrt_->selectedGlobalFrontier_);
  //   dual_state_frontier_->gloabalFrontierUpdate();
  // }
  // else
  // {
  //   dual_state_graph_->clearLocalGraph();
  // }
  // res.success = true;

  return true;
}

void dsvplanner_ns::drrtPlanner::cleanLastSelectedGlobalFrontier()
{
  // only when last plan is global plan, this function will be executed to clear
  // last selected global
  // frontier.
  if (drrt_->nextNodeFound_)
  {
    dual_state_frontier_->updateToCleanFrontier(drrt_->selectedGlobalFrontier_);
    dual_state_frontier_->gloabalFrontierUpdate();
  }
}

bool dsvplanner_ns::drrtPlanner::setParams()
{
  nh_private_.getParam("/rm/kSensorPitch", params_.sensorPitch);
  nh_private_.getParam("/rm/kSensorHorizontal", params_.sensorHorizontalView);
  nh_private_.getParam("/rm/kSensorVertical", params_.sensorVerticalView);
  nh_private_.getParam("/rm/kVehicleHeight", params_.kVehicleHeight);
  nh_private_.getParam("/rm/kBoundX", params_.boundingBox[0]);
  nh_private_.getParam("/rm/kBoundY", params_.boundingBox[1]);
  nh_private_.getParam("/rm/kBoundZ", params_.boundingBox[2]);
  nh_private_.getParam("/drrt/gain/kFree", params_.kGainFree);
  nh_private_.getParam("/drrt/gain/kOccupied", params_.kGainOccupied);
  nh_private_.getParam("/drrt/gain/kUnknown", params_.kGainUnknown);
  nh_private_.getParam("/drrt/gain/kMinEffectiveGain", params_.kMinEffectiveGain);
  nh_private_.getParam("/drrt/gain/kRange", params_.kGainRange);
  nh_private_.getParam("/drrt/gain/kRangeZMinus", params_.kGainRangeZMinus);
  nh_private_.getParam("/drrt/gain/kRangeZPlus", params_.kGainRangeZPlus);
  nh_private_.getParam("/drrt/gain/kZero", params_.kZeroGain);
  nh_private_.getParam("/drrt/tree/kExtensionRange", params_.kExtensionRange);
  nh_private_.getParam("/drrt/tree/kMinExtensionRange", params_.kMinextensionRange);
  nh_private_.getParam("/drrt/tree/kMaxExtensionAlongZ", params_.kMaxExtensionAlongZ);
  nh_private_.getParam("/rrt/tree/kExactRoot", params_.kExactRoot);
  nh_private_.getParam("/drrt/tree/kCuttoffIterations", params_.kCuttoffIterations);
  nh_private_.getParam("/drrt/tree/kGlobalExtraIterations", params_.kGlobalExtraIterations);
  nh_private_.getParam("/drrt/tree/kRemainingNodeScaleSize", params_.kRemainingNodeScaleSize);
  nh_private_.getParam("/drrt/tree/kRemainingBranchScaleSize", params_.kRemainingBranchScaleSize);
  nh_private_.getParam("/drrt/tree/kNewNodeScaleSize", params_.kNewNodeScaleSize);
  nh_private_.getParam("/drrt/tree/kNewBranchScaleSize", params_.kNewBranchScaleSize);
  nh_private_.getParam("/drrt/tfFrame", params_.explorationFrame);
  nh_private_.getParam("/drrt/vertexSize", params_.kVertexSize);
  nh_private_.getParam("/drrt/keepTryingNum", params_.kKeepTryingNum);
  nh_private_.getParam("/drrt/kLoopCountThres", params_.kLoopCountThres);
  nh_private_.getParam("/lb/kMinXLocal", params_.kMinXLocalBound);
  nh_private_.getParam("/lb/kMinYLocal", params_.kMinYLocalBound);
  nh_private_.getParam("/lb/kMinZLocal", params_.kMinZLocalBound);
  nh_private_.getParam("/lb/kMaxXLocal", params_.kMaxXLocalBound);
  nh_private_.getParam("/lb/kMaxYLocal", params_.kMaxYLocalBound);
  nh_private_.getParam("/lb/kMaxZLocal", params_.kMaxZLocalBound);
  nh_private_.getParam("/gb/kMinXGlobal", params_.kMinXGlobalBound);
  nh_private_.getParam("/gb/kMinYGlobal", params_.kMinYGlobalBound);
  nh_private_.getParam("/gb/kMinZGlobal", params_.kMinZGlobalBound);
  nh_private_.getParam("/gb/kMaxXGlobal", params_.kMaxXGlobalBound);
  nh_private_.getParam("/gb/kMaxYGlobal", params_.kMaxYGlobalBound);
  nh_private_.getParam("/gb/kMaxZGlobal", params_.kMaxZGlobalBound);
  nh_private_.getParam("/elevation/kTerrainVoxelSize", params_.kTerrainVoxelSize);
  nh_private_.getParam("/elevation/kTerrainVoxelWidth", params_.kTerrainVoxelWidth);
  nh_private_.getParam("/elevation/kTerrainVoxelHalfWidth", params_.kTerrainVoxelHalfWidth);
  nh_private_.getParam("/planner/odomSubTopic", odomSubTopic);
  nh_private_.getParam("/planner/scanSubTopic", scanSubTopic);
  nh_private_.getParam("/planner/boundarySubTopic", boundarySubTopic);
  nh_private_.getParam("/planner/newTreePathPubTopic", newTreePathPubTopic);
  nh_private_.getParam("/planner/remainingTreePathPubTopic", remainingTreePathPubTopic);
  nh_private_.getParam("/planner/boundaryPubTopic", boundaryPubTopic);
  nh_private_.getParam("/planner/globalSelectedFrontierPubTopic", globalSelectedFrontierPubTopic);
  nh_private_.getParam("/planner/localSelectedFrontierPubTopic", localSelectedFrontierPubTopic);
  nh_private_.getParam("/planner/plantimePubTopic", plantimePubTopic);
  nh_private_.getParam("/planner/nextGoalPubTopic", nextGoalPubTopic);
  nh_private_.getParam("/planner/randomSampledPointsPubTopic", randomSampledPointsPubTopic);
  nh_private_.getParam("/planner/shutDownTopic", shutDownTopic);
  nh_private_.getParam("/planner/plannerServiceName", plannerServiceName);
  nh_private_.getParam("/planner/cleanFrontierServiceName", cleanFrontierServiceName);

  return true;
}

bool dsvplanner_ns::drrtPlanner::init()
{
  if (!setParams())
  {
    ROS_ERROR("Set parameters fail. Cannot start planning!");
  }
  pathSub_ = nh_.subscribe("/pose_with_cov", 10, &dsvplanner_ns::drrtPlanner::posecovCallback, this);
  odomSub_.subscribe(nh_, odomSubTopic, 1);
  scanSub_.subscribe(nh_, scanSubTopic, 1);
  sync_.reset(new Sync(syncPolicy(100), odomSub_, scanSub_));
  sync_->registerCallback(boost::bind(&dsvplanner_ns::drrtPlanner::scanAndOdomSyncHandler, this, _1, _2));
  // odomSub_ = nh_.subscribe(odomSubTopic, 10, &dsvplanner_ns::drrtPlanner::odomCallback, this);
  boundarySub_ = nh_.subscribe(boundarySubTopic, 10, &dsvplanner_ns::drrtPlanner::boundaryCallback, this);

  params_.newTreePathPub_ = nh_.advertise<visualization_msgs::Marker>(newTreePathPubTopic, 1000);
  params_.remainingTreePathPub_ = nh_.advertise<visualization_msgs::Marker>(remainingTreePathPubTopic, 1000);
  params_.boundaryPub_ = nh_.advertise<visualization_msgs::Marker>(boundaryPubTopic, 1000);
  params_.globalSelectedFrontierPub_ = nh_.advertise<sensor_msgs::PointCloud2>(globalSelectedFrontierPubTopic, 1000);
  params_.localSelectedFrontierPub_ = nh_.advertise<sensor_msgs::PointCloud2>(localSelectedFrontierPubTopic, 1000);
  params_.randomSampledPointsPub_ = nh_.advertise<sensor_msgs::PointCloud2>(randomSampledPointsPubTopic, 1000);
  params_.plantimePub_ = nh_.advertise<std_msgs::Float32>(plantimePubTopic, 1000);
  params_.nextGoalPub_ = nh_.advertise<geometry_msgs::PointStamped>(nextGoalPubTopic, 1000);
  params_.shutdownSignalPub = nh_.advertise<std_msgs::Bool>(shutDownTopic, 1000);
  params_.polygonsPub_ = nh_.advertise<visualization_msgs::Marker>("/concavehull_polygon", 1000);
  params_.tspPathPub_ = nh_.advertise<nav_msgs::Path>("/tsp_path", 1000);
  // params_.posePub_ = nh_.advertise<geometry_msgs::PoseStamped>("/exploration_direction", 1000);
  params_.posePub_ = nh_.advertise<visualization_msgs::Marker>("/robot_direction", 1000);
  plannerService_ = nh_.advertiseService(plannerServiceName, &dsvplanner_ns::drrtPlanner::plannerServiceCallback, this);
  cleanFrontierService_ =
      nh_.advertiseService(cleanFrontierServiceName, &dsvplanner_ns::drrtPlanner::cleanFrontierServiceCallback, this);

  return true;
}
