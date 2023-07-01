/*
drrt.cpp
Implementation of Drrt class. Dynamic tree is used to get random viewpoints in
local area. New rrt
is generated based on the pruned tree of last time.

Created by Hongbiao Zhu (hongbiaz@andrew.cmu.edu)
05/25/2020
*/

#ifndef RRTTREE_HPP_
#define RRTTREE_HPP_

#include <cstdlib>
#include <dsvplanner/drrt.h>
#include <misc_utils/misc_utils.h>

#include <concave2d/az_concave2d.hpp>
#include <dsvplanner/chunkmap_helper.h>
#include <visualization_msgs/MarkerArray.h>

// #define outfile

dsvplanner_ns::Drrt::Drrt(ChunkMap::Ptr chunk_map, DualStateGraph* graph,
                          DualStateFrontier* frontier, OccupancyGrid* grid)
{
  // manager_ = manager;
  chunk_map_ = chunk_map;
  grid_ = grid;
  dual_state_graph_ = graph;
  dual_state_frontier_ = frontier;

  ROS_INFO("Successfully launched Drrt node");
}

dsvplanner_ns::Drrt::~Drrt()
{
  delete rootNode_;
  kd_free(kdTree_);
}

void dsvplanner_ns::Drrt::init()
{
  kdTree_ = kd_create(3);
  iterationCount_ = 0;
  bestGain_ = params_.kZeroGain;
  bestNode_ = NULL;
  rootNode_ = NULL;
  nodeCounter_ = 0;
  plannerReady_ = false;
  boundaryLoaded_ = false;
  global_plan_ = false;
  global_plan_pre_ = true;
  local_plan_ = true;
  nextNodeFound_ = false;
  remainingFrontier_ = true;
  return_home_ = false;
  global_vertex_size_ = 0;
  NextBestNodeIdx_ = 0;
  loopflag = false;
  localPlanOnceMore_ = true;
  for (int i = 0; i < params_.kTerrainVoxelWidth * params_.kTerrainVoxelWidth; i++)
  {
    terrain_voxle_elev_.push_back(params_.kVehicleHeight);
  }

  srand((unsigned)time(NULL));
}

void dsvplanner_ns::Drrt::setParams(Params params)
{
  params_ = params;
}

void dsvplanner_ns::Drrt::setRootWithOdom(const nav_msgs::Odometry& pose)
{
  StateVec pre_root = root_;
  root_[0] = pose.pose.pose.position.x;
  root_[1] = pose.pose.pose.position.y;
  root_[2] = pose.pose.pose.position.z;
  transformToMap.setOrigin(
      tf::Vector3(pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z));
  transformToMap.setRotation(tf::Quaternion(pose.pose.pose.orientation.x, pose.pose.pose.orientation.y,
                                            pose.pose.pose.orientation.z, pose.pose.pose.orientation.w) * tf::createQuaternionFromRPY(0, 0, -1.57));

  // geometry_msgs::PoseStamped temp;
  // temp.header.stamp = ros::Time::now();
  // temp.header.frame_id = "map";
  // temp.pose.position = pose.pose.pose.position;
  // tf::Quaternion q = tf::Quaternion(pose.pose.pose.orientation.x, pose.pose.pose.orientation.y,
  //                                           pose.pose.pose.orientation.z, pose.pose.pose.orientation.w);
  // temp.pose.orientation.x = q.x();
  // temp.pose.orientation.y = q.y();
  // temp.pose.orientation.z = q.z();
  // temp.pose.orientation.w = q.w();
  // params_.posePub_.publish(temp);

  tf::Vector3 point(1,0,0);
  tf::StampedTransform temp_tf;
  temp_tf.setOrigin(
      tf::Vector3(pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z));
  temp_tf.setRotation(tf::Quaternion(pose.pose.pose.orientation.x, pose.pose.pose.orientation.y,
                                            pose.pose.pose.orientation.z, pose.pose.pose.orientation.w));
  point = temp_tf * point;
  twist_[0] = point.x();
  twist_[1] = point.y();
  twist_[2] = point.z();
  publishPose(twist_);
}

void dsvplanner_ns::Drrt::publishPose(StateVec twist_) {
  visualization_msgs::Marker poseMarker;
  poseMarker.ns = "pose";
  poseMarker.header.stamp = ros::Time::now();
  poseMarker.header.frame_id = "map";
  poseMarker.type = visualization_msgs::Marker::ARROW;
  poseMarker.action = visualization_msgs::Marker::ADD;
  poseMarker.scale.x = 1;
  poseMarker.scale.y = 1;
  poseMarker.scale.z = 1;
  poseMarker.color.r = 255.0 / 255.0;
  poseMarker.color.g = 255.0 / 255.0;
  poseMarker.color.b = 255.0 / 255.0;
  poseMarker.color.a = 1.0;
  poseMarker.frame_locked = false;
  
  geometry_msgs::Point point1, point2;
  point1.x = root_[0];
  point1.y = root_[1];
  point1.z = root_[2];
  point2.x = twist_[0];
  point2.y = twist_[1];
  point2.z = twist_[2];
  poseMarker.points.push_back(point1);
  poseMarker.points.push_back(point2);

  params_.posePub_.publish(poseMarker);
}

void dsvplanner_ns::Drrt::setBoundary(const geometry_msgs::PolygonStamped& boundary)
{
  boundary_polygon_ = boundary.polygon;
  boundaryLoaded_ = true;
}

void dsvplanner_ns::Drrt::setTerrainVoxelElev()
{
  if (dual_state_frontier_->getTerrainVoxelElev().size() > 0)
  {
    terrain_voxle_elev_.clear();
    terrain_voxle_elev_ = dual_state_frontier_->getTerrainVoxelElev();
  }
}

int dsvplanner_ns::Drrt::getNodeCounter()
{
  return nodeCounter_;
}

int dsvplanner_ns::Drrt::getRemainingNodeCounter()
{
  return remainingNodeCount_;
}

bool dsvplanner_ns::Drrt::gainFound()
{
  return bestGain_ > params_.kZeroGain;
}

double dsvplanner_ns::Drrt::angleDiff(StateVec direction1, StateVec direction2)
{
  double degree;
  degree = acos((direction1[0] * direction2[0] + direction1[1] * direction2[1]) /
                (sqrt(direction1[0] * direction1[0] + direction1[1] * direction1[1]) *
                 sqrt(direction2[0] * direction2[0] + direction2[1] * direction2[1]))) *
           180 / M_PI;
  return degree;
}

double dsvplanner_ns::Drrt::getZvalue(double x_position, double y_position)
{
  int indX =
      int((x_position + params_.kTerrainVoxelSize / 2) / params_.kTerrainVoxelSize) + params_.kTerrainVoxelHalfWidth;
  int indY =
      int((y_position + params_.kTerrainVoxelSize / 2) / params_.kTerrainVoxelSize) + params_.kTerrainVoxelHalfWidth;
  if (x_position + params_.kTerrainVoxelSize / 2 < 0)
    indX--;
  if (y_position + params_.kTerrainVoxelSize / 2 < 0)
    indY--;
  if (indX > params_.kTerrainVoxelWidth - 1)
    indX = params_.kTerrainVoxelWidth - 1;
  if (indX < 0)
    indX = 0;
  if (indY > params_.kTerrainVoxelWidth - 1)
    indY = params_.kTerrainVoxelWidth - 1;
  if (indY < 0)
    indY = 0;
  double z_position = terrain_voxle_elev_[params_.kTerrainVoxelWidth * indX + indY] + params_.kVehicleHeight;
  return z_position;
}

bool dsvplanner_ns::Drrt::inSensorRange(StateVec& node)
{
  StateVec root_node(rootNode_->state_[0], rootNode_->state_[1], rootNode_->state_[2]);
  StateVec init_node = node;
  StateVec dir;
  bool insideFieldOfView = false;
  for (int i = 0; i < localThreeFrontier_->points.size(); i++)
  {
    StateVec frontier_point(localThreeFrontier_->points[i].x, localThreeFrontier_->points[i].y,
                            localThreeFrontier_->points[i].z);
    node[0] = init_node[0] + frontier_point[0];
    node[1] = init_node[1] + frontier_point[1];
    double x_position = node[0] - root_node[0];
    double y_position = node[1] - root_node[1];
    node[2] = getZvalue(x_position, y_position);
    if (!inPlanningBoundary(node))
      continue;

    dir = frontier_point - node;
    // Skip if distance to sensor is too large
    double rangeSq = pow(params_.kGainRange, 2.0);
    if (dir.transpose().dot(dir) > rangeSq)
    {
      continue;
    }

    if (fabs(dir[2] < sqrt(dir[0] * dir[0] + dir[1] * dir[1]) * tan(M_PI * params_.sensorVerticalView / 360)))
    {
      insideFieldOfView = true;
    }
    if (!insideFieldOfView)
    {
      continue;
    }

    // if (manager_->getCellStatusPoint(node) == volumetric_mapping::OctomapManager::CellStatus::kFree)
    if (ChunkmapGetCellStatusPoint(chunk_map_, node) == ChunkCellStatus::FREE)
    {
      // if (volumetric_mapping::OctomapManager::CellStatus::kOccupied !=
      //     this->manager_->getVisibility(node, frontier_point, false))
      if (ChunkCellStatus::OCCUPIED != ChunkmapGetVisibility(chunk_map_, node, frontier_point, false))
      {
        return true;
      }
    }
  }
  return false;
}

bool dsvplanner_ns::Drrt::inPlanningBoundary(StateVec node)
{
  if (node.x() < minX_ + 0.5 * params_.boundingBox.x())
  {
    return false;
  }
  else if (node.y() < minY_ + 0.5 * params_.boundingBox.y())
  {
    return false;
  }
  else if (node.z() < minZ_ + 0.5 * params_.boundingBox.z())
  {
    return false;
  }
  else if (node.x() > maxX_ - 0.5 * params_.boundingBox.x())
  {
    return false;
  }
  else if (node.y() > maxY_ - 0.5 * params_.boundingBox.y())
  {
    return false;
  }
  else if (node.z() > maxZ_ - 0.5 * params_.boundingBox.z())
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool dsvplanner_ns::Drrt::inGlobalBoundary(StateVec node)
{
  if (boundaryLoaded_)
  {
    geometry_msgs::Point node_point;
    node_point.x = node.x();
    node_point.y = node.y();
    node_point.z = node.z();
    if (!misc_utils_ns::PointInPolygon(node_point, boundary_polygon_))
    {
      return false;
    }
  }
  else
  {
    if (node.x() < params_.kMinXGlobalBound + 0.5 * params_.boundingBox.x())
    {
      return false;
    }
    else if (node.y() < params_.kMinYGlobalBound + 0.5 * params_.boundingBox.y())
    {
      return false;
    }
    else if (node.x() > params_.kMaxXGlobalBound - 0.5 * params_.boundingBox.x())
    {
      return false;
    }
    else if (node.y() > params_.kMaxYGlobalBound - 0.5 * params_.boundingBox.y())
    {
      return false;
    }
  }
  if (node.z() > params_.kMaxZGlobalBound - 0.5 * params_.boundingBox.z())
  {
    return false;
  }
  else if (node.z() < params_.kMinZGlobalBound + 0.5 * params_.boundingBox.z())
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool dsvplanner_ns::Drrt::generateRrtNodeToLocalFrontier(StateVec& newNode)
{
  StateVec potentialNode;
  bool nodeFound = false;
  int count = 0;
  double radius = sqrt(SQ(params_.kGainRange) + SQ(params_.kGainRange));
  while (!nodeFound)
  {
    count++;
    if (count >= 300)
    {
      return false;
    }
    potentialNode[0] = 2.0 * radius * (((double)rand()) / ((double)RAND_MAX) - 0.5);
    potentialNode[1] = 2.0 * radius * (((double)rand()) / ((double)RAND_MAX) - 0.5);
    potentialNode[2] = 0;
    if ((SQ(potentialNode[0]) + SQ(potentialNode[1])) > pow(radius, 2.0))
      continue;

    if (!inSensorRange(potentialNode))
    {
      continue;
    }

    if (!inPlanningBoundary(potentialNode) || !inGlobalBoundary(potentialNode))
    {
      continue;
    }
    nodeFound = true;
    newNode[0] = potentialNode[0];
    newNode[1] = potentialNode[1];
    newNode[2] = potentialNode[2];
    return true;
  }
  return false;
}

void dsvplanner_ns::Drrt::getNextNodeToClosestGlobalFrontier()
{
  StateVec p1, p2;
  pcl::PointXYZ p3;
  double length1, length2;
  pcl::PointCloud<pcl::PointXYZ>::Ptr globalSelectedFrontier =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  nextNodeFound_ = false;
  for (int i = dual_state_graph_->local_graph_.vertices.size() - 1; i >= 0; i--)  // Search from end to the begining.
                                                                                  // Nodes with large index means they
                                                                                  // are closer to the current position
                                                                                  // of the robot
  {
    p1.x() = dual_state_graph_->local_graph_.vertices[i].location.x;
    p1.y() = dual_state_graph_->local_graph_.vertices[i].location.y;
    p1.z() = dual_state_graph_->local_graph_.vertices[i].location.z;
    for (int j = 0; j < dual_state_frontier_->global_frontier_->size(); j++)
    {
      p3 = dual_state_frontier_->global_frontier_->points[j];
      p2.x() = dual_state_frontier_->global_frontier_->points[j].x;
      p2.y() = dual_state_frontier_->global_frontier_->points[j].y;
      p2.z() = dual_state_frontier_->global_frontier_->points[j].z;
      length1 = sqrt(SQ(p1.x() - p2.x()) + SQ(p1.y() - p2.y()));
      if (length1 > (/*this->manager_->getSensorMaxRange() TODO: */ 20.0f + params_.kGainRange) ||
          fabs(p1.z() - p2.z()) > params_.kMaxExtensionAlongZ)  // Not only
                                                                // consider the
      // sensor range,
      // also take the
      // node's view range into consideration
      {
        continue;  // When the node is too far away from the frontier or the
                   // difference between the z
      }            // of this node and frontier is too large, then skip to next frontier.
      // No need to use FOV here.
      // if (volumetric_mapping::OctomapManager::CellStatus::kOccupied == manager_->getVisibility(p1, p2, false))
      if (ChunkCellStatus::OCCUPIED == ChunkmapGetVisibility(chunk_map_, p1, p2, false))
      {
        continue;  // Only when there is no occupied voxels between the node
                   // and
                   // frontier, we consider
      }            // the node can potentially see this frontier
      else
      {
        NextBestNodeIdx_ = i;
        nextNodeFound_ = true;
        globalSelectedFrontier->points.push_back(p3);
        selectedGlobalFrontier_ = p3;
        break;
      }
    }
    if (nextNodeFound_)
      break;
  }
  if (nextNodeFound_)
  {
    for (int i = dual_state_graph_->local_graph_.vertices.size() - 1; i >= 0; i--)
    {
      p1.x() = dual_state_graph_->local_graph_.vertices[i].location.x;
      p1.y() = dual_state_graph_->local_graph_.vertices[i].location.y;
      p1.z() = dual_state_graph_->local_graph_.vertices[i].location.z;
      length2 = sqrt(SQ(p1.x() - p2.x()) + SQ(p1.y() - p2.y()));
      if (length2 > length1 || fabs(p1.z() - p2.z()) > params_.kMaxExtensionAlongZ)
      {
        continue;
      }
      // if (volumetric_mapping::OctomapManager::CellStatus::kOccupied ==
      //     manager_->getLineStatusBoundingBox(p1, p2, params_.boundingBox))
      if (ChunkCellStatus::OCCUPIED == ChunkmapGetLineStatusBoundingBox(chunk_map_, p1, p2, params_.boundingBox))
      {
        continue;
      }
      length1 = length2;
      NextBestNodeIdx_ = i;
      nextNodeFound_ = true;
      p3.x = p1.x();
      p3.y = p1.y();
      p3.z = p1.z();
    }
    globalSelectedFrontier->points.push_back(p3);
  }
  sensor_msgs::PointCloud2 globalFrontier;
  pcl::toROSMsg(*globalSelectedFrontier, globalFrontier);
  globalFrontier.header.frame_id = params_.explorationFrame;
  params_.globalSelectedFrontierPub_.publish(globalFrontier);  // publish the next goal node and corresponing frontier
}

void dsvplanner_ns::Drrt::getThreeLocalFrontierPoint()  // Three local frontiers
                                                        // that are most close to
                                                        // the last
                                                        // exploration direciton
{                                                       // will be selected.
  StateVec exploreDirection, frontierDirection;
  double firstDirection = 180, secondDirection = 180, thirdDirection = 180;
  pcl::PointXYZ p1, p2, p3;  // three points to save frontiers.
  exploreDirection = dual_state_graph_->getExploreDirection();
  int localFrontierSize = dual_state_frontier_->local_frontier_->size();
  for (int i = 0; i < localFrontierSize; i++)
  {
    frontierDirection[0] =
        dual_state_frontier_->local_frontier_->points[i].x - root_[0];  // For ground robot, we only consider the
                                                                        // direction along x-y plane
    frontierDirection[1] = dual_state_frontier_->local_frontier_->points[i].y - root_[1];
    double theta = angleDiff(frontierDirection, exploreDirection);
    if (theta < firstDirection)
    {
      thirdDirection = secondDirection;
      secondDirection = firstDirection;
      firstDirection = theta;
      frontier3_direction_ = frontier2_direction_;
      frontier2_direction_ = frontier1_direction_;
      frontier1_direction_ = frontierDirection;
      p3 = p2;
      p2 = p1;
      p1 = dual_state_frontier_->local_frontier_->points[i];
    }
    else if (theta < secondDirection)
    {
      thirdDirection = secondDirection;
      secondDirection = theta;
      frontier3_direction_ = frontier2_direction_;
      frontier2_direction_ = frontierDirection;
      p3 = p2;
      p2 = dual_state_frontier_->local_frontier_->points[i];
    }
    else if (theta < thirdDirection)
    {
      thirdDirection = theta;
      frontier3_direction_ = frontierDirection;
      p3 = dual_state_frontier_->local_frontier_->points[i];
    }
  }

  localThreeFrontier_->clear();
  localThreeFrontier_->points.push_back(p1);
  localThreeFrontier_->points.push_back(p2);
  localThreeFrontier_->points.push_back(p3);
  sensor_msgs::PointCloud2 localThreeFrontier;
  pcl::toROSMsg(*localThreeFrontier_, localThreeFrontier);
  localThreeFrontier.header.frame_id = params_.explorationFrame;
  params_.localSelectedFrontierPub_.publish(localThreeFrontier);
}

bool dsvplanner_ns::Drrt::remainingLocalFrontier()
{
  int localFrontierSize = dual_state_frontier_->local_frontier_->points.size();
  if (localFrontierSize > 0)
    return true;
  return false;
}

void dsvplanner_ns::Drrt::plannerIterate()
{
  // In this function a new configuration is sampled and added to the tree.
  StateVec newState;
  bool generateNodeArroundFrontierSuccess = false;

  double radius = 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_));
  bool candidateFound = false;
  int count = 0;
  while (!candidateFound)
  {
    count++;
    if (count > 1000)
      return;  // Plan fail if cannot find a required node in 1000 iterations
    
    if (((double)rand()) / ((double)RAND_MAX) > 1 && localThreeFrontier_->size() > 0)
    {
      if (local_plan_ == true)
      {
        generateNodeArroundFrontierSuccess = generateRrtNodeToLocalFrontier(newState);
      }
      if (!generateNodeArroundFrontierSuccess)
      {  // Generate node near local
         // frontier fail
        newState[0] = 2.0 * radius * (((double)rand()) / ((double)RAND_MAX) - 0.5);
        newState[1] = 2.0 * radius * (((double)rand()) / ((double)RAND_MAX) - 0.5);
        newState[2] = 0;  // Do not consider z value because ground robot cannot
                          // move along z
        if (SQ(newState[0]) + SQ(newState[1]) > pow(radius, 2.0))
          continue;
        newState[0] += root_[0];
        newState[1] += root_[1];
        newState[2] += root_[2];
        if ((!inPlanningBoundary(newState)) || (!inGlobalBoundary(newState)))
        {
          continue;
        }
      }
    }
    else {
      #if 0
        newState[0] = 2.0 * radius * (((double)rand()) / ((double)RAND_MAX) - 0.5);
        newState[1] = 2.0 * radius * (((double)rand()) / ((double)RAND_MAX) - 0.5);
        newState[2] = 0;
        if (SQ(newState[0]) + SQ(newState[1]) > pow(radius, 2.0))
          continue;
        newState[0] += root_[0];
        newState[1] += root_[1];
        newState[2] += root_[2];
        if ((!inPlanningBoundary(newState)) || (!inGlobalBoundary(newState)))
        {
          continue;
        }
      #else
        static std::mt19937 rdGen;
        int ind = discreteProb(rdGen);
        // double rand01 = ((double)rand()) / ((double)RAND_MAX);
        // int ind;
        // for (ind = 0; ind < 360; ind++) {
        //   if (rand01 <= probIntegration[ind])
        //     break;
        // }
        // ind %= 360;
        double theta = (ind - 0.5 + ((double)rand()) / ((double)RAND_MAX)) * M_PI / 180.0;
        double s = samplePolar[ind] *  (((double)rand()) / ((double)RAND_MAX));
        double r = diffPolar[ind] + s;
        tf::Vector3 initPoint(r * cos(theta), r * sin(theta), root_[2]);
        tf::Vector3 rotPoint = transformToMap * initPoint;
        newState[0] = rotPoint.getX();
        newState[1] = rotPoint.getY();
        newState[2] = root_[2];
        // newState[0] = r * cos(theta) + root_[0];
        // newState[1] = r * sin(theta) + root_[1];
        // newState[2] = root_[2];
        if ((!inPlanningBoundary(newState)) || (!inGlobalBoundary(newState)))
        {
          continue;
        }
      #endif
    }
    candidateFound = true;
  }

  pcl::PointXYZI sampledPoint;
  sampledPoint.x = newState[0];
  sampledPoint.y = newState[1];
  sampledPoint.z = newState[2];
  sampledPoint_->points.push_back(sampledPoint);

  // Find nearest neighbour
  kdres* nearest = kd_nearest3(kdTree_, newState.x(), newState.y(), newState.z());
  if (kd_res_size(nearest) <= 0)
  {
    kd_res_free(nearest);
    return;
  }
  dsvplanner_ns::Node* newParent = (dsvplanner_ns::Node*)kd_res_item_data(nearest);
  kd_res_free(nearest);

  // Check for collision of new connection.
  StateVec origin(newParent->state_[0], newParent->state_[1], newParent->state_[2]);
  StateVec direction(newState[0] - origin[0], newState[1] - origin[1], 0);
  if (direction.norm() < params_.kMinextensionRange)
  {
    return;
  }
  else if (direction.norm() > params_.kExtensionRange)
  {
    direction = params_.kExtensionRange * direction.normalized();
  }
  StateVec endPoint = origin + direction;
  newState[0] = endPoint[0];
  newState[1] = endPoint[1];

  double x_position = newState[0] - root_[0];
  double y_position = newState[1] - root_[1];
  newState[2] = getZvalue(x_position, y_position);

  if (newState[2] >= 1000)  // the sampled position is above the untraversed area
  {
    return;
  }
  // Check if the new node is too close to any existing nodes after extension
  kdres* nearest_node = kd_nearest3(kdTree_, newState.x(), newState.y(), newState.z());
  if (kd_res_size(nearest_node) <= 0)
  {
    kd_res_free(nearest_node);
    return;
  }
  dsvplanner_ns::Node* nearestNode = (dsvplanner_ns::Node*)kd_res_item_data(nearest_node);
  kd_res_free(nearest_node);

  origin[0] = newParent->state_[0];
  origin[1] = newParent->state_[1];
  origin[2] = newParent->state_[2];
  direction[0] = newState[0] - newParent->state_[0];
  direction[1] = newState[1] - newParent->state_[1];
  direction[2] = newState[2] - newParent->state_[2];
  if (direction.norm() < params_.kMinextensionRange || direction[2] > params_.kMaxExtensionAlongZ)
  {
    return;
  }
  // check collision if the new node is in the planning boundary
  if (!inPlanningBoundary(newState))
  {
    return;
  }
  else if((newState.x() > params_.kMinXGlobalBound && newState.x() < params_.kMinXGlobalBound + 2) || 
          (newState.y() > params_.kMinYGlobalBound && newState.y() < params_.kMinYGlobalBound + 2) || 
          (newState.x() < params_.kMaxXGlobalBound && newState.x() > params_.kMaxXGlobalBound - 2) || 
          (newState.y() < params_.kMaxYGlobalBound && newState.y() > params_.kMaxYGlobalBound - 2)) {
            sampledPoint.x = newState[0];
            sampledPoint.y = newState[1];
            sampledPoint.z = newState[2];
            sampledPoint.intensity = 10;
            dual_state_graph_->collisionPoint_->push_back(sampledPoint);

            obstaclePoint_->push_back(sampledPoint);
        }
  else
  {
    auto theStatus = ChunkmapGetLineStatusBoundingBox(chunk_map_, origin, newState, params_.boundingBox);
    // if (volumetric_mapping::OctomapManager::CellStatus::kFree ==
    //         manager_->getLineStatusBoundingBox(origin, newState, params_.boundingBox) &&
    if ((ChunkCellStatus::FREE == theStatus || ChunkCellStatus::UNKNOWN == theStatus) &&
        (!grid_->collisionCheckByTerrainWithVector(origin, newState)))
    {  // connection is free
      // Create new node and insert into tree
      if(ChunkCellStatus::FREE == theStatus)
      {
        dsvplanner_ns::Node* newNode = new dsvplanner_ns::Node;
        newNode->state_ = newState;
        newNode->parent_ = newParent;
        newNode->distance_ = newParent->distance_ + direction.norm();
        newParent->children_.push_back(newNode);
        // newNode->gain_ = gain(newNode->state_);

      kd_insert3(kdTree_, newState.x(), newState.y(), newState.z(), newNode);

          // Display new node
          node_array.push_back(newNode);
          // Update best IG and node if applicable
          // if (newNode->gain_ > bestGain_)
          // {
          //   bestGain_ = newNode->gain_;
          // }
          nodeCounter_++;
      }

      sampledPoint.x = newState[0];
      sampledPoint.y = newState[1];
      sampledPoint.z = newState[2];
      sampledPoint.intensity = 0;
      dual_state_graph_->collisionPoint_->push_back(sampledPoint);
    }
    else
    {
      sampledPoint.x = newState[0];
      sampledPoint.y = newState[1];
      sampledPoint.z = newState[2];
      sampledPoint.intensity = 10;
      dual_state_graph_->collisionPoint_->push_back(sampledPoint);

      obstaclePoint_->push_back(sampledPoint);
    }
  }
}

void dsvplanner_ns::Drrt::getConcaveHull()
{
  if(dual_state_graph_->collisionPoint_->size() <= 10)
    return;
  concavePoint_->clear();
  pclPolygons.clear();
  pcl::VoxelGrid<pcl::PointXYZI> vox;
  vox.setInputCloud(dual_state_graph_->collisionPoint_);
  vox.setLeafSize(0.5,0.5,0.5);
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp = 
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
  vox.filter(*temp);
#if 0
  concaveHull.setInputCloud(temp);
  concaveHull.setAlpha(2);
  concaveHull.setDimension(2);
  concaveHull.setKeepInformation(true);
  concaveHull.reconstruct(*concavePoint_, pclPolygons);
#else
  // pclPolygons = concave2d::concave2d(*temp, 2.5); // indoor
  // pclPolygons = concave2d::concave2d(*temp, 3.5); // garage
  pclPolygons = concave2d::concave2d(*temp, 2.5); // UM
  concavePoint_ = temp;
#endif

  dual_state_graph_->newObstacle(obstaclePoint_);
  obstaclePoint_->clear();
  publishConcaveHull();
  return;
}

void dsvplanner_ns::Drrt::publishConcaveHull() {
  // concave hull polygons
  visualization_msgs::Marker polygonsMarker;
  polygonsMarker.ns = "concavehull_polygons";
  polygonsMarker.header.stamp = ros::Time::now();
  polygonsMarker.header.frame_id = params_.explorationFrame;
  polygonsMarker.type = visualization_msgs::Marker::LINE_LIST;
  polygonsMarker.action = visualization_msgs::Marker::ADD;
  polygonsMarker.scale.x = params_.kRemainingBranchScaleSize * 3;
  polygonsMarker.color.r = 255.0 / 255.0;
  polygonsMarker.color.g = 255.0 / 255.0;
  polygonsMarker.color.b = 255.0 / 255.0;
  polygonsMarker.color.a = 1.0;
  polygonsMarker.frame_locked = false;

  geometry_msgs::Point polygonVertices1, polygonVertices2;
  for(int i = 0; i < pclPolygons.size(); i++) {
    if (pclPolygons[i].vertices.size() < 1) continue;
    polygonVertices1.x = concavePoint_->points[pclPolygons[i].vertices[pclPolygons[i].vertices.size()-1]].x;
    polygonVertices1.y = concavePoint_->points[pclPolygons[i].vertices[pclPolygons[i].vertices.size()-1]].y;
    polygonVertices1.z = concavePoint_->points[pclPolygons[i].vertices[pclPolygons[i].vertices.size()-1]].z;
    for(int j = 0; j < pclPolygons[i].vertices.size(); j++) {
      polygonVertices2.x = concavePoint_->points[pclPolygons[i].vertices[j]].x;
      polygonVertices2.y = concavePoint_->points[pclPolygons[i].vertices[j]].y;
      polygonVertices2.z = concavePoint_->points[pclPolygons[i].vertices[j]].z;
      polygonsMarker.points.push_back(polygonVertices1);
      polygonsMarker.points.push_back(polygonVertices2);
      polygonVertices1 = polygonVertices2;
    }
  }
  params_.polygonsPub_.publish(polygonsMarker);

}

void dsvplanner_ns::Drrt::updateGraphGain() {

  std::ofstream outFile;
  outFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/graphgain.txt");
  for(int i = 0; i < node_array.size(); i++) {
    #ifdef outfile
    outFile << node_array[i]->state_.x() << " " << node_array[i]->state_.y() << " " << node_array[i]->state_.z() 
     << " " << gain(node_array[i]->state_) / pow(chunk_map_->resolution(), 3.0)
     << " " << graphGain_concave(node_array[i]->state_) / pow(chunk_map_->resolution(), 3.0)  
     << std::endl;
    #endif
    
    // node_array[i]->gain_ = gain(node_array[i]->state_);
    node_array[i]->gain_ = graphGain_concave(node_array[i]->state_);

    geometry_msgs::Pose p1;
    p1.position.x = node_array[i]->state_.x();
    p1.position.y = node_array[i]->state_.y();
    p1.position.z = node_array[i]->state_.z();
    p1.orientation.y = node_array[i]->gain_;
    dual_state_graph_->addNewLocalVertexWithoutDuplicates(p1, dual_state_graph_->local_graph_);
    
    if (node_array[i]->gain_ > bestGain_)
      {
        bestGain_ = node_array[i]->gain_;
      }
    dual_state_graph_->execute();
  }
  
  outFile.close();
}

void dsvplanner_ns::Drrt::plannerInit()
{
  // This function is to initialize the tree
  kdTree_ = kd_create(3);

  node_array.clear();
  rootNode_ = new Node;
  rootNode_->distance_ = 0.0;
  rootNode_->gain_ = params_.kZeroGain;
  rootNode_->parent_ = NULL;

  global_vertex_size_ = 0;
  geometry_msgs::Pose p1;
  if (global_plan_ == false)
  {
    std::cout << "Exploration Stage" << std::endl;
    rootNode_->state_ = root_;
    kd_insert3(kdTree_, rootNode_->state_.x(), rootNode_->state_.y(), rootNode_->state_.z(), rootNode_);
    iterationCount_++;

    if (remainingLocalFrontier())
    {
      localPlanOnceMore_ = true;
      loopCount_ = params_.kLoopCountThres;
      normal_local_iteration_ = true;
      keepTryingNum_ = params_.kKeepTryingNum;  // Try 1 or 2 more times even if there
                                                // is no local frontier
      remainingFrontier_ = true;
      getThreeLocalFrontierPoint();
      pruneTree(root_);
      dual_state_graph_->clearLocalGraph();
      dual_state_graph_->local_graph_ = dual_state_graph_->pruned_graph_;
      dual_state_graph_->execute();
    }
    else
    {
      if (!localPlanOnceMore_)
      {
        dual_state_graph_->clearLocalGraph();
        dual_state_graph_->pruned_graph_.vertices.clear();
        // dual_state_graph_->collisionPoint_->clear();
        remainingFrontier_ = false;
        localPlanOnceMore_ = true;
        normal_local_iteration_ = true;
      }
      else
      {
        remainingFrontier_ = true;
        loopCount_ = params_.kLoopCountThres * 3;
        normal_local_iteration_ = false;
        localThreeFrontier_->clear();
        //        pruneTree(root_);
        dual_state_graph_->clearLocalGraph();
        dual_state_graph_->pruned_graph_.vertices.clear();
        // dual_state_graph_->collisionPoint_->clear();
        //        dual_state_graph_->local_graph_ =
        //        dual_state_graph_->pruned_graph_;
        dual_state_graph_->execute();
        keepTryingNum_--;
        if (keepTryingNum_ <= 0)
        {
          localPlanOnceMore_ = false;
          keepTryingNum_ = params_.kKeepTryingNum + 1;  // After switching to relocation stage, give
          // another more chance in case that some frontiers
          // are not updated
        }
      }
    }

    maxX_ = rootNode_->state_.x() + params_.kMaxXLocalBound;
    maxY_ = rootNode_->state_.y() + params_.kMaxYLocalBound;
    maxZ_ = rootNode_->state_.z() + params_.kMaxZLocalBound;
    minX_ = rootNode_->state_.x() + params_.kMinXLocalBound;
    minY_ = rootNode_->state_.y() + params_.kMinYLocalBound;
    minZ_ = rootNode_->state_.z() + params_.kMinZLocalBound;

    scanPolar.clear();
    scanPolar.resize(360, 0);
    pcl::PointXYZ point;
    if(scanCloud_->points.size() == 0) {
      for(int i = 0; i < 360; i++) {
          scanPolar[i] = 12;
      }
    }
    else {
      for(int i = 0; i < scanCloud_->points.size(); i++) {
        point = scanCloud_->points[i];
        double angle = (atan2(point.y, point.x) + M_PI) * 180 / M_PI;
        int ind = int(angle - 0.5);
        ind = (ind + 180) % 360;
        double dis = sqrt(pow(point.x, 2.0) + pow(point.y, 2.0)) + params_.kMinextensionRange;
        // dis = dis > 15 ? 15 : dis; // garage
        dis = dis > 12 ? 12 : dis; // indoor
        scanPolar[ind] = scanPolar[ind] < dis ?  dis : scanPolar[ind];
      }
    }
    
    for(int i = 0; i < scanPolar.size(); i++) {
      int left =  i-1 <        0         ? scanPolar.size() : i-1;
      int right = i+1 == scanPolar.size()?       0          : i+1;
      // if(abs(scanPolar[left] - scanPolar[i]) > 1 && abs(scanPolar[right] - scanPolar[i]) > 1 
      // && abs(scanPolar[right] - scanPolar[left]) < 1)
      if(abs(scanPolar[left] - scanPolar[i]) > abs(scanPolar[right] - scanPolar[left]) && 
         abs(scanPolar[right]- scanPolar[i]) > abs(scanPolar[right] - scanPolar[left]))
        scanPolar[i] = (scanPolar[left] + scanPolar[right]) / 2;
      if(scanPolar[i] > 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_)))
         scanPolar[i] = 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_));
    }

    #ifdef outfile
    std::ofstream rangeFile;
    rangeFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/rangeFile.txt");
    std::ofstream scanPolarFile;
    scanPolarFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/scanPolarFile.txt");
    #endif
    scanPolarPoint_->clear();
    for(int i = 0; i < scanPolar.size(); i++) {
      tf::Vector3 initPoint(scanPolar[i] * cos(i * M_PI / 180),
                            scanPolar[i] * sin(i * M_PI / 180), root_[2]);
      tf::Vector3 rotPoint = transformToMap * initPoint;
      #ifdef outfile
      rangeFile << rotPoint[0] << " " << rotPoint[1] << " " << rotPoint[2] << std::endl;
      scanPolarFile << scanPolar[i] << std::endl;
      #endif
      pcl::PointXYZ tempPoint;
      tempPoint.x = rotPoint.getX();
      tempPoint.y = rotPoint.getY();
      tempPoint.z = rotPoint.getZ();
      // tempPoint.intensity = 0;
      scanPolarPoint_->points.push_back(tempPoint);
    }
    tf::Vector3 initPoint1(scanPolar[0] * cos(0), scanPolar[0] * sin(0), root_[2]);
    tf::Vector3 rotPoint1 = transformToMap * initPoint1;
    #ifdef outfile
    rangeFile << rotPoint1[0] << " " << rotPoint1[1] << " " << rotPoint1[2] << std::endl;
    scanPolarFile << scanPolar[0] << std::endl;
    rangeFile.close();
    scanPolarFile.close();
    #endif

//  Polygon
    polygons.clear();
    diffPolar.clear();
    diffPolar.resize(360, 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_)));
    #ifdef outfile
    std::ofstream diffFile;
    diffFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/diffFile.txt");
    std::ofstream diffPolarFile;
    diffPolarFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/diffPolarFile.txt");
    #endif
    for(int i = 0; i < preScanPolar.size(); i++) {
      tf::Vector3 initPoint(preScanPolar[i] * cos(i * M_PI / 180),
                            preScanPolar[i] * sin(i * M_PI / 180), preTransformToMap.getOrigin().getZ());
      tf::Vector3 rotPoint = transformToMap.inverse() * (preTransformToMap * initPoint);
      polygons.push_back(rotPoint);
    }
    if(polygons.size() == 0 || dual_state_graph_->local_graph_.vertices.size() == 0) {
      diffPolar.clear();
      diffPolar.resize(360, 0);
      polygons.clear();
    }
    for(int k = 0; k < 360; k++) {
      // line segment formula
      tf::Vector3 point1(0.0, 0.0, root_[2]);
      tf::Vector3 point2(100.0 * cos(k * M_PI / 180), 100.0 * sin(k * M_PI / 180), root_[2]);
      tf::Vector3 intersectPoint(0.0, 0.0, root_[2]);

      for(int i = 0, j = polygons.size()-1; i < polygons.size(); j=i++) {
            //快速排斥实验
            if ((polygons[i][0] > polygons[j][0] ? polygons[i][0] : polygons[j][0]) <  (point2[0] < point1[0] ? point2[0] : point1[0]) ||
                (polygons[i][1] > polygons[j][1] ? polygons[i][1] : polygons[j][1]) <  (point2[1] < point1[1] ? point2[1] : point1[1]) ||
                (point2[0] > point1[0] ? point2[0] : point1[0]) < (polygons[i][0] < polygons[j][0] ?  polygons[i][0] : polygons[j][0]) ||
                (point2[1] > point1[1] ? point2[1] : point1[1]) < (polygons[i][1] < polygons[j][1] ?  polygons[i][1] : polygons[j][1])) {
                continue;
            }
            //跨立实验
            if ((((polygons[i][0] - point2[0])*(point1[1] - point2[1]) - (polygons[i][1] - point2[1])*(point1[0] - point2[0])) * 
                 ((polygons[j][0] - point2[0])*(point1[1] - point2[1]) - (polygons[j][1] - point2[1])*(point1[0] - point2[0]))) > 0 || 
                (((point2[0] - polygons[i][0])*(polygons[j][1] - polygons[i][1]) - (point2[1] - polygons[i][1]) * (polygons[j][0] - polygons[i][0])) * 
                ((point1[0] - polygons[i][0]) * (polygons[j][1] - polygons[i][1]) - (point1[1] - polygons[i][1]) * (polygons[j][0] - polygons[i][0]))) > 0) {
                continue;
            }
            // calculate intersection
            double temLeft =  (polygons[i][0] - polygons[j][0]) * (point1[1] - point2[1]) - (point2[0] - point1[0]) * (polygons[j][1] - polygons[i][1]);
            double temRight = (point1[1] - polygons[j][1]) * (point2[0] - point1[0]) * (polygons[i][0] - polygons[j][0]) + 
                                polygons[j][0] * (polygons[i][1] - polygons[j][1]) * (point2[0] - point1[0]) - 
                                point1[0] * (point2[1] - point1[1]) * (polygons[i][0] - polygons[j][0]);
            intersectPoint[0] = temRight / temLeft;

            temLeft = (point1[0] - point2[0]) * (polygons[i][1] - polygons[j][1]) - (point2[1] - point1[1]) * (polygons[j][0] - polygons[i][0]);
            temRight = point2[1] * (point1[0] - point2[0]) * (polygons[i][1] - polygons[j][1]) + 
                      (polygons[i][0] - point2[0]) * (polygons[i][1] - polygons[j][1]) * (point1[1] - point2[1]) - 
                      polygons[i][1] * (polygons[j][0] - polygons[i][0]) * (point2[1] - point1[1]);
            intersectPoint[1] = temRight / temLeft;
            diffPolar[k] = diffPolar[k] > sqrt(SQ(intersectPoint[0]) + SQ(intersectPoint[1])) ? sqrt(SQ(intersectPoint[0]) + SQ(intersectPoint[1])) : diffPolar[k];
      }
      #ifdef outfile
      diffPolarFile << diffPolar[k] << std::endl;
      #endif
      tf::Vector3 initPoint(diffPolar[k] * cos(k * M_PI / 180),
                            diffPolar[k] * sin(k * M_PI / 180), root_[2]);
      tf::Vector3 rotPoint = transformToMap * initPoint;
      #ifdef outfile
      diffFile << rotPoint[0] << " " << rotPoint[1] << " " << rotPoint[2] << std::endl;
      #endif
    }
    #ifdef outfile
    diffPolarFile << diffPolar[0] << std::endl;
    #endif
    tf::Vector3 initPoint(diffPolar[0] * cos(0),
                          diffPolar[0] * sin(0), root_[2]);
    tf::Vector3 rotPoint = transformToMap * initPoint;
    #ifdef outfile
    diffFile << rotPoint[0] << " " << rotPoint[1] << " " << rotPoint[2] << std::endl;
    diffFile.close();
    diffPolarFile.close();
    #endif

    #ifdef outfile
    std::ofstream samplePolarFile;
    samplePolarFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/samplePolarFile.txt");
    #endif
    samplePolar.clear();
    samplePolar.resize(360, 0);
    for(int i = 0; i < scanPolar.size(); i++) {
      if(scanPolar[i] - diffPolar[i] <= 0) {
        samplePolar[i] = 0;
      }
      else if(scanPolar[i] - diffPolar[i]  > 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_))) {
        samplePolar[i] = 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_));
      }
      else {
        samplePolar[i] = scanPolar[i] - diffPolar[i];
      }
      #ifdef outfile
      samplePolarFile << samplePolar[i] << std::endl;
      #endif
    }
    #ifdef outfile
    samplePolarFile << samplePolar[0] << std::endl;
    samplePolarFile.close();
    #endif



    preTransformToMap = transformToMap;
    preScanPolar = scanPolar;
  }
  else
  {
    std::cout << "Relocation Stage" << std::endl;
    localPlanOnceMore_ = true;
    StateVec node1;
    double gain1;
    if (dual_state_graph_->global_graph_.vertices.size() > 0)
    {
      node1[0] = dual_state_graph_->global_graph_.vertices[0].location.x;
      node1[1] = dual_state_graph_->global_graph_.vertices[0].location.y;
      node1[2] = dual_state_graph_->global_graph_.vertices[0].location.z;
      rootNode_->state_ = node1;
      kd_insert3(kdTree_, rootNode_->state_.x(), rootNode_->state_.y(), rootNode_->state_.z(), rootNode_);

      dual_state_graph_->clearLocalGraph();
      dual_state_graph_->collisionPoint_->clear();

      dual_state_graph_->local_graph_ = dual_state_graph_->global_graph_;
      dual_state_graph_->local_graph_.vertices[0].information_gain = rootNode_->gain_;
      dual_state_graph_->execute();
      getNextNodeToClosestGlobalFrontier();
      if (nextNodeFound_)
      {
        dual_state_graph_->local_graph_.vertices[NextBestNodeIdx_].information_gain =
            300000;  // set a large enough value as the best gain
        bestGain_ = 300000;
        nodeCounter_ = dual_state_graph_->global_graph_.vertices.size();
        global_vertex_size_ = nodeCounter_;
        dual_state_graph_->publishGlobalGraph();
      }
      else
      {  // Rebuild the rrt accordingt to current graph and then extend
         // in
         // plannerIterate. This only happens when no
         // global frontiers can be seen. Mostly used at the end of the
         // exploration in case that there are some narrow
         // areas are ignored.
        for (int i = 1; i < dual_state_graph_->global_graph_.vertices.size(); i++)
        {
          p1.position = dual_state_graph_->global_graph_.vertices[i].location;
          node1[0] = p1.position.x;
          node1[1] = p1.position.y;
          node1[2] = p1.position.z;

          kdres* nearest = kd_nearest3(kdTree_, node1.x(), node1.y(), node1.z());
          if (kd_res_size(nearest) <= 0)
          {
            kd_res_free(nearest);
            continue;
          }
          dsvplanner_ns::Node* newParent = (dsvplanner_ns::Node*)kd_res_item_data(nearest);
          kd_res_free(nearest);

          StateVec origin(newParent->state_[0], newParent->state_[1], newParent->state_[2]);
          StateVec direction(node1[0] - origin[0], node1[1] - origin[1], node1[2] - origin[2]);
          if (direction.norm() > params_.kExtensionRange)
          {
            direction = params_.kExtensionRange * direction.normalized();
          }
          node1[0] = origin[0] + direction[0];
          node1[1] = origin[1] + direction[1];
          node1[2] = origin[2] + direction[2];
          global_vertex_size_++;
          // Create new node and insert into tree
          dsvplanner_ns::Node* newNode = new dsvplanner_ns::Node;
          newNode->state_ = node1;
          newNode->parent_ = newParent;
          newNode->distance_ = newParent->distance_ + direction.norm();
          newParent->children_.push_back(newNode);
          newNode->gain_ = gain(newNode->state_);

          kd_insert3(kdTree_, node1.x(), node1.y(), node1.z(), newNode);

          // save new node to node_array
          dual_state_graph_->local_graph_.vertices[i].information_gain = newNode->gain_;
          node_array.push_back(newNode);

          if (newNode->gain_ > bestGain_)
          {
            if (std::find(executedBestNodeList_.begin(), executedBestNodeList_.end(), i) != executedBestNodeList_.end())
            {
              bestGain_ = newNode->gain_;
              bestNodeId_ = i;
            }
          }
          // nodeCounter_++;
        }
        executedBestNodeList_.push_back(bestNodeId_);
        nodeCounter_ = dual_state_graph_->global_graph_.vertices.size();
      }
    }
    else
    {
      rootNode_->state_ = root_;
      kd_insert3(kdTree_, rootNode_->state_.x(), rootNode_->state_.y(), rootNode_->state_.z(), rootNode_);
      iterationCount_++;
    }
    maxX_ = params_.kMaxXGlobalBound;
    maxY_ = params_.kMaxYGlobalBound;
    maxZ_ = params_.kMaxZGlobalBound;
    minX_ = params_.kMinXGlobalBound;
    minY_ = params_.kMinYGlobalBound;
    minZ_ = params_.kMinZGlobalBound;
  }
  publishPlanningHorizon();
}

void dsvplanner_ns::Drrt::plannerInit_szz()
{
  // This function is to initialize the tree
  kdTree_ = kd_create(3);

  node_array.clear();
  rootNode_ = new Node;
  rootNode_->distance_ = 0.0;
  rootNode_->gain_ = params_.kZeroGain;
  rootNode_->parent_ = NULL;

  global_vertex_size_ = 0;
  geometry_msgs::Pose p1;

  if (global_plan_ == false)
  {
    std::cout << "Exploration Stage" << std::endl;
    rootNode_->state_ = root_;
    kd_insert3(kdTree_, rootNode_->state_.x(), rootNode_->state_.y(), rootNode_->state_.z(), rootNode_);
    iterationCount_++;

    loopCount_ = params_.kLoopCountThres;
    normal_local_iteration_ = true;
    keepTryingNum_ = params_.kKeepTryingNum;  // Try 1 or 2 more times even if there
                                                // is no local frontier
    remainingFrontier_ = true;
    pruneTree(root_);
    dual_state_graph_->clearLocalGraph();
    dual_state_graph_->local_graph_ = dual_state_graph_->pruned_graph_;
    dual_state_graph_->execute();
    

    maxX_ = rootNode_->state_.x() + params_.kMaxXLocalBound;
    maxY_ = rootNode_->state_.y() + params_.kMaxYLocalBound;
    maxZ_ = rootNode_->state_.z() + params_.kMaxZLocalBound;
    minX_ = rootNode_->state_.x() + params_.kMinXLocalBound;
    minY_ = rootNode_->state_.y() + params_.kMinYLocalBound;
    minZ_ = rootNode_->state_.z() + params_.kMinZLocalBound;

    scanPolar.clear();
    scanPolar.resize(360, 0);
    pcl::PointXYZ point;
    if(scanCloud_->points.size() == 0) {
      for(int i = 0; i < 360; i++) {
          scanPolar[i] = 15;
      }
    }
    else {
      for(int i = 0; i < scanCloud_->points.size(); i++) {
        point = scanCloud_->points[i];
        double angle = (atan2(point.y, point.x) + M_PI) * 180 / M_PI;
        int ind = int(angle - 0.5);
        ind = (ind + 180) % 360;
        double dis = sqrt(pow(point.x, 2.0) + pow(point.y, 2.0)); //  + params_.kExtensionRange;
        // dis = dis > 15 ? 15 : dis; // forest
        // dis = dis > 12 ? 12 : dis; // indoor
        // dis = dis < 1 ? 15 : dis;  // virtual scan range = 0 if no obstacles
        dis = dis > 15 ? 15 : dis; // forest
        scanPolar[ind] = scanPolar[ind] < dis ?  dis : scanPolar[ind];
      }
    }
    for(int i = 0; i < scanPolar.size(); i++) {
      int left =  i-1 <        0         ? scanPolar.size() : i-1;
      int right = i+1 == scanPolar.size()?       0          : i+1;
      // if(abs(scanPolar[left] - scanPolar[i]) > 1 && abs(scanPolar[right] - scanPolar[i]) > 1 
      // && abs(scanPolar[right] - scanPolar[left]) < 1)
      if(abs(scanPolar[left] - scanPolar[i]) > abs(scanPolar[right] - scanPolar[left]) && 
         abs(scanPolar[right]- scanPolar[i]) > abs(scanPolar[right] - scanPolar[left]))
        scanPolar[i] = (scanPolar[left] + scanPolar[right]) / 2;
      if(scanPolar[i] > 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_)))
         scanPolar[i] = 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_));
    }

    #ifdef outfile
    std::ofstream rangeFile;
    rangeFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/rangeFile.txt");
    std::ofstream scanPolarFile;
    scanPolarFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/scanPolarFile.txt");
    #endif
    scanPolarPoint_->clear();
    for(int i = 0; i < scanPolar.size(); i++) {
      tf::Vector3 initPoint(scanPolar[i] * cos(i * M_PI / 180),
                            scanPolar[i] * sin(i * M_PI / 180), root_[2]);
      tf::Vector3 rotPoint = transformToMap * initPoint;
      #ifdef outfile
      rangeFile << rotPoint[0] << " " << rotPoint[1] << " " << rotPoint[2] << std::endl;
      scanPolarFile << scanPolar[i] << std::endl;
      #endif
      pcl::PointXYZ tempPoint;
      tempPoint.x = rotPoint.getX();
      tempPoint.y = rotPoint.getY();
      tempPoint.z = rotPoint.getZ();
      // tempPoint.intensity = 0;
      scanPolarPoint_->points.push_back(tempPoint);
    }
    tf::Vector3 initPoint1(scanPolar[0] * cos(0), scanPolar[0] * sin(0), root_[2]);
    tf::Vector3 rotPoint1 = transformToMap * initPoint1;
    #ifdef outfile
    rangeFile << rotPoint1[0] << " " << rotPoint1[1] << " " << rotPoint1[2] << std::endl;
    scanPolarFile << scanPolar[0] << std::endl;
    rangeFile.close();
    scanPolarFile.close();
    #endif

//  Polygon
    polygons.clear();
    diffPolar.clear();
    diffPolar.resize(360, 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_)));
    #ifdef outfile
    std::ofstream diffFile;
    diffFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/diffFile.txt");
    std::ofstream diffPolarFile;
    diffPolarFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/diffPolarFile.txt");
    #endif
    std::vector<std::vector<tf::Vector3>> multiPolygons;
    for (const auto &poly : pclPolygons) {
      std::vector<tf::Vector3> localPolygons;
      for(const auto & p : poly.vertices) {
        auto pvalue = concavePoint_->at(p);
        tf::Vector3 initPoint(pvalue.x, pvalue.y, transformToMap.getOrigin().getZ());
        tf::Vector3 rotPoint = transformToMap.inverse() * initPoint;
        localPolygons.push_back(rotPoint);
      }
      multiPolygons.push_back(localPolygons);
    }
    if(multiPolygons.size() == 0 || dual_state_graph_->local_graph_.vertices.size() == 0) {
      diffPolar.clear();
      diffPolar.resize(360, 0);
      polygons.clear();
    }
    for(int k = 0; k < 360; k++) {
      // line segment formula
      tf::Vector3 point1(0.0, 0.0, root_[2]);
      tf::Vector3 point2(100.0 * cos(k * M_PI / 180), 100.0 * sin(k * M_PI / 180), root_[2]);
      tf::Vector3 intersectPoint(0.0, 0.0, root_[2]);

      for (const auto &polygons : multiPolygons) {
        for(int i = 0, j = polygons.size()-1; i < polygons.size(); j=i++) {
              //快速排斥实验
              if ((polygons[i][0] > polygons[j][0] ? polygons[i][0] : polygons[j][0]) <  (point2[0] < point1[0] ? point2[0] : point1[0]) ||
                  (polygons[i][1] > polygons[j][1] ? polygons[i][1] : polygons[j][1]) <  (point2[1] < point1[1] ? point2[1] : point1[1]) ||
                  (point2[0] > point1[0] ? point2[0] : point1[0]) < (polygons[i][0] < polygons[j][0] ?  polygons[i][0] : polygons[j][0]) ||
                  (point2[1] > point1[1] ? point2[1] : point1[1]) < (polygons[i][1] < polygons[j][1] ?  polygons[i][1] : polygons[j][1])) {
                  continue;
              }
              //跨立实验
              if ((((polygons[i][0] - point2[0])*(point1[1] - point2[1]) - (polygons[i][1] - point2[1])*(point1[0] - point2[0])) * 
                  ((polygons[j][0] - point2[0])*(point1[1] - point2[1]) - (polygons[j][1] - point2[1])*(point1[0] - point2[0]))) > 0 || 
                  (((point2[0] - polygons[i][0])*(polygons[j][1] - polygons[i][1]) - (point2[1] - polygons[i][1]) * (polygons[j][0] - polygons[i][0])) * 
                  ((point1[0] - polygons[i][0]) * (polygons[j][1] - polygons[i][1]) - (point1[1] - polygons[i][1]) * (polygons[j][0] - polygons[i][0]))) > 0) {
                  continue;
              }
              // calculate intersection
              double temLeft =  (polygons[i][0] - polygons[j][0]) * (point1[1] - point2[1]) - (point2[0] - point1[0]) * (polygons[j][1] - polygons[i][1]);
              double temRight = (point1[1] - polygons[j][1]) * (point2[0] - point1[0]) * (polygons[i][0] - polygons[j][0]) + 
                                  polygons[j][0] * (polygons[i][1] - polygons[j][1]) * (point2[0] - point1[0]) - 
                                  point1[0] * (point2[1] - point1[1]) * (polygons[i][0] - polygons[j][0]);
              intersectPoint[0] = temRight / temLeft;

              temLeft = (point1[0] - point2[0]) * (polygons[i][1] - polygons[j][1]) - (point2[1] - point1[1]) * (polygons[j][0] - polygons[i][0]);
              temRight = point2[1] * (point1[0] - point2[0]) * (polygons[i][1] - polygons[j][1]) + 
                        (polygons[i][0] - point2[0]) * (polygons[i][1] - polygons[j][1]) * (point1[1] - point2[1]) - 
                        polygons[i][1] * (polygons[j][0] - polygons[i][0]) * (point2[1] - point1[1]);
              intersectPoint[1] = temRight / temLeft;
              diffPolar[k] = diffPolar[k] > sqrt(SQ(intersectPoint[0]) + SQ(intersectPoint[1])) ? sqrt(SQ(intersectPoint[0]) + SQ(intersectPoint[1])) : diffPolar[k];
        }
      }
      #ifdef outfile
      diffPolarFile << diffPolar[k] << std::endl;
      #endif
      tf::Vector3 initPoint(diffPolar[k] * cos(k * M_PI / 180),
                            diffPolar[k] * sin(k * M_PI / 180), root_[2]);
      tf::Vector3 rotPoint = transformToMap * initPoint;
      #ifdef outfile
      diffFile << rotPoint[0] << " " << rotPoint[1] << " " << rotPoint[2] << std::endl;
      #endif
    }
    #ifdef outfile
    diffPolarFile << diffPolar[0] << std::endl;
    #endif
    tf::Vector3 initPoint(diffPolar[0] * cos(0),
                          diffPolar[0] * sin(0), root_[2]);
    tf::Vector3 rotPoint = transformToMap * initPoint;
    #ifdef outfile
    diffFile << rotPoint[0] << " " << rotPoint[1] << " " << rotPoint[2] << std::endl;
    diffFile.close();
    diffPolarFile.close();
    #endif

    #ifdef outfile
    std::ofstream samplePolarFile;
    samplePolarFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/samplePolarFile.txt");
    #endif
    samplePolar.clear();
    samplePolar.resize(360, 0);
    // probIntegration.clear();
    // probIntegration.resize(360, 0);
    std::vector<double> probVec;
    probVec.resize(360,0);
    double sum = 0;
    for(int i = 0; i < scanPolar.size(); i++) {
      if(scanPolar[i] - diffPolar[i] <= 0) {
        samplePolar[i] = 0;
      }
      else if(scanPolar[i] - diffPolar[i]  > 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_))) {
        samplePolar[i] = 0.5 * sqrt(SQ(minX_ - maxX_) + SQ(minY_ - maxY_));
      }
      else {
        samplePolar[i] = scanPolar[i] - diffPolar[i];
      }
      #ifdef outfile
      samplePolarFile << samplePolar[i] << std::endl;
      #endif
      probVec[i] = samplePolar[i];
      sum += probVec[i];
    }
    for (int i = 0; i < probVec.size(); i++) {
      probVec[i] /= sum;
    }
    discreteProb = std::discrete_distribution<>(probVec.begin(), probVec.end());
    #ifdef outfile
    samplePolarFile << samplePolar[0] << std::endl;
    samplePolarFile.close();
    #endif

    preTransformToMap = transformToMap;
    preScanPolar = scanPolar;
  }
  else
  {
    std::cout << "Relocation Stage" << std::endl;
    remainingFrontier_ = true;
    // localPlanOnceMore_ = true;
    StateVec node1;
    double gain1;
    if (dual_state_graph_->global_graph_.vertices.size() > 0)
    {
      node1[0] = dual_state_graph_->global_graph_.vertices[0].location.x;
      node1[1] = dual_state_graph_->global_graph_.vertices[0].location.y;
      node1[2] = dual_state_graph_->global_graph_.vertices[0].location.z;
      rootNode_->state_ = node1;
      kd_insert3(kdTree_, rootNode_->state_.x(), rootNode_->state_.y(), rootNode_->state_.z(), rootNode_);

      dual_state_graph_->clearLocalGraph();
      dual_state_graph_->collisionPoint_->clear();

      dual_state_graph_->local_graph_ = dual_state_graph_->global_graph_;
      dual_state_graph_->local_graph_.vertices[0].information_gain = rootNode_->gain_;
      dual_state_graph_->execute();
      getNextNodeToClosestGlobalFrontier();
      if (nextNodeFound_)
      {
        dual_state_graph_->local_graph_.vertices[NextBestNodeIdx_].information_gain =
            300000;  // set a large enough value as the best gain
        bestGain_ = 300000;
        nodeCounter_ = dual_state_graph_->global_graph_.vertices.size();
        global_vertex_size_ = nodeCounter_;
        dual_state_graph_->publishGlobalGraph();
      }
      else
      {  // Rebuild the rrt accordingt to current graph and then extend
         // in
         // plannerIterate. This only happens when no
         // global frontiers can be seen. Mostly used at the end of the
         // exploration in case that there are some narrow
         // areas are ignored.
        for (int i = 1; i < dual_state_graph_->global_graph_.vertices.size(); i++)
        {
          p1.position = dual_state_graph_->global_graph_.vertices[i].location;
          node1[0] = p1.position.x;
          node1[1] = p1.position.y;
          node1[2] = p1.position.z;

          kdres* nearest = kd_nearest3(kdTree_, node1.x(), node1.y(), node1.z());
          if (kd_res_size(nearest) <= 0)
          {
            kd_res_free(nearest);
            continue;
          }
          dsvplanner_ns::Node* newParent = (dsvplanner_ns::Node*)kd_res_item_data(nearest);
          kd_res_free(nearest);

          StateVec origin(newParent->state_[0], newParent->state_[1], newParent->state_[2]);
          StateVec direction(node1[0] - origin[0], node1[1] - origin[1], node1[2] - origin[2]);
          if (direction.norm() > params_.kExtensionRange)
          {
            direction = params_.kExtensionRange * direction.normalized();
          }
          node1[0] = origin[0] + direction[0];
          node1[1] = origin[1] + direction[1];
          node1[2] = origin[2] + direction[2];
          global_vertex_size_++;
          // Create new node and insert into tree
          dsvplanner_ns::Node* newNode = new dsvplanner_ns::Node;
          newNode->state_ = node1;
          newNode->parent_ = newParent;
          newNode->distance_ = newParent->distance_ + direction.norm();
          newParent->children_.push_back(newNode);
          newNode->gain_ = gain(newNode->state_);

          kd_insert3(kdTree_, node1.x(), node1.y(), node1.z(), newNode);

          // save new node to node_array
          dual_state_graph_->local_graph_.vertices[i].information_gain = newNode->gain_;
          node_array.push_back(newNode);

          if (newNode->gain_ > bestGain_)
          {
            if (std::find(executedBestNodeList_.begin(), executedBestNodeList_.end(), i) != executedBestNodeList_.end())
            {
              bestGain_ = newNode->gain_;
              bestNodeId_ = i;
            }
          }
          // nodeCounter_++;
        }
        executedBestNodeList_.push_back(bestNodeId_);
        nodeCounter_ = dual_state_graph_->global_graph_.vertices.size();
      }
    }
    else
    {
      rootNode_->state_ = root_;
      kd_insert3(kdTree_, rootNode_->state_.x(), rootNode_->state_.y(), rootNode_->state_.z(), rootNode_);
      iterationCount_++;
    }
    maxX_ = params_.kMaxXGlobalBound;
    maxY_ = params_.kMaxYGlobalBound;
    maxZ_ = params_.kMaxZGlobalBound;
    minX_ = params_.kMinXGlobalBound;
    minY_ = params_.kMinYGlobalBound;
    minZ_ = params_.kMinZGlobalBound;
  }
  publishPlanningHorizon();
}

void dsvplanner_ns::Drrt::publishPlanningHorizon()
{  // Publish visualization of
   // current planning horizon
  visualization_msgs::Marker p;
  p.header.stamp = ros::Time::now();
  p.header.frame_id = params_.explorationFrame;
  p.id = 0;
  p.ns = "boundary";
  p.type = visualization_msgs::Marker::CUBE;
  p.action = visualization_msgs::Marker::ADD;
  p.pose.position.x = 0.5 * (minX_ + maxX_);
  p.pose.position.y = 0.5 * (minY_ + maxY_);
  p.pose.position.z = 0.5 * (minZ_ + maxZ_);
  tf::Quaternion quat;
  quat.setEuler(0.0, 0.0, 0.0);
  p.pose.orientation.x = quat.x();
  p.pose.orientation.y = quat.y();
  p.pose.orientation.z = quat.z();
  p.pose.orientation.w = quat.w();
  p.scale.x = maxX_ - minX_;
  p.scale.y = maxY_ - minY_;
  p.scale.z = maxZ_ - minZ_;
  p.color.r = 252.0 / 255.0;
  p.color.g = 145.0 / 255.0;
  p.color.b = 37.0 / 255.0;
  p.color.a = 0.3;
  p.lifetime = ros::Duration(0.0);
  p.frame_locked = false;
  params_.boundaryPub_.publish(p);
}

void dsvplanner_ns::Drrt::pruneTree(StateVec root)
{
  dual_state_graph_->pruned_graph_.vertices.clear();
  geometry_msgs::Pose p1;
  p1.position.x = root[0];
  p1.position.y = root[1];
  p1.position.z = root[2];
  p1.orientation.y = params_.kZeroGain;
  dual_state_graph_->addNewLocalVertexWithoutDuplicates(p1, dual_state_graph_->pruned_graph_);

  geometry_msgs::Point root_point;
  root_point.x = root[0];
  root_point.y = root[1];
  root_point.z = root[2];
  dual_state_graph_->pruneGraph(root_point);
  dual_state_graph_->pruneGraph_szz(root_point);
  getConcaveHull();
  StateVec node;
  for (int i = 1; i < dual_state_graph_->pruned_graph_.vertices.size(); i++)
  {
    node[0] = dual_state_graph_->pruned_graph_.vertices[i].location.x;
    node[1] = dual_state_graph_->pruned_graph_.vertices[i].location.y;
    node[2] = dual_state_graph_->pruned_graph_.vertices[i].location.z;

    kdres* nearest = kd_nearest3(kdTree_, node.x(), node.y(), node.z());
    if (kd_res_size(nearest) <= 0)
    {
      kd_res_free(nearest);
      continue;
    }
    dsvplanner_ns::Node* newParent = (dsvplanner_ns::Node*)kd_res_item_data(nearest);
    kd_res_free(nearest);

    // Check for collision
    StateVec origin(newParent->state_[0], newParent->state_[1], newParent->state_[2]);
    StateVec direction(node[0] - origin[0], node[1] - origin[1], node[2] - origin[2]);

    dsvplanner_ns::Node* newNode = new dsvplanner_ns::Node;
    newNode->state_ = node;
    newNode->parent_ = newParent;
    newNode->distance_ = newParent->distance_ + direction.norm();
    newParent->children_.push_back(newNode);
    if (dual_state_graph_->pruned_graph_.vertices[i].information_gain > 0)
      // newNode->gain_ = gain(newNode->state_);
      newNode->gain_ = graphGain_concave(newNode->state_);
    else
    {
      newNode->gain_ = 0;
    }
    kd_insert3(kdTree_, node.x(), node.y(), node.z(), newNode);
    node_array.push_back(newNode);

    if (newNode->gain_ > bestGain_)
    {
      bestGain_ = newNode->gain_;
    }
    dual_state_graph_->pruned_graph_.vertices[i].information_gain = newNode->gain_;
  }
  remainingNodeCount_ = node_array.size();
}

double dsvplanner_ns::Drrt::graphGain_concave(StateVec state) {
  double gain = 0.0;
  // const double disc = manager_->getResolution();
  const double disc = chunk_map_->resolution();
  StateVec origin(state[0], state[1], state[2]);
  StateVec vec;
  double rangeSq = pow(params_.kGainRange, 2.0);

  // Iterate over all nodes within the allowed distance
  for (vec[0] = std::max(state[0] - params_.kGainRange, minX_); vec[0] < std::min(state[0] + params_.kGainRange, maxX_);
       vec[0] += disc)
  {
    for (vec[1] = std::max(state[1] - params_.kGainRange, minY_);
         vec[1] < std::min(state[1] + params_.kGainRange, maxY_); vec[1] += disc)
    {
      // Check cell whether in Concave Hull
      int c = 0;
      for(int k = 0; k < pclPolygons.size(); k++) {
        for(int i = 0, j = pclPolygons[k].vertices.size()-1; i < pclPolygons[k].vertices.size(); j=i++) {
          if(concavePoint_->points[pclPolygons[k].vertices[i]].y < vec[1] && concavePoint_->points[pclPolygons[k].vertices[j]].y >= vec[1] ||
              concavePoint_->points[pclPolygons[k].vertices[j]].y < vec[1] && concavePoint_->points[pclPolygons[k].vertices[i]].y >= vec[1]) {
                if(concavePoint_->points[pclPolygons[k].vertices[i]].x + (vec[1]-concavePoint_->points[pclPolygons[k].vertices[i]].y)/
                  (concavePoint_->points[pclPolygons[k].vertices[j]].y-concavePoint_->points[pclPolygons[k].vertices[i]].y)*
                  (concavePoint_->points[pclPolygons[k].vertices[j]].x-concavePoint_->points[pclPolygons[k].vertices[i]].x) < vec[0]) {
                  c = !c;
                  }
              }
        }
      }
      bool success = false;
      if(c == 0){
        for(int k = 0; k < pclPolygons.size(); k++) {
          for(int i = 0, j = pclPolygons[k].vertices.size()-1; i < pclPolygons[k].vertices.size(); j=i++) {
            //快速排斥实验
            if ((concavePoint_->points[pclPolygons[k].vertices[i]].x > concavePoint_->points[pclPolygons[k].vertices[j]].x ? 
                  concavePoint_->points[pclPolygons[k].vertices[i]].x : concavePoint_->points[pclPolygons[k].vertices[j]].x) < 
                (vec[0] < origin[0] ? vec[0] : origin[0]) ||
                (concavePoint_->points[pclPolygons[k].vertices[i]].y > concavePoint_->points[pclPolygons[k].vertices[j]].y ? 
                  concavePoint_->points[pclPolygons[k].vertices[i]].y : concavePoint_->points[pclPolygons[k].vertices[j]].y) < 
                (vec[1] < origin[1] ? vec[1] : origin[1]) ||
                (vec[0] > origin[0] ? vec[0] : origin[0]) < 
                (concavePoint_->points[pclPolygons[k].vertices[i]].x < concavePoint_->points[pclPolygons[k].vertices[j]].x ? 
                  concavePoint_->points[pclPolygons[k].vertices[i]].x : concavePoint_->points[pclPolygons[k].vertices[j]].x) ||
                (vec[1] > origin[1] ? vec[1] : origin[1]) < 
                (concavePoint_->points[pclPolygons[k].vertices[i]].y < concavePoint_->points[pclPolygons[k].vertices[j]].y ? 
                  concavePoint_->points[pclPolygons[k].vertices[i]].y : concavePoint_->points[pclPolygons[k].vertices[j]].y)) {
                continue;
            }
            //跨立实验
            if ((((concavePoint_->points[pclPolygons[k].vertices[i]].x - vec[0])*(origin[1] - vec[1]) - 
                  (concavePoint_->points[pclPolygons[k].vertices[i]].y - vec[1])*(origin[0] - vec[0])) * 
                  ((concavePoint_->points[pclPolygons[k].vertices[j]].x - vec[0])*(origin[1] - vec[1]) - 
                  (concavePoint_->points[pclPolygons[k].vertices[j]].y - vec[1])*(origin[0] - vec[0]))) > 0 || 
                (((vec[0] - concavePoint_->points[pclPolygons[k].vertices[i]].x)*(concavePoint_->points[pclPolygons[k].vertices[j]].y - 
                    concavePoint_->points[pclPolygons[k].vertices[i]].y) - (vec[1] - concavePoint_->points[pclPolygons[k].vertices[i]].y) * 
                  (concavePoint_->points[pclPolygons[k].vertices[j]].x - concavePoint_->points[pclPolygons[k].vertices[i]].x)) * 
                  ((origin[0] - concavePoint_->points[pclPolygons[k].vertices[i]].x) * 
                  (concavePoint_->points[pclPolygons[k].vertices[j]].y - concavePoint_->points[pclPolygons[k].vertices[i]].y) - 
                  (origin[1] - concavePoint_->points[pclPolygons[k].vertices[i]].y) * 
                  (concavePoint_->points[pclPolygons[k].vertices[j]].x - concavePoint_->points[pclPolygons[k].vertices[i]].x))) > 0) {
                continue;
            }
            // TODO: check shortest edge
            if( !(concavePoint_->points[pclPolygons[k].vertices[i]].intensity > 0 && 
                  concavePoint_->points[pclPolygons[k].vertices[j]].intensity > 0)) {
                  // gain += params_.kGainUnknown;
                  success = true;
                  break;
                }
          }
        }
      }
      if (!success)
        continue;

      for (vec[2] = std::max(state[2] - params_.kGainRangeZMinus, minZ_);
           vec[2] < std::min(state[2] + params_.kGainRangeZPlus, maxZ_); vec[2] += disc)
      {
        StateVec dir = vec - origin;
        // Skip if distance is too large
        if (dir.transpose().dot(dir) > rangeSq)
        {
          continue;
        }
        bool insideAFieldOfView = false;
        // Check that voxel center is inside the field of view. This check is
        // for velodyne.
        if (fabs(dir[2] < sqrt(dir[0] * dir[0] + dir[1] * dir[1]) * tan(M_PI * params_.sensorVerticalView / 360)))
        {
          insideAFieldOfView = true;
        }
        if (!insideAFieldOfView)
        {
          continue;
        }

        gain += params_.kGainUnknown;
      }
    }
  }   

  // Scale with volume
  if (gain < params_.kMinEffectiveGain)
    gain = 0;
  gain *= pow(disc, 3.0);

  return gain;
}


double dsvplanner_ns::Drrt::gain(StateVec state)
{
  // This function computes the gain
  double gain = 0.0;
  // const double disc = manager_->getResolution();
  const double disc = chunk_map_->resolution();
  StateVec origin(state[0], state[1], state[2]);
  StateVec vec;
  double rangeSq = pow(params_.kGainRange, 2.0);

  // Iterate over all nodes within the allowed distance
  for (vec[0] = std::max(state[0] - params_.kGainRange, minX_); vec[0] < std::min(state[0] + params_.kGainRange, maxX_);
       vec[0] += disc)
  {
    for (vec[1] = std::max(state[1] - params_.kGainRange, minY_);
         vec[1] < std::min(state[1] + params_.kGainRange, maxY_); vec[1] += disc)
    {
      // for (vec[2] = std::max(state[2] - params_.kGainRangeZMinus, minZ_);
      //      vec[2] < std::min(state[2] + params_.kGainRangeZPlus, maxZ_); vec[2] += disc)
      // {
        StateVec dir = vec - origin;
        // Skip if distance is too large
        if (dir.transpose().dot(dir) > rangeSq)
        {
          continue;
        }
        bool insideAFieldOfView = false;
        // Check that voxel center is inside the field of view. This check is
        // for velodyne.
        if (fabs(dir[2] < sqrt(dir[0] * dir[0] + dir[1] * dir[1]) * tan(M_PI * params_.sensorVerticalView / 360)))
        {
          insideAFieldOfView = true;
        }
        if (!insideAFieldOfView)
        {
          continue;
        }

        // Check cell status and add to the gain considering the corresponding
        // factor.
        // double probability;
        // volumetric_mapping::OctomapManager::CellStatus node = manager_->getCellProbabilityPoint(vec, &probability);
        ChunkCellStatus node = ChunkmapGetCellStatusPoint(chunk_map_, vec);
        // if (node == volumetric_mapping::OctomapManager::CellStatus::kUnknown)
        if (node == ChunkCellStatus::UNKNOWN)
        {
          // if (volumetric_mapping::OctomapManager::CellStatus::kOccupied !=
          //     this->manager_->getVisibility(origin, vec, false))
          if (ChunkCellStatus::OCCUPIED != ChunkmapGetVisibility(chunk_map_, origin, vec, false))
          {
            gain += params_.kGainUnknown;
          }
        }
        // else if (node == volumetric_mapping::OctomapManager::CellStatus::kOccupied)
        else if (node == ChunkCellStatus::OCCUPIED)
        {
          // if (volumetric_mapping::OctomapManager::CellStatus::kOccupied !=
          //     this->manager_->getVisibility(origin, vec, false))
          if (ChunkCellStatus::OCCUPIED != ChunkmapGetVisibility(chunk_map_, origin, vec, false))
          {
            gain += params_.kGainOccupied;
          }
        }
        else
        {
          // if (volumetric_mapping::OctomapManager::CellStatus::kOccupied !=
          //     this->manager_->getVisibility(origin, vec, false))
          if (ChunkCellStatus::OCCUPIED != ChunkmapGetVisibility(chunk_map_, origin, vec, false))
          {
            gain += params_.kGainFree;
          }
        }
      // }
    }
  }

  // Scale with volume
  if (gain < params_.kMinEffectiveGain)
    gain = 0;
  gain *= pow(disc, 3.0);

  return gain;
}

void dsvplanner_ns::Drrt::clear()
{
  delete rootNode_;
  rootNode_ = NULL;

  nodeCounter_ = 0;
  bestGain_ = params_.kZeroGain;
  bestNode_ = NULL;
  if (nextNodeFound_)
  {
    dual_state_graph_->clearLocalGraph();
  }
  nextNodeFound_ = false;
  remainingFrontier_ = false;
  remainingNodeCount_ = 0;

  sampledPoint_->clear();

  kd_free(kdTree_);
}

void dsvplanner_ns::Drrt::publishTSPPath() {
  nav_msgs::Path tspPath;
  for(int i = 0; i < tspTour.size(); i++) {
    geometry_msgs::PoseStamped pose;
    pose.pose.position = tspTour[i];
    tspPath.poses.push_back(pose);
  }
  tspPath.header.frame_id = "map";
  tspPath.header.stamp = ros::Time::now();
  params_.tspPathPub_.publish(tspPath);
}

void dsvplanner_ns::Drrt::publishNode()
{
  // pub rrt graph
  sensor_msgs::PointCloud2 random_sampled_points_pc;
  // pcl::toROSMsg(*sampledPoint_, random_sampled_points_pc);
  // pcl::toROSMsg(*scanPolarPoint_, random_sampled_points_pc);
  pcl::toROSMsg(*concavePoint_, random_sampled_points_pc);
  // pcl::toROSMsg(*dual_state_graph_->collisionPoint_, random_sampled_points_pc);
  random_sampled_points_pc.header.frame_id = params_.explorationFrame;
  params_.randomSampledPointsPub_.publish(random_sampled_points_pc);

  visualization_msgs::Marker node;
  visualization_msgs::Marker branch;
  node.header.stamp = ros::Time::now();
  node.header.frame_id = params_.explorationFrame;
  node.ns = "drrt_node";
  node.type = visualization_msgs::Marker::POINTS;
  node.action = visualization_msgs::Marker::ADD;
  node.scale.x = params_.kRemainingNodeScaleSize;
  node.color.r = 167.0 / 255.0;
  node.color.g = 167.0 / 255.0;
  node.color.b = 0.0;
  node.color.a = 1.0;
  node.frame_locked = false;

  branch.ns = "drrt_branches";
  branch.header.stamp = ros::Time::now();
  branch.header.frame_id = params_.explorationFrame;
  branch.type = visualization_msgs::Marker::LINE_LIST;
  branch.action = visualization_msgs::Marker::ADD;
  branch.scale.x = params_.kRemainingBranchScaleSize;
  branch.color.r = 167.0 / 255.0;
  branch.color.g = 167.0 / 255.0;
  branch.color.b = 0.0;
  branch.color.a = 1.0;
  branch.frame_locked = false;

  geometry_msgs::Point node_position;
  geometry_msgs::Point parent_position;
  if (remainingNodeCount_ > 0 && remainingNodeCount_ <= node_array.size())
  {
    for (int i = 0; i < remainingNodeCount_; i++)
    {
      node_position.x = node_array[i]->state_[0];
      node_position.y = node_array[i]->state_[1];
      node_position.z = node_array[i]->state_[2];
      node.points.push_back(node_position);

      if (node_array[i]->parent_)
      {
        parent_position.x = node_array[i]->parent_->state_[0];
        parent_position.y = node_array[i]->parent_->state_[1];
        parent_position.z = node_array[i]->parent_->state_[2];

        branch.points.push_back(parent_position);
        branch.points.push_back(node_position);
      }
    }
    params_.remainingTreePathPub_.publish(node);
    params_.remainingTreePathPub_.publish(branch);
    node.points.clear();
    branch.points.clear();
    node.scale.x = params_.kNewNodeScaleSize;
    node.color.r = 167.0 / 255.0;
    node.color.g = 0.0 / 255.0;
    node.color.b = 167.0 / 255.0;
    node.color.a = 1.0;
    branch.scale.x = params_.kNewBranchScaleSize;
    branch.color.r = 167.0 / 255.0;
    branch.color.g = 0.0 / 255.0;
    branch.color.b = 167.0 / 255.0;
    branch.color.a = 1.0;
    for (int i = remainingNodeCount_; i < node_array.size(); i++)
    {
      node_position.x = node_array[i]->state_[0];
      node_position.y = node_array[i]->state_[1];
      node_position.z = node_array[i]->state_[2];
      node.points.push_back(node_position);

      if (node_array[i]->parent_)
      {
        parent_position.x = node_array[i]->parent_->state_[0];
        parent_position.y = node_array[i]->parent_->state_[1];
        parent_position.z = node_array[i]->parent_->state_[2];

        branch.points.push_back(parent_position);
        branch.points.push_back(node_position);
      }
    }
    params_.newTreePathPub_.publish(node);
    params_.newTreePathPub_.publish(branch);
  }
  else
  {
    for (int i = 0; i < node_array.size(); i++)
    {
      node_position.x = node_array[i]->state_[0];
      node_position.y = node_array[i]->state_[1];
      node_position.z = node_array[i]->state_[2];
      node.points.push_back(node_position);

      if (node_array[i]->parent_)
      {
        parent_position.x = node_array[i]->parent_->state_[0];
        parent_position.y = node_array[i]->parent_->state_[1];
        parent_position.z = node_array[i]->parent_->state_[2];

        branch.points.push_back(parent_position);
        branch.points.push_back(node_position);
      }
    }
    params_.newTreePathPub_.publish(node);
    params_.newTreePathPub_.publish(branch);

    // When there is no remaining node, publish an empty one
    node.points.clear();
    branch.points.clear();
    params_.remainingTreePathPub_.publish(node);
    params_.remainingTreePathPub_.publish(branch);
  }
}

void dsvplanner_ns::Drrt::gotoxy(int x, int y)
{
  printf("%c[%d;%df", 0x1B, y, x);
}

#endif
