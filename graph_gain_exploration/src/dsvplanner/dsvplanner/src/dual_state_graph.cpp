/**************************************************************************
dual_state_graph.cpp
Implementation of dual_state graph. Create local and global graph according
to the new node from dynamic rrt.

Hongbiao Zhu(hongbiaz@andrew.cmu.edu)
5/25/2020
**************************************************************************/

#include "dsvplanner/dual_state_graph.h"

#include <graph_utils.h>
#include <misc_utils/misc_utils.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <dsvplanner/chunkmap_helper.h>

// #define outfile

namespace dsvplanner_ns
{
bool DualStateGraph::readParameters()
{
  nh_private_.getParam("/graph/world_frame_id", world_frame_id_);
  nh_private_.getParam("/graph/pub_local_graph_topic", pub_local_graph_topic_);
  nh_private_.getParam("/graph/pub_global_graph_topic", pub_global_graph_topic_);
  nh_private_.getParam("/graph/pub_global_points_topic", pub_global_points_topic_);
  nh_private_.getParam("/graph/sub_keypose_topic", sub_keypose_topic_);
  nh_private_.getParam("/graph/sub_path_topic", sub_path_topic_);
  nh_private_.getParam("/graph/sub_graph_planner_status_topic", sub_graph_planner_status_topic_);
  nh_private_.getParam("/graph/kCropPathWithTerrain", kCropPathWithTerrain);
  nh_private_.getParam("/graph/kConnectVertexDistMax", kConnectVertexDistMax);
  nh_private_.getParam("/graph/kDistBadEdge", kDistBadEdge);
  nh_private_.getParam("/graph/kDegressiveCoeff", kDegressiveCoeff);
  nh_private_.getParam("/graph/kDirectionCoeff", kDirectionCoeff);
  nh_private_.getParam("/graph/kExistingPathRatioThresholdGlobal", kExistingPathRatioThresholdGlobal);
  nh_private_.getParam("/graph/kExistingPathRatioThreshold", kExistingPathRatioThreshold);
  nh_private_.getParam("/graph/kLongEdgePenaltyMultiplier", kLongEdgePenaltyMultiplier);
  nh_private_.getParam("/graph/kMaxLongEdgeDist", kMaxLongEdgeDist);
  nh_private_.getParam("/graph/kMaxVertexAngleAlongZ", kMaxVertexAngleAlongZ);
  nh_private_.getParam("/graph/kMaxVertexDiffAlongZ", kMaxVertexDiffAlongZ);
  nh_private_.getParam("/graph/kMaxDistToPrunedRoot", kMaxDistToPrunedRoot);
  nh_private_.getParam("/graph/kMaxPrunedNodeDist", kMaxPrunedNodeDist);
  nh_private_.getParam("/graph/kMinVertexDist", kMinVertexDist);
  nh_private_.getParam("/graph/kSurroundRange", kSurroundRange);
  nh_private_.getParam("/graph/kMinGainRange", kMinGainRange);
  nh_private_.getParam("/graph/kMinDistanceToRobotToCheck", kMinDistanceToRobotToCheck);
  nh_private_.getParam("/rm/kBoundX", robot_bounding[0]);
  nh_private_.getParam("/rm/kBoundY", robot_bounding[1]);
  nh_private_.getParam("/rm/kBoundZ", robot_bounding[2]);
  return true;
}

void DualStateGraph::publishLocalGraph()
{
  // Publish the current local graph
  local_graph_.header.stamp = ros::Time::now();
  local_graph_.header.frame_id = world_frame_id_;
  local_graph_pub_.publish(local_graph_);

  // graph_point is used to detect frontiers
  graph_point_cloud_->points.clear();
  for (int i = 0; i < local_graph_.vertices.size(); i++)
  {
    pcl::PointXYZ p1;
    p1.x = local_graph_.vertices[i].location.x;
    p1.y = local_graph_.vertices[i].location.y;
    p1.z = local_graph_.vertices[i].location.z;
    graph_point_cloud_->points.push_back(p1);
  }
  sensor_msgs::PointCloud2 graph_pc;
  pcl::toROSMsg(*graph_point_cloud_, graph_pc);
  graph_pc.header.frame_id = "map";
  graph_points_pub_.publish(graph_pc);
}

void DualStateGraph::publishGlobalGraph()
{
  // Publish the current global graph
  global_graph_.header.stamp = ros::Time::now();
  global_graph_.header.frame_id = world_frame_id_;
  for(auto& vertex : global_graph_.vertices) {
    for(auto& edge : vertex.edges) {
      int start_idx = edge.vertex_id_end;
      int end_idx = vertex.vertex_id;
      bool flag = false;
      for(auto& invedge : global_graph_.vertices[start_idx].edges) {
        if(invedge.vertex_id_end == end_idx) {
          flag = true;
          break;
        }
      }
      if(flag == false) {
        auto tempedge = edge;
        tempedge.vertex_id_end = end_idx;
        global_graph_.vertices[start_idx].edges.push_back(tempedge);
      }
    }
  }
  global_graph_pub_.publish(global_graph_);
}

void DualStateGraph::addNewLocalVertexWithoutDuplicates(geometry_msgs::Pose& vertex_msg,
                                                        graph_utils::TopologicalGraph& graph)
{
  // Same as addNewLocalVertex but only adds vertex if a similar vertex doesn't
  // already exist

  // Extract the point
  geometry_msgs::Point new_vertex_location;
  new_vertex_location = vertex_msg.position;

  // Check if a similar vertex already exists
  bool already_exists = false;
  bool too_long = false;
  double distance = -1;
  int closest_vertex_idx = -1;
  if (!graph.vertices.empty())
  {
    closest_vertex_idx = graph_utils_ns::GetClosestVertexIdxToPoint(graph, new_vertex_location);
    auto& closest_vertex_location = graph.vertices[closest_vertex_idx].location;
    distance = misc_utils_ns::PointXYZDist(new_vertex_location, closest_vertex_location);
    if (distance < kMinVertexDist)
    {
      already_exists = true;
    }
    if (distance > kMaxLongEdgeDist || fabs(new_vertex_location.z - closest_vertex_location.z) > kMaxVertexDiffAlongZ)
    {
      too_long = true;
    }
  }

  // If not, add a new one
  if (!already_exists && !too_long)
  {
    prev_track_vertex_idx_ = closest_vertex_idx;
    addNewLocalVertex(vertex_msg, graph);
  }
}

void DualStateGraph::addNewPrunedVertex(geometry_msgs::Pose& vertex_msg, graph_utils::TopologicalGraph& graph)
{
  // Extract the point
  geometry_msgs::Point new_vertex_location;
  new_vertex_location = vertex_msg.position;

  bool already_exists = false;
  bool too_long = false;
  double distance = -1;
  int closest_vertex_idx = -1;
  geometry_msgs::Point closest_vertex_location;
  if (!graph.vertices.empty())
  {
    closest_vertex_idx = graph_utils_ns::GetClosestVertexIdxToPoint(graph, new_vertex_location);
    closest_vertex_location = graph.vertices[closest_vertex_idx].location;
    distance = misc_utils_ns::PointXYZDist(new_vertex_location, closest_vertex_location);
    if (distance < kMinVertexDist)
    {
      already_exists = true;
    }
    if (distance > kMaxLongEdgeDist || fabs(new_vertex_location.z - closest_vertex_location.z) > kMaxVertexDiffAlongZ)
    {
      too_long = true;
    }
  }

  // Check again if there is collision between the new node and its closest
  // node. Although this has been done for local
  // graph, but the octomap might be updated
  if (!already_exists && !too_long)
  {
    Eigen::Vector3d origin;
    Eigen::Vector3d end;
    origin.x() = new_vertex_location.x;
    origin.y() = new_vertex_location.y;
    origin.z() = new_vertex_location.z;
    end.x() = closest_vertex_location.x;
    end.y() = closest_vertex_location.y;
    end.z() = closest_vertex_location.z;
    // if (volumetric_mapping::OctomapManager::CellStatus::kFree ==
    //     manager_->getLineStatusBoundingBox(origin, end, robot_bounding))
    if (ChunkCellStatus::FREE == ChunkmapGetLineStatusBoundingBox(chunk_map_, origin, end, robot_bounding))
    {
      prev_track_vertex_idx_ = closest_vertex_idx;
      addNewLocalVertex(vertex_msg, graph);
    }
  }
}

void DualStateGraph::addNewLocalVertex(geometry_msgs::Pose& vertex_msg, graph_utils::TopologicalGraph& graph)
{
  // Add a new vertex to the graph associated with the input keypose
  // Return the index of the vertex

  // Extract the point
  geometry_msgs::Point new_vertex_location;
  new_vertex_location = vertex_msg.position;

  // Create a new vertex
  graph_utils::Vertex vertex;
  vertex.location = new_vertex_location;
  vertex.vertex_id = (int)graph.vertices.size();
  vertex.information_gain = vertex_msg.orientation.y;

  // Add this vertex to the graph
  graph.vertices.push_back(vertex);
  track_localvertex_idx_ = (int)graph.vertices.size() - 1;
  auto& vertex_new = graph.vertices[track_localvertex_idx_];
  // If there are already other vertices
  if (graph.vertices.size() > 1)
  {
    // Add a parent backpointer to previous vertex
    vertex_new.parent_idx = prev_track_vertex_idx_;

    // Add an edge to previous vertex
    addEdgeWithoutCheck(vertex_new.parent_idx, track_localvertex_idx_, graph);

    // Also add edges to nearby vertices
    for (auto& graph_vertex : graph.vertices)
    {
      // If within a distance
      float dist_diff = misc_utils_ns::PointXYZDist(graph_vertex.location, vertex_new.location);
      if (dist_diff < kConnectVertexDistMax)
      {
        // Add edge from this nearby vertex to current vertex
        addEdge(graph_vertex.vertex_id, vertex_new.vertex_id, graph);
      }
    }
  }
  else
  {
    // This is the first vertex -- no edges
    vertex.parent_idx = -1;
  }
}

void DualStateGraph::addNewLocalVertexWithoutEdge(geometry_msgs::Pose& vertex_msg, graph_utils::TopologicalGraph& graph)
{
  // Add a new vertex to the graph associated with the input keypose
  // Return the index of the vertex

  // Extract the point
  geometry_msgs::Point new_vertex_location;
  new_vertex_location = vertex_msg.position;

  // Create a new vertex
  graph_utils::Vertex vertex;
  vertex.location = new_vertex_location;
  vertex.vertex_id = (int)graph.vertices.size();
  vertex.information_gain = vertex_msg.orientation.y;

  // Add this vertex to the graph
  graph.vertices.push_back(vertex);
  track_localvertex_idx_ = (int)graph.vertices.size() - 1;
  auto& vertex_new = graph.vertices[track_localvertex_idx_];
}

void DualStateGraph::addNewGlobalVertexWithoutDuplicates(geometry_msgs::Pose& vertex_msg)
{
  // Same as addNewLocalVertex but only adds vertex if a similar vertex doesn't
  // already exist

  // Extract the point
  geometry_msgs::Point new_vertex_location;
  new_vertex_location = vertex_msg.position;

  // Check if a similar vertex already exists
  bool already_exists = false;
  bool too_long = false;
  double distance = -1;
  int closest_vertex_idx = -1;
  if (!global_graph_.vertices.empty())
  {
    closest_vertex_idx = graph_utils_ns::GetClosestVertexIdxToPoint(global_graph_, new_vertex_location);
    auto& closest_vertex_location = global_graph_.vertices[closest_vertex_idx].location;
    distance = misc_utils_ns::PointXYZDist(new_vertex_location, closest_vertex_location);
    if (distance < kMinVertexDist)
    {
      already_exists = true;
    }
    if (distance > kMaxLongEdgeDist || fabs(new_vertex_location.z - closest_vertex_location.z) > kMaxVertexDiffAlongZ)
    {
      too_long = true;
    }
  }

  // If not, add a new one
  if (!already_exists && !too_long)
  {
    prev_track_vertex_idx_ = closest_vertex_idx;
    addNewGlobalVertex(vertex_msg);
  }
}

void DualStateGraph::addNewGlobalVertexWithKeypose(geometry_msgs::Pose& vertex_msg)
{
  // Same as addNewLocalVertex but only adds vertex if a similar vertex doesn't
  // already exist

  // Extract the point
  geometry_msgs::Point new_vertex_location;
  new_vertex_location = vertex_msg.position;

  // Check if a similar vertex already exists
  bool already_exists = false;
  bool too_long = false;
  double distance = -1;
  int closest_vertex_idx = -1;
  if (!global_graph_.vertices.empty())
  {
    closest_vertex_idx = graph_utils_ns::GetClosestVertexIdxToPoint(global_graph_, new_vertex_location);
    auto& closest_vertex_location = global_graph_.vertices[closest_vertex_idx].location;
    distance = misc_utils_ns::PointXYZDist(new_vertex_location, closest_vertex_location);
    if (distance < kMinVertexDist / 2)
    {
      already_exists = true;
    }
    if (distance > kMaxLongEdgeDist || fabs(new_vertex_location.z - closest_vertex_location.z) > kMaxVertexDiffAlongZ)
    {
      too_long = true;
    }
  }

  // If not, add a new one
  if (!already_exists && !too_long)
  {
    prev_track_vertex_idx_ = closest_vertex_idx;
    addNewGlobalVertex(vertex_msg);
    addGlobalEdgeWithoutCheck(prev_track_keypose_vertex_idx_, track_globalvertex_idx_, true);
    prev_track_keypose_vertex_idx_ = track_globalvertex_idx_;
  }
}

void DualStateGraph::addNewGlobalVertex(geometry_msgs::Pose& vertex_msg)
{
  // Extract the point
  geometry_msgs::Point new_vertex_location;
  new_vertex_location = vertex_msg.position;

  // Create a new vertex
  graph_utils::Vertex vertex;
  vertex.location = new_vertex_location;
  vertex.vertex_id = (int)global_graph_.vertices.size();
  vertex.information_gain = vertex_msg.orientation.y;

  // Add this vertex to the graph
  global_graph_.vertices.push_back(vertex);
  track_globalvertex_idx_ = (int)global_graph_.vertices.size() - 1;
  auto& vertex_new = global_graph_.vertices[track_globalvertex_idx_];
  // If there are already other vertices
  if (global_graph_.vertices.size() > 1)
  {
    // Add a parent backpointer to previous vertex
    vertex_new.parent_idx = prev_track_vertex_idx_;

    // Add an edge to previous vertex
    addGlobalEdgeWithoutCheck(prev_track_vertex_idx_, track_globalvertex_idx_, false);

    // Also add edges to nearby vertices
    for (auto& graph_vertex : global_graph_.vertices)
    {
      // If within a distance
      float dist_diff = misc_utils_ns::PointXYZDist(graph_vertex.location, vertex_new.location);
      if (dist_diff < kConnectVertexDistMax)
      {
        // Add edge from this nearby vertex to current vertex
        addGlobalEdge(graph_vertex.vertex_id, vertex_new.vertex_id);
      }
    }
  }
  else
  {
    // This is the first vertex -- no edges
    vertex.parent_idx = -1;
  }
}

void DualStateGraph::addEdgeWithoutCheck(int start_vertex_idx, int end_vertex_idx, graph_utils::TopologicalGraph& graph)
{
  // Add an edge in the graph from vertex start_vertex_idx to vertex
  // end_vertex_idx

  // Get the two vertices
  auto& start_vertex = graph.vertices[start_vertex_idx];
  auto& end_vertex = graph.vertices[end_vertex_idx];

  // Check if edge already exists -- don't duplicate
  for (auto& edge : start_vertex.edges)
  {
    if (edge.vertex_id_end == end_vertex_idx)
    {
      // don't add duplicate edge
      return;
    }
  }

  // Compute the distance between the two points
  float dist_diff = misc_utils_ns::PointXYZDist(start_vertex.location, end_vertex.location);

  // Add an edge connecting the current node and the existing node on the graph

  // Create two edge objects
  graph_utils::Edge edge_to_start;
  graph_utils::Edge edge_to_end;

  // Join the two edges
  edge_to_start.vertex_id_end = start_vertex_idx;
  edge_to_end.vertex_id_end = end_vertex_idx;

  // Compute the traversal cost
  // For now, this is just Euclidean distance
  edge_to_start.traversal_costs = dist_diff;
  edge_to_end.traversal_costs = dist_diff;

  // Add these two edges to the vertices
  start_vertex.edges.push_back(edge_to_end);
  end_vertex.edges.push_back(edge_to_start);

  localEdgeSize_++;
}

void DualStateGraph::addEdge(int start_vertex_idx, int end_vertex_idx, graph_utils::TopologicalGraph& graph)
{
  if (start_vertex_idx == end_vertex_idx)
  {
    // don't add edge to itself
    return;
  }

  // Get the edge distance
  float dist_edge =
      misc_utils_ns::PointXYZDist(graph.vertices[start_vertex_idx].location, graph.vertices[end_vertex_idx].location);

  if (dist_edge > kMaxLongEdgeDist)
  {
  }
  else
  {
    std::vector<int> path;
    graph_utils_ns::ShortestPathBtwVertex(path, graph, start_vertex_idx, end_vertex_idx);
    bool path_exists = true;
    float dist_path = 0;
    if (path.empty())
    {
      path_exists = false;
    }
    else
    {
      dist_path = graph_utils_ns::PathLength(path, graph);
    }

    if ((!path_exists || (path_exists && ((dist_path / dist_edge) >= kExistingPathRatioThreshold))) &&
        (!zCollisionCheck(start_vertex_idx, end_vertex_idx, graph))/* &&
        (!grid_->collisionCheckByTerrain(graph.vertices[start_vertex_idx].location,
                                         graph.vertices[end_vertex_idx].location))*/)
    {
      Eigen::Vector3d origin;
      Eigen::Vector3d end;
      origin.x() = graph.vertices[start_vertex_idx].location.x;
      origin.y() = graph.vertices[start_vertex_idx].location.y;
      origin.z() = graph.vertices[start_vertex_idx].location.z;
      end.x() = graph.vertices[end_vertex_idx].location.x;
      end.y() = graph.vertices[end_vertex_idx].location.y;
      end.z() = graph.vertices[end_vertex_idx].location.z;
      // if (volumetric_mapping::OctomapManager::CellStatus::kFree ==
      //     manager_->getLineStatusBoundingBox(origin, end, robot_bounding))
      if (ChunkCellStatus::FREE == ChunkmapGetLineStatusBoundingBox(chunk_map_, origin, end, robot_bounding))
      {
        addEdgeWithoutCheck(start_vertex_idx, end_vertex_idx, graph);
      }
    }
  }
}

void DualStateGraph::addGlobalEdgeWithoutCheck(int start_vertex_idx, int end_vertex_idx, bool trajectory_edge)
{
  // Add an edge in the graph from vertex start_vertex_idx to vertex
  // end_vertex_idx

  // Get the two vertices
  auto& start_vertex = global_graph_.vertices[start_vertex_idx];
  auto& end_vertex = global_graph_.vertices[end_vertex_idx];

  // Check if edge already exists -- don't duplicate
  for (auto& edge : start_vertex.edges)
  {
    if (edge.vertex_id_end == end_vertex_idx)
    {
      // don't add duplicate edge
      edge.keypose_edge = trajectory_edge;
      for (auto& edge : end_vertex.edges)
      {
        if (edge.vertex_id_end == start_vertex_idx)
        {
          // don't add duplicate edge
          edge.keypose_edge = trajectory_edge;
          break;
        }
      }
      return;
    }
  }

  // Compute the distance between the two points
  float dist_diff = misc_utils_ns::PointXYZDist(start_vertex.location, end_vertex.location);

  // Create two edge objects
  graph_utils::Edge edge_to_start;
  graph_utils::Edge edge_to_end;

  // Join the two edges
  edge_to_start.vertex_id_end = start_vertex_idx;
  edge_to_end.vertex_id_end = end_vertex_idx;

  // Compute the traversal cost
  // For now, this is just Euclidean distance
  edge_to_start.traversal_costs = dist_diff;
  edge_to_end.traversal_costs = dist_diff;

  edge_to_start.keypose_edge = trajectory_edge;
  edge_to_end.keypose_edge = trajectory_edge;

  // Add these two edges to the vertices
  start_vertex.edges.push_back(edge_to_end);
  end_vertex.edges.push_back(edge_to_start);

  globalEdgeSize_++;
}

void DualStateGraph::addGlobalEdge(int start_vertex_idx, int end_vertex_idx)
{
  if (start_vertex_idx == end_vertex_idx)
  {
    return;
  }

  // Get the edge distance
  float dist_edge = misc_utils_ns::PointXYZDist(global_graph_.vertices[start_vertex_idx].location,
                                                global_graph_.vertices[end_vertex_idx].location);

  if (dist_edge > kMaxLongEdgeDist)
  {
    // VERY long edge. Don't add it
  }
  else
  {
    // Get the path distance
    std::vector<int> path;
    graph_utils_ns::ShortestPathBtwVertex(path, global_graph_, start_vertex_idx, end_vertex_idx);
    bool path_exists = true;
    float dist_path = 0;

    // Check if there is an existing path
    if (path.empty())
    {
      // No path exists
      path_exists = false;
    }
    else
    {
      // Compute path length
      dist_path = graph_utils_ns::PathLength(path, global_graph_);
      // ROS_WARN("path exists of edge of length %f, where path exists of length
      // %f", dist_edge, dist_path);
    }

    // Check if ratio is beyond threshold
    // bool collision = false;

    if ((!path_exists || (path_exists && ((dist_path / dist_edge) >= kExistingPathRatioThresholdGlobal))) &&
        (!zCollisionCheck(start_vertex_idx, end_vertex_idx, global_graph_)))
    {
      Eigen::Vector3d origin;
      Eigen::Vector3d end;
      origin.x() = global_graph_.vertices[start_vertex_idx].location.x;
      origin.y() = global_graph_.vertices[start_vertex_idx].location.y;
      origin.z() = global_graph_.vertices[start_vertex_idx].location.z;
      end.x() = global_graph_.vertices[end_vertex_idx].location.x;
      end.y() = global_graph_.vertices[end_vertex_idx].location.y;
      end.z() = global_graph_.vertices[end_vertex_idx].location.z;
      // if (volumetric_mapping::OctomapManager::CellStatus::kFree ==
      //     manager_->getLineStatusBoundingBox(origin, end, robot_bounding))
      if (ChunkCellStatus::FREE == ChunkmapGetLineStatusBoundingBox(chunk_map_, origin, end, robot_bounding))
      {
        addGlobalEdgeWithoutCheck(start_vertex_idx, end_vertex_idx, false);
      }
    }
  }
}

bool DualStateGraph::zCollisionCheck(int start_vertex_idx, int end_vertex_idx, graph_utils::TopologicalGraph graph)
{
  auto& start_vertex_location = graph.vertices[start_vertex_idx].location;
  auto& end_vertex_location = graph.vertices[end_vertex_idx].location;
  float x_diff = start_vertex_location.x - end_vertex_location.x;
  float y_diff = start_vertex_location.y - end_vertex_location.y;
  float z_diff = start_vertex_location.z - end_vertex_location.z;
  float ang_diff = atan(fabs(z_diff) / sqrt(x_diff * x_diff + y_diff * y_diff)) * 180 / M_PI;
  if (fabs(ang_diff) > kMaxVertexAngleAlongZ || fabs(z_diff) > kMaxVertexDiffAlongZ)
  {
    return true;
  }
  return false;
}

int DualStateGraph::getLocalVertexSize()
{
  return (local_graph_.vertices.size());
}

int DualStateGraph::getLocalEdgeSize()
{
  return localEdgeSize_;
}

int DualStateGraph::getGlobalVertexSize()
{
  return (global_graph_.vertices.size());
}

int DualStateGraph::getGlobalEdgeSize()
{
  return globalEdgeSize_;
}

void DualStateGraph::pruneGraph(geometry_msgs::Point root)
{
  int closest_vertex_idx_to_root;
  if (!local_graph_.vertices.empty())
  {
    closest_vertex_idx_to_root = graph_utils_ns::GetClosestVertexIdxToPoint(local_graph_, root);
    auto& closest_vertex_location = local_graph_.vertices[closest_vertex_idx_to_root].location;
    double distance = misc_utils_ns::PointXYZDist(root, closest_vertex_location);
    Eigen::Vector3d origin(root.x, root.y, root.z);
    Eigen::Vector3d end(closest_vertex_location.x, closest_vertex_location.y, closest_vertex_location.z);
    if (distance > kMaxDistToPrunedRoot)
    {
      return;
    }
    // else if (volumetric_mapping::OctomapManager::CellStatus::kFree !=
    //          manager_->getLineStatusBoundingBox(origin, end, robot_bounding))

    // Bug here, always occupied
    // else if (ChunkCellStatus::FREE != ChunkmapGetLineStatusBoundingBox(chunk_map_, origin, end, robot_bounding))
    // {
    //   return;
    // }
  }

  std::vector<int> path;
  for (int path_id = 0; path_id < local_graph_.vertices.size(); path_id++)
  {
    path.clear();
    graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, closest_vertex_idx_to_root, path_id);
    if (path.size() == 0)
    {
      continue;
    }
    if (misc_utils_ns::PointXYZDist(root, local_graph_.vertices[path_id].location) > kMaxPrunedNodeDist)
      continue;
    for (int j = 0; j < path.size(); j++)
    {
      tempvertex_.position = local_graph_.vertices[path[j]].location;
      tempvertex_.position.z = MIN(tempvertex_.position.z, root.z);
      tempvertex_.orientation.y = local_graph_.vertices[path[j]].information_gain;
      addNewPrunedVertex(tempvertex_, pruned_graph_);
    }
  }
}

void DualStateGraph::newObstacle(pcl::PointCloud<pcl::PointXYZI>::Ptr localObs) {
  *obstaclePoint_ += *localObs;
  pcl::VoxelGrid<pcl::PointXYZI> vox;
  vox.setInputCloud(obstaclePoint_);
  vox.setLeafSize(0.5,0.5,0.5);
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp = 
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
  vox.filter(*temp);
  obstaclePoint_ = temp;
  return;
}

void DualStateGraph::pruneGraph_szz(geometry_msgs::Point root) {
  pcl::VoxelGrid<pcl::PointXYZI> vox;
  vox.setInputCloud(collisionPoint_);
  vox.setLeafSize(0.5, 0.5, 0.5);
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp = 
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
  vox.filter(*temp);
  collisionPoint_->clear();
  
  // prune local graph
  for(int i = 0; i < temp->points.size(); i++) {
    double dis = sqrt((root.x - temp->points[i].x) * (root.x - temp->points[i].x) + 
                      (root.y - temp->points[i].y) * (root.y - temp->points[i].y));
    if(dis > kMaxPrunedNodeDist+2)
      continue;
    else if(dis < kMaxPrunedNodeDist+2 && dis > kMaxPrunedNodeDist) {
      temp->points[i].intensity = 10;
      collisionPoint_->push_back(temp->points[i]);
    }
    else
      collisionPoint_->push_back(temp->points[i]);
  }

  int closest_vertex_idx_to_root;
  if (!global_graph_.vertices.empty())
  {
    closest_vertex_idx_to_root = graph_utils_ns::GetClosestVertexIdxToPoint(global_graph_, root);
    auto& closest_vertex_location = global_graph_.vertices[closest_vertex_idx_to_root].location;
    double distance = misc_utils_ns::PointXYZDist(root, closest_vertex_location);
    Eigen::Vector3d origin(root.x, root.y, root.z);
    Eigen::Vector3d end(closest_vertex_location.x, closest_vertex_location.y, closest_vertex_location.z);
    if (distance > kMaxDistToPrunedRoot)
    {
      return;
    }
    // else if (volumetric_mapping::OctomapManager::CellStatus::kFree != manager_->getLineStatusBoundingBox(origin, end, robot_bounding))
    else if (ChunkCellStatus::FREE != ChunkmapGetLineStatusBoundingBox(chunk_map_, origin, end, robot_bounding))
    {
      return;
    }
  }
  pcl::PointXYZI tempPoint;
  // std::vector<int> path;
#if 1
  for (int path_id = 0; path_id < global_graph_.vertices.size(); path_id++)
  {
    // double dis = misc_utils_ns::PointXYZDist(root, global_graph_.vertices[path_id].location);
    double dis = std::max(std::abs(root.x - global_graph_.vertices[path_id].location.x), std::abs(root.y - global_graph_.vertices[path_id].location.y));
    if( dis > kMaxPrunedNodeDist+2 || std::abs(root.z - global_graph_.vertices[path_id].location.z) > 1)
      continue;
    if (dis > kMaxPrunedNodeDist) {
        tempPoint.x = global_graph_.vertices[path_id].location.x;
        tempPoint.y = global_graph_.vertices[path_id].location.y;
        tempPoint.z = global_graph_.vertices[path_id].location.z;
        tempPoint.intensity =  10;
        collisionPoint_->push_back(tempPoint);
        continue;
    } 
    else {
      tempPoint.x = global_graph_.vertices[path_id].location.x;
      tempPoint.y = global_graph_.vertices[path_id].location.y;
      tempPoint.z = global_graph_.vertices[path_id].location.z;
      tempPoint.intensity =  0;
      collisionPoint_->push_back(tempPoint);

      tempvertex_.position = global_graph_.vertices[path_id].location;
      // tempvertex_.position.z = MIN(tempvertex_.position.z, root.z);
      tempvertex_.orientation.y = global_graph_.vertices[path_id].information_gain;
      addNewPrunedVertex(tempvertex_, pruned_graph_);
    }
  }
#endif

  Eigen::Vector3f rootVec(root.x, root.y, root.z);
  for (const auto &point : *obstaclePoint_)
  {
    Eigen::Vector3f delta = (rootVec - point.getVector3fMap()).cwiseAbs();
    double dis = std::max(delta.x(), delta.y());
    if( dis > kMaxPrunedNodeDist+2 || delta.z() > 1)
      continue;
    // point.intensity = 10;
    collisionPoint_->points.push_back(point);
  }
}

void DualStateGraph::pruneGlobalGraph()
{
  if (global_graph_.vertices.size() > 0)
  {
    for (int i = 0; i < global_graph_.vertices.size(); i++)
    {
      Eigen::Vector3d vertex_position(global_graph_.vertices[i].location.x, global_graph_.vertices[i].location.y,
                                      global_graph_.vertices[i].location.z);
      // if (volumetric_mapping::OctomapManager::CellStatus::kFree != manager_->getCellStatusPoint(vertex_position))
      if (ChunkCellStatus::FREE != ChunkmapGetCellStatusPoint(chunk_map_, vertex_position))
      {
        for (int j = 0; j < global_graph_.vertices[i].edges.size(); j++)
        {
          int end_vertex_id = global_graph_.vertices[i].edges[j].vertex_id_end;
          for (int m = 0; m < global_graph_.vertices[end_vertex_id].edges.size(); m++)
          {
            if (global_graph_.vertices[end_vertex_id].edges[m].vertex_id_end == i)
            {
              global_graph_.vertices[end_vertex_id].edges.erase(global_graph_.vertices[end_vertex_id].edges.begin() +
                                                                m);
              break;
            }
          }
        }
      }
    }
  }
}

void DualStateGraph::setCurrentPlannerStatus(bool status)
{
  planner_status_ = status;
}

void DualStateGraph::clearLocalGraph()
{
  local_graph_.vertices.clear();
  track_localvertex_idx_ = 0;
  robot_yaw_ = 0.0;
  localEdgeSize_ = 0;
  best_gain_ = 0;
  best_vertex_id_ = 0;
  gainID_.clear();
}

bool cmpGain(graph_utils::Vertex a, graph_utils::Vertex b) {
  return a.information_gain > b.information_gain;
}

std::vector <geometry_msgs::Point> DualStateGraph::getAsymmetricTSPMatrix(std::vector<geometry_msgs::Point> loopPoint) {

  // filter radius
  geometry_msgs::Point circle;
  circle.x = (loopPoint[0].x + loopPoint[1].x)/2;
  circle.y = (loopPoint[0].y + loopPoint[1].y)/2;
  circle.z = (loopPoint[0].z + loopPoint[1].z)/2;
  double radius = misc_utils_ns::PointXYDist(loopPoint[0], circle) *2;

  // find all frontier in radius with information gain > 0
  std::vector <graph_utils::Vertex> top5;
  for(auto& graph_vertex : local_graph_.vertices) {
    double dist = misc_utils_ns::PointXYDist(graph_vertex.location, circle);
    if (dist > radius) continue;
    if(graph_vertex.information_gain > 0) {
        top5.push_back(graph_vertex);
        // if(top5.size() > 5) {
        //   sort(top5.begin(), top5.end(), cmpGain);
        //   top5.pop_back();
        // }
    }
  }
  sort(top5.begin(), top5.end(), cmpGain);
  for(int i = 0; i < top5.size(); i++) {
    for(int j = 0; j < i; j++) {
      if(misc_utils_ns::PointXYDist(top5[i].location, top5[j].location) < 5) {
        top5.erase(top5.begin()+i);
        i--;
        break;
      }
    }
  }
    

  for(int i = 0; i < top5.size(); i++) {
    loopPoint.push_back(top5[i].location);
    // loopPointId.push_back(graph_vertex.id);
  }

  return loopPoint;
}

std::vector <geometry_msgs::Point> DualStateGraph::solveTSP(std::vector<geometry_msgs::Point> loopPoint) {
  #if 1
  // construct ATSP Matrix
  int dimen = loopPoint.size();
  Eigen::MatrixXd mat;
  mat.resize(dimen, dimen);
  for(int i = 0; i < loopPoint.size(); i++) 
    for(int j = 1; j < loopPoint.size(); j++) {
      mat(i, j) = misc_utils_ns::PointXYDist(loopPoint[i], loopPoint[j]);
    }
  mat.leftCols<1>().setZero();
  #else 
  int dimen = loopPoint.size();
  Eigen::MatrixXd mat;
  mat.resize(dimen, dimen);
  
  for(int i = 0; i < loopPoint.size(); i++) 
    for(int j = 0; j < loopPoint.size(); j++) {
      mat(i, j) = misc_utils_ns::PointXYDist(loopPoint[i], loopPoint[j]);
    }
  mat((dimen-1), 0) = 0;
  // mat(0, 1) *= 3;
  #endif

  // solve TSP
  std::ofstream par_file("/ros_ws/chunkmap_20221210_szz/src/lkh_tsp_solver/resource/single.par");
  par_file << "PROBLEM_FILE = " << "/ros_ws/chunkmap_20221210_szz/src/lkh_tsp_solver/resource/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << "/ros_ws/chunkmap_20221210_szz/src/lkh_tsp_solver/resource/single.txt\n";
  par_file << "RUNS = 1\n";
  par_file.close();

  std::ofstream prob_file("/ros_ws/chunkmap_20221210_szz/src/lkh_tsp_solver/resource/single.tsp");
  std::string prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + std::to_string(dimen) +
      "\nEDGE_WEIGHT_TYPE : "
      "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";
  prob_file << prob_spec;
  const int scale = 100;
  for (int i = 0; i < loopPoint.size(); ++i) {
      for (int j = 0; j < loopPoint.size(); ++j) {
        int int_cost = mat(i, j) * scale;
        prob_file << int_cost << " ";
      }
      prob_file << "\n";
  }
  prob_file << "EOF";
  prob_file.close();
  std::string lkh_path = "/ros_ws/chunkmap_20221210_szz/src/lkh_tsp_solver/resource/single.par";
  solveTSPLKH(lkh_path.c_str());
  std::ifstream res_file("/ros_ws/chunkmap_20221210_szz/src/lkh_tsp_solver/resource/single.txt");
  std::string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0) break;
  }
  std::vector <int> indices;
  while (getline(res_file, res)) {
      // Read indices of frontiers in optimal tour
      int id = std::stoi(res);
      if (id == 1)  // Ignore the current state
        continue;
      if (id == -1) break;
      indices.push_back(id - 1);  // Idx of solver-2 == Idx of frontier
    }
  res_file.close();

  std::vector<geometry_msgs::Point> tspTour;
  tspTour.push_back(loopPoint[0]);
  for(int i = indices.size()-1; i >= 0; i--) {
    tspTour.push_back(loopPoint[indices[i]]);
  }
  
  return tspTour;
}

double DualStateGraph::getGain(geometry_msgs::Point robot_position)
{
  for (auto& graph_vertex : local_graph_.vertices)
  {
    if (graph_vertex.vertex_id > 0)
    {
      // Get the path distance
      std::vector<int> path;
      graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, 0, graph_vertex.vertex_id);
      bool path_exists = true;
      float dist_path = 0;

      // Check if there is an existing path
      if (path.empty())
      {
        // No path exists
        path_exists = false;
        graph_vertex.information_gain = 0;
        continue;
      }
      else
      {
        // Compute path length
        dist_path = graph_utils_ns::PathLength(path, local_graph_);
      }
      int NodeCountArround = 1;
      for (auto& vertex : local_graph_.vertices)
      {
        float dist_edge = misc_utils_ns::PointXYZDist(graph_vertex.location, vertex.location);
        if (dist_edge < kSurroundRange)
        {
          NodeCountArround++;
        }
      }
      if (misc_utils_ns::PointXYZDist(graph_vertex.location, robot_position) < kMinGainRange)
        graph_vertex.information_gain = 0;

      if (graph_vertex.information_gain > 0)
      {
        if (std::isnan(explore_direction_.x()) || std::isnan(explore_direction_.y()))
          DTWValue_ = exp(1);
        else
        {
          DTW(path, robot_position);
          graph_vertex.information_gain = 0.3 * graph_vertex.information_gain / log(DTWValue_ * kDirectionCoeff);
        }
      }
      graph_vertex.information_gain =
          graph_vertex.information_gain * kDegressiveCoeff / dist_path * exp(0.1 * NodeCountArround);
    }
  }
  for (auto& graph_vertex : local_graph_.vertices)
  {
    if (graph_vertex.vertex_id > 0)
    {
      double path_information_gain = 0;
      std::vector<int> path;
      graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, 0, graph_vertex.vertex_id);
      if (path.empty())
      {
        // No path exists
        continue;
      }
      for (int i = 0; i < path.size(); i++)
      {
        path_information_gain += local_graph_.vertices[path[i]].information_gain;
      }
      if (path_information_gain > 0)
      {
        gainID_.push_back(graph_vertex.vertex_id);
      }

      if (path_information_gain > best_gain_)
      {
        best_vertex_id_ = graph_vertex.vertex_id;
        best_gain_ = path_information_gain;
      }
    }
  }
  // ROS_INFO("Best gain is %f.\n Best vertex id is %d", best_gain_,
  // best_vertex_id_);
  return best_gain_;
}

double DualStateGraph::getGain_szz(geometry_msgs::Point robot_position, StateVec twist, float dist)
{
  for (auto& graph_vertex : local_graph_.vertices)
  {
    graph_vertex.exploration_gain = graph_vertex.information_gain;
    if (graph_vertex.vertex_id > 0)
    {
      // Get the path distance
      std::vector<int> path;
      graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, 0, graph_vertex.vertex_id);
      bool path_exists = true;
      float dist_path = 0;

      // Check if there is an existing path
      if (path.empty())
      {
        // No path exists
        path_exists = false;
        graph_vertex.information_gain = 0;
        graph_vertex.exploration_gain = 0;
        continue;
      }
      else
      {
        // Compute path length
        dist_path = graph_utils_ns::PathLength(path, local_graph_);
      }
      int NodeCountArround = 1;
      for (auto& vertex : local_graph_.vertices)
      {
        float dist_edge = misc_utils_ns::PointXYZDist(graph_vertex.location, vertex.location);
        if (dist_edge < kSurroundRange)
        {
          NodeCountArround++;
        }
      }
      if (misc_utils_ns::PointXYZDist(graph_vertex.location, robot_position) < kMinGainRange){
        graph_vertex.information_gain = 0;
        graph_vertex.exploration_gain = 0;
      }
      // if (1)
      if (graph_vertex.information_gain > 0)
      {
          int node_num = path.size();
          dist_path = graph_utils_ns::PathLength(path, local_graph_) / node_num;
          
          double direction_cost = 0;

          // distance cost
          // geometry_msgs::Point node_position;
          // for (int i = 0; i < node_num; i++)
          // { 
            // node_position.x = robot_position.x + twist.x()* dist_path * i;
            // node_position.y = robot_position.y + twist.y() * dist_path * i;
            // direction_cost += misc_utils_ns::PointXYDist(local_graph_.vertices[path[i]].location, node_position);
          // }

          //angle cost
          Eigen::Vector3d dir;
          dir[0] = graph_vertex.location.x - robot_position.x;
          dir[1] = graph_vertex.location.y - robot_position.y;
          dir[2] = graph_vertex.location.z - robot_position.z;
          direction_cost = acos(dir.dot(twist) / dir.norm() / twist.norm());
          if(direction_cost < 0.1) direction_cost = 0.1;
          // DTW_szz(path, robot_position, twist);
          // graph_vertex.information_gain = graph_vertex.information_gain / log(DTWValue_ * kDirectionCoeff);

          // graph_vertex.information_gain = graph_vertex.information_gain / direction_cost;
          graph_vertex.exploration_gain = graph_vertex.information_gain / direction_cost;
      }
      // graph_vertex.information_gain =
      //     graph_vertex.information_gain * kDegressiveCoeff / dist_path * exp(0.1 * NodeCountArround);
    }
  }
  #ifdef outfile
  std::ofstream directionFile;
  directionFile.open("/home/szz/Documents/dsv_planner_graphgain+tsp_20230110/result/directionFile.txt");
  #endif
  for (auto& graph_vertex : local_graph_.vertices)
  {
    if (graph_vertex.vertex_id > 0)
    {
      double path_information_gain = 0;
      std::vector<int> path;
      graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, 0, graph_vertex.vertex_id);
      if (path.empty())
      {
        // No path exists
        continue;
      }
      #if 0
      path_information_gain = graph_vertex.information_gain;
      #else
      for (int i = 0; i < path.size(); i++)
      {
        // path_information_gain += local_graph_.vertices[path[i]].information_gain;
        path_information_gain += local_graph_.vertices[path[i]].exploration_gain;
      }
      #endif
      if (path_information_gain > 0)
      {
        gainID_.push_back(graph_vertex.vertex_id);
      }

      // if (dist < 3 && path_information_gain > best_gain_) // indoor
      // if (dist < 4 && path_information_gain > best_gain_) // indoor
      if (path_information_gain > best_gain_)
      {
        best_vertex_id_ = graph_vertex.vertex_id;
        best_gain_ = path_information_gain;
      }
    }
    #ifdef outfile
      directionFile << graph_vertex.location.x << " " << graph_vertex.location.y << " " << graph_vertex.location.z
                    << " " << graph_vertex.exploration_gain << std::endl;
    #endif
  }
  // ROS_INFO("Best gain is %f.\n Best vertex id is %d", best_gain_,
  // best_vertex_id_);
  return best_gain_;
}

void DualStateGraph::DTW_szz(std::vector<int> path, geometry_msgs::Point robot_position, StateVec twist)
{
  float dist_path = 0;
  int node_num = path.size();
  dist_path = graph_utils_ns::PathLength(path, local_graph_) / node_num;
  std::vector<geometry_msgs::Point> exp_node;
  geometry_msgs::Point node_position;
  for (int i = 0; i < node_num; i++)
  {
    node_position.x = robot_position.x + twist.x()* dist_path * (i + 1);
    node_position.y = robot_position.y + twist.y() * dist_path * (i + 1);
    exp_node.push_back(node_position);
  }

  std::vector<std::vector<double>> DTWValue;
  std::vector<double> sub_DTWValue;
  for (int i = 0; i < node_num + 1; i++)
  {
    for (int j = 0; j < node_num + 1; j++)
    {
      sub_DTWValue.push_back(1000000);
    }
    DTWValue.push_back(sub_DTWValue);
    sub_DTWValue.clear();
  }
  DTWValue[0][0] = 0;

  double dist = 0;
  for (int i = 1; i < node_num + 1; i++)
  {
    for (int j = 1; j < node_num + 1; j++)
    {
      dist = misc_utils_ns::PointXYDist(local_graph_.vertices[path[i - 1]].location, exp_node[j - 1]);
      DTWValue[i][j] = dist + std::fmin(std::fmin(DTWValue[i - 1][j], DTWValue[i][j - 1]), DTWValue[i - 1][j - 1]);
    }
  }
  DTWValue_ = DTWValue[node_num][node_num];
}

void DualStateGraph::DTW(std::vector<int> path, geometry_msgs::Point robot_position)
{
  float dist_path = 0;
  int node_num = path.size();
  dist_path = graph_utils_ns::PathLength(path, local_graph_) / node_num;
  std::vector<geometry_msgs::Point> exp_node;
  geometry_msgs::Point node_position;
  for (int i = 0; i < node_num; i++)
  {
    node_position.x = robot_position.x + explore_direction_.x() * dist_path * (i + 1);
    node_position.y = robot_position.y + explore_direction_.y() * dist_path * (i + 1);
    exp_node.push_back(node_position);
  }

  std::vector<std::vector<double>> DTWValue;
  std::vector<double> sub_DTWValue;
  for (int i = 0; i < node_num + 1; i++)
  {
    for (int j = 0; j < node_num + 1; j++)
    {
      sub_DTWValue.push_back(1000000);
    }
    DTWValue.push_back(sub_DTWValue);
    sub_DTWValue.clear();
  }
  DTWValue[0][0] = 0;

  double dist = 0;
  for (int i = 1; i < node_num + 1; i++)
  {
    for (int j = 1; j < node_num + 1; j++)
    {
      dist = misc_utils_ns::PointXYDist(local_graph_.vertices[path[i - 1]].location, exp_node[j - 1]);
      DTWValue[i][j] = dist + std::fmin(std::fmin(DTWValue[i - 1][j], DTWValue[i][j - 1]), DTWValue[i - 1][j - 1]);
    }
  }
  DTWValue_ = DTWValue[node_num][node_num];
}

void DualStateGraph::updateGlobalGraph()
{
  std::vector<int> path;
  graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, 0, best_vertex_id_);
  for (int i = 0; i < path.size(); i++)
  {
    tempvertex_.position = local_graph_.vertices[path[i]].location;
    tempvertex_.orientation.y = local_graph_.vertices[path[i]].information_gain;
    addNewGlobalVertexWithoutDuplicates(tempvertex_);
  }
  if (gainID_.size() > 0)
  {
    for (int i = 0; i < gainID_.size(); i++)
    {
      path.clear();
      graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, 0, gainID_[i]);
      for (int j = 0; j < path.size(); j++)
      {
        tempvertex_.position = local_graph_.vertices[path[j]].location;
        tempvertex_.orientation.y = local_graph_.vertices[path[j]].information_gain;
        addNewGlobalVertexWithoutDuplicates(tempvertex_);
      }
    }
  }
  publishGlobalGraph();
}

void DualStateGraph::updateExploreDirection()
{
  std::vector<int> path;
  graph_utils_ns::ShortestPathBtwVertex(path, local_graph_, 0, best_vertex_id_);
  if (path.empty() || best_vertex_id_ >= local_graph_.vertices.size())
  {
    return;
  }
  if (path.size() > 5)
  {
    explore_direction_[0] =
        local_graph_.vertices[path[path.size() - 1]].location.x - local_graph_.vertices[path[0]].location.x;
    explore_direction_[1] =
        local_graph_.vertices[path[path.size() - 1]].location.y - local_graph_.vertices[path[0]].location.y;
  }
  else
  {
    explore_direction_[0] =
        local_graph_.vertices[path[path.size() - 1]].location.x - local_graph_.vertices[path[0]].location.x;
    explore_direction_[1] =
        local_graph_.vertices[path[path.size() - 1]].location.y - local_graph_.vertices[path[0]].location.y;
  }
  double length = sqrt(explore_direction_[0] * explore_direction_[0] + explore_direction_[1] * explore_direction_[1]);
  explore_direction_[0] = explore_direction_[0] / length;
  explore_direction_[1] = explore_direction_[1] / length;
}

Eigen::Vector3d DualStateGraph::getExploreDirection()
{
  Eigen::Vector3d exploreDirection(explore_direction_[0], explore_direction_[1], explore_direction_[2]);
  return exploreDirection;
}

geometry_msgs::Point DualStateGraph::getBestLocalVertexPosition()
{
  if(local_graph_.vertices.size() == 0) {
    geometry_msgs::Point point;
    point.x = 0;
    point.y = 0;
    point.z = 0;
    return point;
  }
  return local_graph_.vertices[best_vertex_id_].location;
}

geometry_msgs::Point DualStateGraph::getBestGlobalVertexPosition()
{
  geometry_msgs::Point best_vertex_location = local_graph_.vertices[best_vertex_id_].location;
  return best_vertex_location;
}

void DualStateGraph::keyposeCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  geometry_msgs::Pose keypose;
  keypose.position.x = msg->pose.pose.position.x;
  keypose.position.y = msg->pose.pose.position.y;
  keypose.position.z = msg->pose.pose.position.z;
  keypose.orientation.y = 0;
  addNewGlobalVertexWithKeypose(keypose);
}

void DualStateGraph::pathCallback(const nav_msgs::Path::ConstPtr& graph_path)
{
  if (graph_path->poses.size() > 2 && !local_graph_.vertices.empty())
  {
    for (int i = 1; i < graph_path->poses.size() - 1; i++)
    {
      int origin_vertex_id;
      int end_vertex_id;
      geometry_msgs::Point origin_position;
      geometry_msgs::Point end_position;
      origin_vertex_id = graph_utils_ns::GetClosestVertexIdxToPoint(local_graph_, graph_path->poses[i].pose.position);
      end_vertex_id = graph_utils_ns::GetClosestVertexIdxToPoint(local_graph_, graph_path->poses[i + 1].pose.position);
      origin_position = local_graph_.vertices[origin_vertex_id].location;
      end_position = local_graph_.vertices[end_vertex_id].location;
      Eigen::Vector3d origin(origin_position.x, origin_position.y, origin_position.z);
      Eigen::Vector3d end(end_position.x, end_position.y, end_position.z);
      double distance = misc_utils_ns::PointXYZDist(origin_position, robot_pos_);
      // if (volumetric_mapping::OctomapManager::CellStatus::kFree != manager_->getLineStatus(origin, end) ||
      if (ChunkCellStatus::FREE != ChunkmapGetLineStatus(chunk_map_, origin, end) ||
          (kCropPathWithTerrain && distance < kMinDistanceToRobotToCheck/* &&
           grid_->collisionCheckByTerrain(origin_position, end_position)*/))
      {
        for (int j = 0; j < local_graph_.vertices[origin_vertex_id].edges.size(); j++)
        {
          if (local_graph_.vertices[origin_vertex_id].edges[j].vertex_id_end == end_vertex_id &&
              local_graph_.vertices[origin_vertex_id].edges[j].keypose_edge == false)
          {
            local_graph_.vertices[origin_vertex_id].edges.erase(local_graph_.vertices[origin_vertex_id].edges.begin() +
                                                                j);
            if (planner_status_ == true && global_graph_.vertices[origin_vertex_id].edges[j].keypose_edge == false)
            {
              global_graph_.vertices[origin_vertex_id].edges.erase(
                  global_graph_.vertices[origin_vertex_id].edges.begin() + j);
            }
            break;
          }
        }
        for (int j = 0; j < local_graph_.vertices[end_vertex_id].edges.size(); j++)
        {
          if (local_graph_.vertices[end_vertex_id].edges[j].vertex_id_end == origin_vertex_id &&
              local_graph_.vertices[origin_vertex_id].edges[j].keypose_edge == false)
          {
            local_graph_.vertices[end_vertex_id].edges.erase(local_graph_.vertices[end_vertex_id].edges.begin() + j);
            if (planner_status_ == true && global_graph_.vertices[origin_vertex_id].edges[j].keypose_edge == false)
            {
              global_graph_.vertices[end_vertex_id].edges.erase(global_graph_.vertices[end_vertex_id].edges.begin() +
                                                                j);
            }
            break;
          }
        }
        execute();
        break;
      }
    }
  }
  execute();
}

void DualStateGraph::graphPlannerStatusCallback(const graph_planner::GraphPlannerStatusConstPtr& status)
{
  graph_planner::GraphPlannerStatus prev_status = graph_planner_status_;
  graph_planner_status_ = *status;
  if (prev_status.status != graph_planner_status_.status &&
      graph_planner_status_.status == graph_planner::GraphPlannerStatus::STATUS_IN_PROGRESS)
  {
    if (planner_status_ == false)
    {
      pruneGlobalGraph();
      publishGlobalGraph();
    }
  }
}

DualStateGraph::DualStateGraph(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private,
                               ChunkMap::Ptr chunk_map, OccupancyGrid* grid)
  : nh_(nh), nh_private_(nh_private)
{
  // manager_ = manager;
  chunk_map_ = chunk_map;
  grid_ = grid;
  initialize();
}

DualStateGraph::~DualStateGraph()
{
}

bool DualStateGraph::initialize()
{
  best_gain_ = 0;
  best_vertex_id_ = 0;
  explore_direction_[0] = 0;
  explore_direction_[1] = 0;
  explore_direction_[2] = 0;
  globalEdgeSize_ = 0;
  localEdgeSize_ = 0;
  robot_yaw_ = 0.0;
  track_localvertex_idx_ = 0;
  track_globalvertex_idx_ = 0;
  prev_track_keypose_vertex_idx_ = 0;
  DTWValue_ = 0;

  // Read in parameters
  if (!readParameters())
    return false;

  // Initialize subscriber
  key_pose_sub_ = nh_.subscribe<nav_msgs::Odometry>(sub_keypose_topic_, 5, &DualStateGraph::keyposeCallback, this);
  graph_planner_path_sub_ = nh_.subscribe<nav_msgs::Path>(sub_path_topic_, 1, &DualStateGraph::pathCallback, this);
  graph_planner_status_sub_ = nh_.subscribe<graph_planner::GraphPlannerStatus>(
      sub_graph_planner_status_topic_, 1, &DualStateGraph::graphPlannerStatusCallback, this);

  // Initialize publishers
  local_graph_pub_ = nh_.advertise<graph_utils::TopologicalGraph>(pub_local_graph_topic_, 2);
  global_graph_pub_ = nh_.advertise<graph_utils::TopologicalGraph>(pub_global_graph_topic_, 2);
  graph_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(pub_global_points_topic_, 2);

  ROS_INFO("Successfully launched DualStateGraph node");

  return true;
}

bool DualStateGraph::execute()
{
  // Update the graph
  publishLocalGraph();

  return true;
}
}
// namespace dsvplanner_ns
