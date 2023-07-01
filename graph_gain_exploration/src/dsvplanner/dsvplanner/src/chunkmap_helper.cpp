#include "dsvplanner/chunkmap_helper.h"

ChunkCellStatus ChunkmapGetCellStatusPoint(const ChunkMap::Ptr &chunk_map, const StateVec &node)
{
  Eigen::Vector3f point = node.cast<float>();
  ChunkMap::CellIndex index;
  if (!chunk_map->query(point, index))
  {
    return ChunkCellStatus::UNKNOWN;
  }
  return ChunkmapGetCellStatusPoint(chunk_map, index);
}

ChunkCellStatus ChunkmapGetCellStatusPoint(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &index)
{
  const auto &layer = chunk_map->at(index.chunk).getLayers()[index.layer];
  if (layer.observe(index.cell) != 255)
    return ChunkCellStatus::UNKNOWN;
  if (layer.occupancy(index.cell) <= 127)
    return ChunkCellStatus::OCCUPIED;
  if (layer.occupancy(index.cell) >= 130)
    return ChunkCellStatus::FREE;
  else
    return ChunkCellStatus::UNKNOWN;
}

Eigen::Vector2i ChunkmapCellIndexDiff(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &a, const ChunkMap::CellIndex &b)
{
  return {(a.chunk.x - b.chunk.x)* chunk_map->chunkSize() + a.cell.x - b.cell.x, (a.chunk.y - b.chunk.y)* chunk_map->chunkSize() + a.cell.y - b.cell.y};
}

std::optional<ChunkMap::CellIndex> ChunkmapCellIndexMove(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &base, const Eigen::Vector2i &direct)
{
  // TODO: height limit
  auto resolution = chunk_map->resolution();
  // if(direct.x() != 0 && direct.y() != 0 || direct == Eigen::Vector2i{0,0})
  // {
  //   return {};
  // }
  // Eigen::Vector3f offset{direct.x() / std::abs(direct.x()) * resolution, direct.y() / std::abs(direct.y()) * resolution, 0};
  Eigen::Vector3f offset{direct.x() * resolution, direct.y() * resolution, 0};
  Eigen::Vector3f baseF;
  chunk_map->query(base, baseF);
  baseF += offset;
  ChunkMap::CellIndex next;
  if (!chunk_map->query(baseF, next))
    return {};
  return next;
}

int signum(int x)
{
  return x == 0 ? 0 : x < 0 ? -1 : 1;
}

double mod(double value, double modulus)
{
  return std::fmod(std::fmod(value, modulus) + modulus, modulus);
}

double intbound(double s, double ds)
{
  // Find the smallest positive t such that s+t*ds is an integer.
  if (ds < 0)
  {
    return intbound(-s, -ds);
  }
  else
  {
    s = mod(s, 1);
    // problem is now s+t*ds = 1
    return (1 - s) / ds;
  }
}

std::vector<ChunkMap::CellIndex> ChunkmapRayCast(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &origin, const ChunkMap::CellIndex &goal)
{
  std::vector<ChunkMap::CellIndex> grid_pairs;
  if (origin == goal)
  {
    grid_pairs.push_back(origin);
    return grid_pairs;
  }

  Eigen::Vector3f originE;
  chunk_map->query(origin, originE);

  auto diff = ChunkmapCellIndexDiff(chunk_map, goal, origin);

  double max_dist = diff.cast<float>().norm();

  int step_x = signum(diff.x());
  int step_y = signum(diff.y());
  double t_max_x = step_x == 0 ? DBL_MAX : intbound(originE.x(), diff.x());
  double t_max_y = step_y == 0 ? DBL_MAX : intbound(originE.y(), diff.y());
  double t_delta_x = step_x == 0 ? DBL_MAX : (double)step_x / (double)diff.x();
  double t_delta_y = step_y == 0 ? DBL_MAX : (double)step_y / (double)diff.y();
  double dist = 0;
  ChunkMap::CellIndex cur_sub = origin;

  while (true)
  {
    grid_pairs.push_back(cur_sub);
    dist = ChunkmapCellIndexDiff(chunk_map, cur_sub, origin).cast<float>().norm();
    if (cur_sub == goal || dist > max_dist)
    {
      return grid_pairs;
    }
    if (t_max_x < t_max_y)
    {
      auto local = ChunkmapCellIndexMove(chunk_map, cur_sub, {step_x , 0});
      if (!local.has_value())
        return grid_pairs;
      cur_sub = *local;
      t_max_x += t_delta_x;
    }
    else
    {
      auto local = ChunkmapCellIndexMove(chunk_map, cur_sub, {0, step_y});
      if (!local.has_value())
        return grid_pairs;
      cur_sub = *local;
      t_max_y += t_delta_y;
    }
  }
}

ChunkCellStatus ChunkmapGetVisibility(const ChunkMap::Ptr &chunk_map, const StateVec &view_point, const StateVec &voxel_to_test, bool stop_at_unknown_cell)
{
  ChunkMap::CellIndex begin, end;
  if (!chunk_map->query(view_point.cast<float>(), begin) || !chunk_map->query(voxel_to_test.cast<float>(), end))
  {
    return ChunkCellStatus::UNKNOWN;
  }
  auto ray = ChunkmapRayCast(chunk_map, begin, end);
  for (const auto &pt : ray)
  {
    if (pt == end)
      continue;
    auto status = ChunkmapGetCellStatusPoint(chunk_map, pt);
    if (status == ChunkCellStatus::OCCUPIED)
      return ChunkCellStatus::OCCUPIED;
    if (stop_at_unknown_cell && status == ChunkCellStatus::UNKNOWN)
      return ChunkCellStatus::UNKNOWN;
  }
  return ChunkCellStatus::FREE;
}

ChunkCellStatus ChunkmapGetLineStatus(const ChunkMap::Ptr &chunk_map, const StateVec &start, const StateVec &finish)
{
  ChunkMap::CellIndex begin, end;
  if (!chunk_map->query(start.cast<float>(), begin) || !chunk_map->query(finish.cast<float>(), end))
  {
    return ChunkCellStatus::UNKNOWN;
  }
  auto ray = ChunkmapRayCast(chunk_map, begin, end);
  for (const auto &pt : ray)
  {
    auto status = ChunkmapGetCellStatusPoint(chunk_map, pt);
    if (status != ChunkCellStatus::FREE)
      return status;
  }
  return ChunkCellStatus::FREE;
}

ChunkCellStatus ChunkmapGetLineStatusBoundingBox(const ChunkMap::Ptr &chunk_map, const StateVec& start, const StateVec& finish,
                                                                const StateVec& bounding_box_size)
{
  const double resolution = chunk_map->resolution();

  int x_disc = bounding_box_size.x() / resolution;
  int y_disc = bounding_box_size.y() / resolution;
  int z_disc = bounding_box_size.z() / resolution;

  ChunkMap::CellIndex begin, end;
  if (!chunk_map->query(start.cast<float>(), begin) || !chunk_map->query(finish.cast<float>(), end))
  {
    return ChunkCellStatus::UNKNOWN;
  }
  auto ray = ChunkmapRayCast(chunk_map, begin, end);
  std::set<ChunkMap::CellIndex> cellSet;
  for (const auto & pt : ray)
  {
    for (int x = -x_disc; x <= x_disc; x += 1)
      for (int y = -y_disc; y <= y_disc; y += 1)
      // for (int z = -z_disc; z<= z_disc; z += 1)
      {
        auto next = ChunkmapCellIndexMove(chunk_map, pt, {x, y});
        if (!next.has_value())
          return ChunkCellStatus::UNKNOWN;
        cellSet.insert(next.value());
      }
  }
  for (const auto &pt : cellSet)
  {
    auto status = ChunkmapGetCellStatusPoint(chunk_map, pt);
    if (status != ChunkCellStatus::FREE)
      return status;
  }
  return ChunkCellStatus::FREE;
}
