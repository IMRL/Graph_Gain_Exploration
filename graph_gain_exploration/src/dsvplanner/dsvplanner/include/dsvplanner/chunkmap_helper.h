#include <optional>
#include "chunkmap/chunk_map.h"

enum class ChunkCellStatus
{
  UNKNOWN,
  FREE,
  OCCUPIED
};

using StateVec = Eigen::Vector3d;

ChunkCellStatus ChunkmapGetCellStatusPoint(const ChunkMap::Ptr &chunk_map, const StateVec &node);

ChunkCellStatus ChunkmapGetCellStatusPoint(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &index);

Eigen::Vector2i ChunkmapCellIndexDiff(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &a, const ChunkMap::CellIndex &b);

std::optional<ChunkMap::CellIndex> ChunkmapCellIndexMove(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &base, Eigen::Vector2i &direct);

std::vector<ChunkMap::CellIndex> ChunkmapRayCast(const ChunkMap::Ptr &chunk_map, const ChunkMap::CellIndex &origin, const ChunkMap::CellIndex &goal);

ChunkCellStatus ChunkmapGetVisibility(const ChunkMap::Ptr &chunk_map, const StateVec &view_point, const StateVec &voxel_to_test, bool stop_at_unknown_cell);

ChunkCellStatus ChunkmapGetLineStatus(const ChunkMap::Ptr &chunk_map, const StateVec &start, const StateVec &finish);

ChunkCellStatus ChunkmapGetLineStatusBoundingBox(const ChunkMap::Ptr &chunk_map, const StateVec& start, const StateVec& finish,
                                                                const StateVec& bounding_box_size);
