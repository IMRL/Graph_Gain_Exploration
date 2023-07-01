#include <chunkmap/chunk_map.h>
#include <chunkmap/simple_probability_values.h>

void ChunkData::init(const ChunkMap::Index &index, ChunkMap *map, uint32_t chunk_size)
{
    index_ = index;
    map_ = map;
    chunk_size_ = chunk_size;
    // cv::Size size{chunk_size, chunk_size};
    // observe = cv::Mat1b::zeros(size);
    // occupancy = cv::Mat1f::ones(size) * 127;
    // elevation = cv::Mat1f::zeros(size);
    // elevation_sigma = -cv::Mat1f::ones(size);
}

ChunkLayer ChunkData::createLayer() const
{
    cv::Size size{chunk_size_, chunk_size_};
    ChunkLayer layer;
    layer.observe = cv::Mat1b::zeros(size);
    layer.occupancy = cv::Mat1f::ones(size) * 127;
    layer.elevation = cv::Mat1f::zeros(size);
    layer.elevation_sigma = -cv::Mat1f::ones(size);
    layer.obstacle_info = ObstacleTypeDesc_create(map_->obstacle_segments, map_->obstacle_bits, size);
    // layer.image = cv::Mat3b::zeros(size);
    // layer.image_observe = cv::Mat1b::zeros(size);
    // layer.image_sigma = -cv::Mat1f::ones(size);
    return layer;
}

ChunkMap::ChunkMap(const Config &config) : config_(config)
{
    chunk_size = std::round(config_.chunk_base / config_.resolution);
}

ChunkData &ChunkMap::operator[](const Index &index)
{
    auto iter = chunks.find(index);
    if (iter != chunks.end())
        return iter->second;
    ChunkData chunk;
    chunk.init(index, this, chunk_size);
    chunks.insert({index, chunk});
    return chunks[index];
}

const ChunkData &ChunkMap::at(const Index &index) const
{
    auto iter = chunks.find(index);
    if (iter != chunks.end())
        return iter->second;
    throw std::out_of_range("");
}

std::vector<ChunkMap::Index> ChunkMap::keys() const
{
    std::vector<Index> result;
    for (const auto &item : chunks)
        result.push_back(item.first);
    return result;
}

bool ChunkMap::has(const ChunkMap::Index &index) const
{
    return chunks.find(index) != chunks.end();
}

bool ChunkMap::query(const Eigen::Vector3f &point, CellIndex &index, float *error) const
{
    int64_t linearX = std::round(point.x() / config_.resolution);
    int64_t linearY = std::round(point.y() / config_.resolution);
    Index chunkIndex{(int64_t)std::floor(linearX / (double)chunk_size), (int64_t)std::floor(linearY / (double)chunk_size)};
    cv::Point cellIndex{int(linearX - chunkIndex.x * chunk_size), int(linearY - chunkIndex.y * chunk_size)};
    if (chunks.find(chunkIndex) == chunks.end())
    {
        return false;
    }
    if (cellIndex.x < 0 || cellIndex.x >= chunk_size || cellIndex.y < 0 || cellIndex.y >= chunk_size)
    {
        return false;
    }
    const auto &layers = chunks.at(chunkIndex).getLayers();
    if (layers.size() == 0)
    {
        return false;
    }
    int minIndex = -1;
    float minDistance = std::numeric_limits<float>::max();
    for (int l = 0; l < layers.size(); l++)
    {
        float distance = std::abs(point.z() - layers[l].elevation(cellIndex));
        if (distance < minDistance)
        {
            minIndex = l;
            minDistance = distance;
        }
    }
    if (minIndex < 0)
    {
        return false;
    }
    index.chunk = chunkIndex;
    index.layer = minIndex;
    index.cell = cellIndex;
    if (error != nullptr)
        *error = point.z() - layers[minIndex].elevation(cellIndex);
    return true;
}

bool ChunkMap::query(const CellIndex &index, Eigen::Vector3f &point) const
{
    if (chunks.find(index.chunk) == chunks.end())
        return false;
    point.x() = (index.chunk.x * chunk_size + index.cell.x) * config_.resolution;
    point.y() = (index.chunk.y * chunk_size + index.cell.y) * config_.resolution;
    point.z() = chunks.at(index.chunk).getLayers()[index.layer].elevation(index.cell);
    return true;
}

template <int N = 0>
size_t ObstacleTypeDesc_serial_id_impl(const ObstacleFeature::ObstacleDescIndex &key)
{
    constexpr auto item = std::get<N>(ObstacleFeature::ObstacleDescList);
    if (key == item)
        return ChunkMap::ObstacleTypeDesc<item.first, item.second>::serial_id;
    else
        return ObstacleTypeDesc_serial_id_impl<N + 1>(key);
}
template <>
size_t ObstacleTypeDesc_serial_id_impl<ObstacleFeature::ObstacleDescListLength>(const ObstacleFeature::ObstacleDescIndex &)
{
    return 0;
}
size_t ObstacleTypeDesc_serial_id(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit)
{
    return ObstacleTypeDesc_serial_id_impl({seg, bit});
}

template <int N = 0>
size_t ObstacleTypeDesc_max_height_impl(const ObstacleFeature::ObstacleDescIndex &key)
{
    constexpr auto item = std::get<N>(ObstacleFeature::ObstacleDescList);
    if (key == item)
        return ChunkMap::ObstacleTypeDesc<item.first, item.second>::max_height;
    else
        return ObstacleTypeDesc_max_height_impl<N + 1>(key);
}
template <>
size_t ObstacleTypeDesc_max_height_impl<ObstacleFeature::ObstacleDescListLength>(const ObstacleFeature::ObstacleDescIndex &)
{
    return 0;
}
size_t ObstacleTypeDesc_max_height(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit)
{
    return ObstacleTypeDesc_max_height_impl({seg, bit});
}

template <int N = 0>
cv::Mat ObstacleTypeDesc_create_impl(const ObstacleFeature::ObstacleDescIndex &key, const cv::Size &size)
{
    constexpr auto item = std::get<N>(ObstacleFeature::ObstacleDescList);
    if (key == item)
        return ChunkMap::ObstacleTypeDesc<item.first, item.second>::Type::zeros(size);
    else
        return ObstacleTypeDesc_create_impl<N + 1>(key, size);
}
template <>
cv::Mat ObstacleTypeDesc_create_impl<ObstacleFeature::ObstacleDescListLength>(const ObstacleFeature::ObstacleDescIndex &, const cv::Size &)
{
    return {};
}
cv::Mat ObstacleTypeDesc_create(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit, const cv::Size &size)
{
    return ObstacleTypeDesc_create_impl({seg, bit}, size);
}

// uint64_t ObstacleTypeDesc_at(const ChunkMap::ObstacleBits &type, const cv::Mat &mat, int r, int c)
// {
//     if (type == ChunkMap::ObstacleBits::Bit8)
//         return mat.at<ChunkMap::ObstacleTypeDesc<ChunkMap::ObstacleBits::Bit8>::ElemType>(r, c);
//     else if (type == ChunkMap::ObstacleBits::Bit32)
//         return mat.at<ChunkMap::ObstacleTypeDesc<ChunkMap::ObstacleBits::Bit32>::ElemType>(r, c);
//     else
//         return 0;
// }

template <int N = 0>
uint64_t ObstacleTypeDesc_at_impl(const ObstacleFeature::ObstacleDescIndex &key, const cv::Mat &mat, int r, int c)
{
#if (__GNUC__ <= 7)
    return 0;
#else
    constexpr auto item = std::get<N>(ObstacleFeature::ObstacleDescList);
    if (key == item)
    {
        using Desc = ChunkMap::ObstacleTypeDesc<item.first, item.second>;
        if constexpr (item.second > ObstacleFeature::Bit1)
        {
            uint64_t ret = 0;
            const uint16_t thres = simple_prob_values<item.second>::init_value;
            const auto elem = Desc::wrapper((typename Desc::Type(mat))(r, c));
            for (int i = item.first - 1; i >= 0; i--)
            {
                ret <<= 1;
                if ((uint16_t)elem(i * item.second, item.second) > thres)
                    ret |= 1;
            }
            return ret;
        }
        else
            return Desc::wrapper((typename Desc::Type(mat))(r, c))(); // TODO: for bits
    }
    else
        return ObstacleTypeDesc_at_impl<N + 1>(key, mat, r, c);
#endif
}
template <>
uint64_t ObstacleTypeDesc_at_impl<ObstacleFeature::ObstacleDescListLength>(const ObstacleFeature::ObstacleDescIndex &, const cv::Mat &, int, int)
{
    return 0;
}
uint64_t ObstacleTypeDesc_at(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit, const cv::Mat &mat, int r, int c)
{
    return ObstacleTypeDesc_at_impl({seg, bit}, mat, r, c);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ChunkMap::generateObstacleCloud()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (auto &chunk : chunks)
    {
        for (auto &layer : chunk.second.getLayers())
            for (int r = 0; r < chunk_size; r++)
                for (int c = 0; c < chunk_size; c++)
                // if (layer.observe(r, c))
                {
                    float x = (chunk.first.x * chunk_size + c) * config_.resolution;
                    float y = (chunk.first.y * chunk_size + r) * config_.resolution;
                    float z = layer.elevation(r, c);
                    // int32_t value = layer.obstacle_info.at<int32_t>(r, c);
                    auto value = ObstacleTypeDesc_at(obstacle_segments, obstacle_bits, layer.obstacle_info, r, c);
                    for (int b = 0; b <= ObstacleTypeDesc_max_height(obstacle_segments, obstacle_bits); b++)
                        if ((value & (1 << b)) != 0)
                        {
                            pcl::PointXYZ p;
                            p.x = x;
                            p.y = y;
                            p.z = z + b; // + 0.5;
                            cloud->push_back(p);
                        }
                }
    }
    return cloud;
}

template <int N = 0>
std::function<uint64_t(const cv::Mat &mat, int r, int c)> ObstacleTypeDesc_at_impl_functor(const ObstacleFeature::ObstacleDescIndex &key)
{
#if (__GNUC__ <= 7)
    return 0;
#else
    constexpr auto item = std::get<N>(ObstacleFeature::ObstacleDescList);
    if (key == item)
    {
        using Desc = ChunkMap::ObstacleTypeDesc<item.first, item.second>;
        if constexpr (item.second > ObstacleFeature::Bit1)
        {
            using SimpleProb = simple_prob_values<item.second>;
            return [&item](const cv::Mat &mat, int r, int c)
            {
                uint64_t ret = 0;
                const uint16_t thres = SimpleProb::init_value;
                const auto elem = Desc::wrapper((typename Desc::Type(mat))(r, c));
                for (int i = item.first - 1; i >= 0; i--)
                {
                    ret <<= 1;
                    if ((uint16_t)elem(i * item.second, item.second) > thres)
                        ret |= 1;
                }
                return ret;
            };
        }
        else
            return [&item](const cv::Mat &mat, int r, int c)
            {
                return Desc::wrapper((typename Desc::Type(mat))(r, c))(); // TODO: for bits
            };
    }
    else
        return ObstacleTypeDesc_at_impl_functor<N + 1>(key);
#endif
}
template <>
std::function<uint64_t(const cv::Mat &mat, int r, int c)> ObstacleTypeDesc_at_impl_functor<ObstacleFeature::ObstacleDescListLength>(const ObstacleFeature::ObstacleDescIndex &)
{
    return [](const cv::Mat &, int, int)
    { return 0; };
}
std::function<uint64_t(const cv::Mat &mat, int r, int c)> ObstacleTypeDesc_at_functor(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit)
{
    return ObstacleTypeDesc_at_impl_functor({seg, bit});
}

std::pair<int, int> ChunkMap::ObstacleCount()
{
    int obs_N = 0;
    int free_N = 0;
    auto functor = ObstacleTypeDesc_at_functor(obstacle_segments, obstacle_bits);
    int max_heigt = ObstacleTypeDesc_max_height(obstacle_segments, obstacle_bits);
    // for (auto &chunk : chunks)
    for (auto &k : keys())
    {
        const auto &chunk = chunks.at(k);
        // for (auto &layer : chunk.second.getLayers())
        for (auto &layer : chunk.getLayers())
            for (int r = 0; r < chunk_size; r++)
                for (int c = 0; c < chunk_size; c++)
                // if (layer.observe(r, c))
                {
                    auto value = functor(layer.obstacle_info, r, c);
                    for (int b = 0; b <= max_heigt; b++)
                        if ((value & (1 << b)) != 0)
                            obs_N += 1;
                        else
                            free_N += 1;
                }
    }
    return {obs_N, free_N};
}
