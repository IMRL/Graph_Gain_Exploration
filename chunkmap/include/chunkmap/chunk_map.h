#ifndef _CHUNK_MAP_CHUNK_MAP_H_
#define _CHUNK_MAP_CHUNK_MAP_H_

#include <memory>
#include <map>
#include <any>
#include <variant>
#include <array>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "aztools/mat.hpp"
#include "LidarIris/LidarIris.h"

struct ChunkData;

namespace ObstacleFeature
{
    enum Segments
    {
        Seg8 = 8,
        Seg32 = 32
    };

    enum Bits
    {
        Default = 0, // used by old format
        Bit1 = 1,
        Bit4 = 4
    };

    using ObstacleDescIndex = std::pair<ObstacleFeature::Segments, ObstacleFeature::Bits>;
    constexpr auto ObstacleDescList = std::array{
        ObstacleDescIndex{ObstacleFeature::Seg8, ObstacleFeature::Default},
        ObstacleDescIndex{ObstacleFeature::Seg8, ObstacleFeature::Bit1},
        ObstacleDescIndex{ObstacleFeature::Seg8, ObstacleFeature::Bit4},
        ObstacleDescIndex{ObstacleFeature::Seg32, ObstacleFeature::Default},
        ObstacleDescIndex{ObstacleFeature::Seg32, ObstacleFeature::Bit1},
        ObstacleDescIndex{ObstacleFeature::Seg32, ObstacleFeature::Bit4},
    };
    constexpr int ObstacleDescListLength = ObstacleDescList.size();
}

class ChunkMap
{
public:
    struct Index
    {
        int64_t x;
        int64_t y;
        bool operator<(const Index &b) const
        {
            return (x < b.x) || (x == b.x && y < b.y);
        }
    };

    struct CellIndex
    {
        Index chunk;
        size_t layer;
        cv::Point cell;
        bool operator==(const CellIndex &v) const { return (chunk.x == v.chunk.x) && (chunk.y == v.chunk.y) && layer == v.layer && (cell == v.cell); }
        bool operator<(const CellIndex &v) const
        {
            if (chunk.x < v.chunk.x) return true;
            else if (chunk.x > v.chunk.x) return false;
            if (chunk.y < v.chunk.y) return true;
            else if (chunk.y > v.chunk.y) return false;
            if (layer < v.layer) return true;
            else if (layer > v.layer) return false;
            if (cell.x < v.cell.x) return true;
            else if (cell.x > v.cell.x) return false;
            if (cell.y < v.cell.y) return true;
            else if (cell.y > v.cell.y) return false;
            return false;
        }
    };

    struct Config
    {
        float resolution;
        float chunk_base;
    };

    struct CompressedDesc
    {
        std::vector<uint8_t> img;
        std::vector<uint8_t> T;
        std::vector<uint8_t> M;
    };

    enum class DescType
    {
        uncompressed,
        compressed
    };

    template <ObstacleFeature::Segments S, ObstacleFeature::Bits B>
    struct ObstacleTypeDesc
    {
        using ElemType = aztools::bitsmat::bitset<S * B, uint8_t>;
        using Type = aztools::bitsmat::Mat_<ElemType>;
        static constexpr size_t serial_id = uint32_t(B) << 16 | uint16_t(S);
        static constexpr size_t max_height = ElemType::MAX_BITS;
        static constexpr auto wrapper = aztools::bitsmat::wrap<ElemType>{};
    };

    struct KeyFrameData
    {
        std::variant<LidarIris::FeatureDesc, CompressedDesc> feature;
        Eigen::Matrix4f pose;
    };

    using Ptr = std::shared_ptr<ChunkMap>;
    ChunkMap(const Config &config);

    // void dirty(const Index &index) { dirty_set.insert(index); }
    // std::set<Index> update() { return std::move(dirty_set); }
    
    virtual ~ChunkMap() {}
    ChunkData &operator[](const Index &index);
    const ChunkData &at(const Index &index) const;
    std::vector<Index> keys() const;
    bool has(const Index &index) const;

    bool query(const Eigen::Vector3f &point, CellIndex &index, float *error = nullptr) const;
    bool query(const CellIndex &index, Eigen::Vector3f &point) const;

    static void load(std::istream &stream, const ChunkMap::Ptr &map);
    static void save(std::ostream &stream, const ChunkMap::Ptr &map);

    std::vector<KeyFrameData> key_frames;
    DescType desc_type = DescType::uncompressed;
    // ObstacleBits obstacle_bits = ObstacleBits::Bit32;
    ObstacleFeature::Segments obstacle_segments = ObstacleFeature::Seg32;
    ObstacleFeature::Bits obstacle_bits = ObstacleFeature::Bit1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr generateObstacleCloud();
    std::pair<int, int> ObstacleCount();

    float resolution() const { return config_.resolution; }
    float chunkBase() const { return config_.chunk_base; }
    uint32_t chunkSize() const { return chunk_size; }

protected:
    Config config_;
    std::map<Index, ChunkData> chunks;
    uint32_t chunk_size;

    // std::set<Index> dirty_set;
};

template <ObstacleFeature::Segments S>
struct ChunkMap::ObstacleTypeDesc<S, ObstacleFeature::Default>
{
    using ElemType = aztools::bitsmat::bitset<S, uint8_t>;
    using Type = aztools::bitsmat::Mat_<ElemType>;
    static constexpr size_t serial_id = uint16_t(S);
    static constexpr size_t max_height = ElemType::MAX_BITS;
    static constexpr auto wrapper = aztools::bitsmat::wrap<ElemType>{};
};

size_t ObstacleTypeDesc_serial_id(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit);
size_t ObstacleTypeDesc_max_height(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit);
cv::Mat ObstacleTypeDesc_create(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit, const cv::Size &size);

uint64_t ObstacleTypeDesc_at(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit, const cv::Mat &mat, int r, int c);

struct ChunkLayer
{
    cv::Mat1b observe;
    cv::Mat1f occupancy;
    cv::Mat1f elevation;
    cv::Mat1f elevation_sigma;
    // cv::Mat_<int32_t> obstacle_info;
    cv::Mat obstacle_info;
    // cv::Mat3b image;
    // cv::Mat1b image_observe;
    // cv::Mat1f image_sigma;
};

class ChunkData
{
public:
    void init(const ChunkMap::Index &index, ChunkMap *map, uint32_t chunk_size);

    // void dirty() { map_->dirty(index_); }

    ChunkLayer createLayer() const;

    const std::vector<ChunkLayer> &getLayers() const { return layers_; }

    void setLayers(std::vector<ChunkLayer> &layers)
    {
        layers_.swap(layers);
        // map_->dirty(index_);
    }

    std::any meta;

private:
    ChunkMap *map_;
    ChunkMap::Index index_;
    uint32_t chunk_size_;
    std::vector<ChunkLayer> layers_;
};

#endif // _CHUNK_MAP_CHUNK_MAP_H_
