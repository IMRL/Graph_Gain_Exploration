#include <chunkmap/chunk_map.h>
#define MSGPACK_USE_DEFINE_MAP
#include <msgpack.hpp>

struct IndexHelper : ChunkMap::Index
{
    MSGPACK_DEFINE(x, y);
};

struct ChunkLayerHelper
{
    std::vector<uint8_t> observe;         // cv::Mat1b
    std::vector<uint8_t> occupancy;       // cv::Mat1f
    std::vector<uint8_t> elevation;       // cv::Mat1f
    std::vector<uint8_t> elevation_sigma; // cv::Mat1f
    std::vector<uint8_t> obstacle_info;   // cv::Mat
    // std::vector<uint8_t> image; // cv::Mat3b
    // std::vector<uint8_t> image_observe; // cv::Mat1b
    // std::vector<uint8_t> image_sigma; // cv::Mat1f
    MSGPACK_DEFINE(observe, occupancy, elevation, elevation_sigma, obstacle_info); //, image, image_observe, image_sigma);
};

struct ChunkHelper
{
    IndexHelper index;
    std::vector<ChunkLayerHelper> layers;
    MSGPACK_DEFINE(index, layers);
};

struct KeyFrameDataHelper
{
    std::vector<uint8_t> img;
    std::vector<uint8_t> T;
    std::vector<uint8_t> M;
    std::vector<float> pose;
    MSGPACK_DEFINE(img, T, M, pose);
};

struct ChunkMapHelper
{
    float resolution;
    float chunk_base;
    uint32_t chunk_size;
    size_t obstacle_bits;
    std::vector<ChunkHelper> chunks;
    std::vector<KeyFrameDataHelper> key_frames;
    MSGPACK_DEFINE(resolution, chunk_base, chunk_size, obstacle_bits, chunks, key_frames);
};

template<typename T>
void binaryEncode(const cv::Mat_<T> &input, std::vector<uint8_t> &output, T thres)
{
    size_t N = (input.rows * input.cols + 7) >> 3;
    output.reserve(N);
    for (int i = 0; i < N; i++)
        output[N] = 0;
    int n = 0;
    for (int r = 0; r < input.rows; r++)
    {
        for (int c = 0; c < input.cols; c++)
        {
            if (input(r, c) >= thres)
                output[n >> 3] |= 1 << (n & 7);
            n += 1;
        }
    }
}

template<typename T>
void binaryDecode(const std::vector<uint8_t> &input, cv::Mat_<T> &output)
{
    size_t N = (output.rows * output.cols + 7) >> 3;
    int n = 0;
    for (int r = 0; r < output.rows; r++)
    {
        for (int c = 0; c < output.cols; c++)
        {
            output = input[n >> 3] & (1 << (n & 7)) ? 255 : 0;
            n += 1;
        }
    }
}

// std::map<size_t, ChunkMap::ObstacleBits> obstacle_deserial_map = {
//     {8, ChunkMap::ObstacleBits::Bit8},
//     {32, ChunkMap::ObstacleBits::Bit32},
// };
template <int N = 0>
struct obstacle_deserial_map_generate
{
    template <typename... T>
    std::map<size_t, ObstacleFeature::ObstacleDescIndex> operator()(T &&...pre)
    {
        constexpr auto item = std::get<N>(ObstacleFeature::ObstacleDescList);
        using Desc = ChunkMap::ObstacleTypeDesc<item.first, item.second>;
        return obstacle_deserial_map_generate<N + 1>{}(std::move(pre)..., std::pair<size_t, ObstacleFeature::ObstacleDescIndex>{Desc::serial_id, item});
    }
};
template <>
struct obstacle_deserial_map_generate<ObstacleFeature::ObstacleDescListLength>
{
    template <typename... T>
    std::map<size_t, ObstacleFeature::ObstacleDescIndex> operator()(T &&...pre)
    {
        return {pre...};
    }
};
auto obstacle_deserial_map = obstacle_deserial_map_generate<0>{}();

void ChunkMap::load(std::istream &stream, const ChunkMap::Ptr &map)
{
    ChunkMapHelper map_helper;
    {
        static const size_t try_read_size = 1024;
        msgpack::unpacker unp;
        while (true)
        {
            unp.reserve_buffer(try_read_size);
            size_t actual_read_size = stream.readsome(unp.buffer(), try_read_size);
            unp.buffer_consumed(actual_read_size);
            msgpack::object_handle handle;
            if (unp.next(handle))
            {
                map_helper = msgpack::object(handle.get()).as<ChunkMapHelper>();
                break;
            }
        }
    }
    map->config_.resolution = map_helper.resolution;
    map->config_.chunk_base = map_helper.chunk_base;
    map->chunk_size = map_helper.chunk_size;
    // map->obstacle_bits = obstacle_deserial_map[map_helper.obstacle_bits];
    std::tie(map->obstacle_segments, map->obstacle_bits) = obstacle_deserial_map[map_helper.obstacle_bits];
    for (const auto &chunk : map_helper.chunks)
    {
        ChunkMap::Index index;
        index.x = chunk.index.x;
        index.y = chunk.index.y;
        std::vector<ChunkLayer> layers;
        for (const auto &layer_bin : chunk.layers)
        {
            ChunkLayer layer = (*map)[index].createLayer();
            binaryDecode(layer_bin.observe, layer.observe);
            binaryDecode(layer_bin.occupancy, layer.occupancy);
            // memcpy(layer.observe.data, layer_bin.observe.data(), layer.observe.total() * sizeof(uint8_t) * layer.observe.channels());
            // memcpy(layer.occupancy.data, layer_bin.occupancy.data(), layer.occupancy.total() * sizeof(float) * layer.occupancy.channels());
            memcpy(layer.elevation.data, layer_bin.elevation.data(), layer.elevation.total() * sizeof(float) * layer.elevation.channels());
            memcpy(layer.elevation_sigma.data, layer_bin.elevation_sigma.data(), layer.elevation_sigma.total() * sizeof(float) * layer.elevation_sigma.channels());
            memcpy(layer.obstacle_info.data, layer_bin.obstacle_info.data(), layer.obstacle_info.total() * layer.obstacle_info.elemSize());

            // memcpy(layer.image.data, layer_bin.image.data(), layer.image.total() * sizeof(uint8_t) * layer.image.channels());
            // memcpy(layer.image_observe.data, layer_bin.image_observe.data(), layer.image_observe.total() * sizeof(uint8_t) * layer.image_observe.channels());
            // memcpy(layer.image_sigma.data, layer_bin.image_sigma.data(), layer.image_sigma.total() * sizeof(float) * layer.image_sigma.channels());
            layers.push_back(layer);
        }
        (*map)[index].setLayers(layers);
    }
    for (const auto &key_frame_bin : map_helper.key_frames)
    {
        KeyFrameData key_frame;
        if (map->desc_type == DescType::uncompressed)
        {
            std::get<0>(key_frame.feature).img = cv::imdecode(key_frame_bin.img, cv::IMREAD_UNCHANGED);
            std::get<0>(key_frame.feature).T = cv::imdecode(key_frame_bin.T, cv::IMREAD_UNCHANGED);
            std::get<0>(key_frame.feature).M = cv::imdecode(key_frame_bin.M, cv::IMREAD_UNCHANGED);
        }
        else
        {
            std::get<1>(key_frame.feature).img = key_frame_bin.img;
            std::get<1>(key_frame.feature).T = key_frame_bin.T;
            std::get<1>(key_frame.feature).M = key_frame_bin.M;
        }
        key_frame.pose = Eigen::Matrix4f::Identity();
        key_frame.pose(0, 0) = key_frame_bin.pose[0];
        key_frame.pose(0, 1) = key_frame_bin.pose[1];
        key_frame.pose(0, 2) = key_frame_bin.pose[2];
        key_frame.pose(0, 3) = key_frame_bin.pose[3];
        key_frame.pose(1, 0) = key_frame_bin.pose[4];
        key_frame.pose(1, 1) = key_frame_bin.pose[5];
        key_frame.pose(1, 2) = key_frame_bin.pose[6];
        key_frame.pose(1, 3) = key_frame_bin.pose[7];
        key_frame.pose(2, 0) = key_frame_bin.pose[8];
        key_frame.pose(2, 1) = key_frame_bin.pose[9];
        key_frame.pose(2, 2) = key_frame_bin.pose[10];
        key_frame.pose(2, 3) = key_frame_bin.pose[11];
        map->key_frames.push_back(key_frame);
    }
}

void ChunkMap::save(std::ostream &stream, const ChunkMap::Ptr &map)
{
    ChunkMapHelper map_helper;
    map_helper.resolution = map->config_.resolution;
    map_helper.chunk_base = map->config_.chunk_base;
    map_helper.chunk_size = map->chunk_size;
    // if(map->obstacle_bits == ObstacleBits::Bit8)
    // {
    //     map_helper.obstacle_bits = ObstacleTypeDesc<ObstacleBits::Bit8>::serial_id;
    // }
    // else if(map->obstacle_bits == ObstacleBits::Bit32)
    // {
    //     map_helper.obstacle_bits = ObstacleTypeDesc<ObstacleBits::Bit32>::serial_id;
    // }
    map_helper.obstacle_bits = ObstacleTypeDesc_serial_id(map->obstacle_segments, map->obstacle_bits);
    for (const auto &chunk : map->chunks)
    {
        ChunkHelper chunk_helper;
        chunk_helper.index.x = chunk.first.x;
        chunk_helper.index.y = chunk.first.y;
        for (const auto &layer : chunk.second.getLayers())
        {
            ChunkLayerHelper layer_helper;
            // layer_helper.observe.reserve(layer.observe.total() * sizeof(uint8_t) * layer.observe.channels());
            // layer_helper.observe.assign((uint8_t *)layer.observe.data, (uint8_t *)layer.observe.data + layer.observe.total() * sizeof(uint8_t) * layer.observe.channels());
            // layer_helper.occupancy.reserve(layer.occupancy.total() * sizeof(float) * layer.occupancy.channels());
            // layer_helper.occupancy.assign((uint8_t *)layer.occupancy.data, (uint8_t *)layer.occupancy.data + layer.occupancy.total() * sizeof(float) * layer.occupancy.channels());
            layer_helper.elevation.reserve(layer.elevation.total() * sizeof(float) * layer.elevation.channels());
            layer_helper.elevation.assign((uint8_t *)layer.elevation.data, (uint8_t *)layer.elevation.data + layer.elevation.total() * sizeof(float) * layer.elevation.channels());
            layer_helper.elevation_sigma.reserve(layer.elevation_sigma.total() * sizeof(float) * layer.elevation_sigma.channels());
            layer_helper.elevation_sigma.assign((uint8_t *)layer.elevation_sigma.data, (uint8_t *)layer.elevation_sigma.data + layer.elevation_sigma.total() * sizeof(float) * layer.elevation_sigma.channels());
            layer_helper.obstacle_info.reserve(layer.obstacle_info.total() * layer.obstacle_info.elemSize());
            layer_helper.obstacle_info.assign((uint8_t *)layer.obstacle_info.data, (uint8_t *)layer.obstacle_info.data + layer.obstacle_info.total() * layer.obstacle_info.elemSize());

            //
            binaryEncode<uint8_t>(layer.observe, layer_helper.observe, 255);
            binaryEncode<float>(layer.occupancy, layer_helper.occupancy, 128.0f);
            // cam-layer
            // layer_helper.image.reserve(layer.image.total() * layer.image.elemSize());
            // layer_helper.image.assign((uint8_t *)layer.image.data, (uint8_t *)layer.image.data + layer.image.total() * layer.image.elemSize());
            // layer_helper.image_observe.reserve(layer.image_observe.total() * layer.image_observe.elemSize());
            // layer_helper.image_observe.assign((uint8_t *)layer.image_observe.data, (uint8_t *)layer.image_observe.data + layer.image_observe.total() * layer.image_observe.elemSize());
            // layer_helper.image_sigma.reserve(layer.image_sigma.total() * layer.image_sigma.elemSize());
            // layer_helper.image_sigma.assign((uint8_t *)layer.image_sigma.data, (uint8_t *)layer.image_sigma.data + layer.image_sigma.total() * layer.image_sigma.elemSize());
            chunk_helper.layers.push_back(layer_helper);
        }
        map_helper.chunks.push_back(chunk_helper);
    }
    for (const auto &key_frame : map->key_frames)
    {
        KeyFrameDataHelper key_frame_helper;
        if (map->desc_type == DescType::uncompressed)
        {
            cv::imencode(".png", std::get<0>(key_frame.feature).img, key_frame_helper.img);
            cv::imencode(".png", std::get<0>(key_frame.feature).T, key_frame_helper.T);
            cv::imencode(".png", std::get<0>(key_frame.feature).M, key_frame_helper.M);
        }
        else
        {
            key_frame_helper.img = std::get<1>(key_frame.feature).img;
            key_frame_helper.T = std::get<1>(key_frame.feature).T;
            key_frame_helper.M = std::get<1>(key_frame.feature).M;
        }
        key_frame_helper.pose.push_back(key_frame.pose(0, 0));
        key_frame_helper.pose.push_back(key_frame.pose(0, 1));
        key_frame_helper.pose.push_back(key_frame.pose(0, 2));
        key_frame_helper.pose.push_back(key_frame.pose(0, 3));
        key_frame_helper.pose.push_back(key_frame.pose(1, 0));
        key_frame_helper.pose.push_back(key_frame.pose(1, 1));
        key_frame_helper.pose.push_back(key_frame.pose(1, 2));
        key_frame_helper.pose.push_back(key_frame.pose(1, 3));
        key_frame_helper.pose.push_back(key_frame.pose(2, 0));
        key_frame_helper.pose.push_back(key_frame.pose(2, 1));
        key_frame_helper.pose.push_back(key_frame.pose(2, 2));
        key_frame_helper.pose.push_back(key_frame.pose(2, 3));
        map_helper.key_frames.push_back(key_frame_helper);
    }
    msgpack::pack(stream, map_helper);
}
