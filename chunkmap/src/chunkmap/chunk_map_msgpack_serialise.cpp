#include <chunkmap/chunk_map.h>
#define MSGPACK_USE_DEFINE_MAP
#include <msgpack.hpp>
#include <algorithm>
#include <fstream>

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
    MSGPACK_DEFINE(observe, occupancy, elevation, elevation_sigma, obstacle_info);//, image, image_observe, image_sigma);
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

// std::map<size_t, ChunkMap::ObstacleBits> obstacle_deserial_map = {
//     {8, ChunkMap::ObstacleBits::Bit8},
//     {32, ChunkMap::ObstacleBits::Bit32},
// };
template <int N = 0>
struct obstacle_deserial_map_generate
{
    template<typename... T>
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
            memcpy(layer.observe.data, layer_bin.observe.data(), layer.observe.total() * sizeof(uint8_t) * layer.observe.channels());
            memcpy(layer.occupancy.data, layer_bin.occupancy.data(), layer.occupancy.total() * sizeof(float) * layer.occupancy.channels());
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
            key_frame.feature = LidarIris::FeatureDesc{};
            std::get<0>(key_frame.feature).img = cv::imdecode(key_frame_bin.img, cv::IMREAD_UNCHANGED);
            std::get<0>(key_frame.feature).T = cv::imdecode(key_frame_bin.T, cv::IMREAD_UNCHANGED);
            std::get<0>(key_frame.feature).M = cv::imdecode(key_frame_bin.M, cv::IMREAD_UNCHANGED);
        }
        else
        {
            key_frame.feature = CompressedDesc{};
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
            layer_helper.observe.reserve(layer.observe.total() * sizeof(uint8_t) * layer.observe.channels());
            layer_helper.observe.assign((uint8_t *)layer.observe.data, (uint8_t *)layer.observe.data + layer.observe.total() * sizeof(uint8_t) * layer.observe.channels());
            layer_helper.occupancy.reserve(layer.occupancy.total() * sizeof(float) * layer.occupancy.channels());
            layer_helper.occupancy.assign((uint8_t *)layer.occupancy.data, (uint8_t *)layer.occupancy.data + layer.occupancy.total() * sizeof(float) * layer.occupancy.channels());
            layer_helper.elevation.reserve(layer.elevation.total() * sizeof(float) * layer.elevation.channels());
            layer_helper.elevation.assign((uint8_t *)layer.elevation.data, (uint8_t *)layer.elevation.data + layer.elevation.total() * sizeof(float) * layer.elevation.channels());
            layer_helper.elevation_sigma.reserve(layer.elevation_sigma.total() * sizeof(float) * layer.elevation_sigma.channels());
            layer_helper.elevation_sigma.assign((uint8_t *)layer.elevation_sigma.data, (uint8_t *)layer.elevation_sigma.data + layer.elevation_sigma.total() * sizeof(float) * layer.elevation_sigma.channels());
            layer_helper.obstacle_info.reserve(layer.obstacle_info.total() * layer.obstacle_info.elemSize());
            layer_helper.obstacle_info.assign((uint8_t *)layer.obstacle_info.data, (uint8_t *)layer.obstacle_info.data + layer.obstacle_info.total() * layer.obstacle_info.elemSize());
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
            cv::imencode(".png", std::get<0>(key_frame.feature).img, key_frame_helper.img);  // iris related fields, for localization
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

    // save to bmp
    // std::vector<int64_t> indice_x, indice_y;
    // for (const auto &chunk : map->chunks)
    // {
    //     indice_x.push_back(chunk.first.x);
    //     indice_y.push_back(chunk.first.y);
    // }
    // std::sort(indice_x.begin(), indice_x.end());
    // std::sort(indice_y.begin(), indice_y.end());
    // double minx = indice_x[0], miny = indice_y[0], maxx = indice_x[indice_x.size()-1], maxy = indice_y[indice_y.size()-1];
    // // int minidxx[2], minidxy[2];
    // // cv::minMaxIdx(indice_x, &minx, &maxx, minidxx, minidxy);
    // // cv::minMaxIdx(indice_y, &miny, &maxy, minidxx, minidxy);
    // int w = maxx - minx+1, h = maxy - miny+1, step = map->chunk_size;
    // cv::Mat1f full_map(h*step, w*step);
    // full_map.setTo(127);
    // std::cout << h << ' ' << w << ' '  << step << std::endl;
    // for (const auto &chunk : map->chunks)
    // {
    //     for (const auto &layer : chunk.second.getLayers())
    //     {   
    //         std::cout << chunk.first.x << ' ' << chunk.first.y << ' ' << layer.occupancy.size() << std::endl;
    //         layer.occupancy.copyTo(full_map(cv::Rect((chunk.first.x-((int)minx)) * step, (chunk.first.y - ((int)miny)) * step, step, step)));
    //     }
    // }
    // // cv::threshold(full_map, full_map, 0.64*255, 255, CV_THRESH_BINARY);
    // std::string sipath("/home/dell/test.bmp");
    // cv::imwrite(sipath, full_map);
    // std::ofstream ofs;
    // ofs.open("/home/dell/test.yml", std::ios::out);
    // map_helper.resolution = map->config_.resolution;
    // map_helper.chunk_base = map->config_.chunk_base;
    // map_helper.chunk_size = map->chunk_size;
    // ofs << "image: " << sipath << std::endl;
    // ofs << "resolution: " << map->config_.resolution << std::endl;
    // char strbuf[996];
    // sprintf(strbuf, "origin: [%f, %f, %f]", minx*step, miny*step, 0.0);
    // ofs << strbuf << std::endl;
    // ofs << "negate: " << 0 << std::endl;
    // ofs << "occupied_thresh: " << 0.5 << std::endl;
    // ofs << "free_thresh: " << 0.3 << std::endl;
    // ofs.close();
}
