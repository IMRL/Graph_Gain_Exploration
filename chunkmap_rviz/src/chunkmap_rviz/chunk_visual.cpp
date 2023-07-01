#include "chunk_visual.h"
#include <OGRE/OgreHardwarePixelBuffer.h>

namespace IMRL
{
    ChunkVisual::ChunkVisual(Ogre::SceneManager *scene_manager, Ogre::SceneNode *parent, Ogre::MeshPtr &mesh, float resolution, uint32_t chunk_size, const ChunkMap::Index &index, size_t layer_id) : chunk_size_(chunk_size)
    {
        std::string index_name = "(" + std::to_string(index.x) + "," + std::to_string(index.y) + ":" + std::to_string(layer_id) + ")";

        float base_offset = resolution / 2;
        float chunk_base = chunk_size * resolution;

        scene_manager_ = scene_manager;

        material_ = Ogre::MaterialManager::getSingleton().getByName("chunk_map/ChunkCell");
        material_ = material_->clone("ChunkMtrl" + index_name);

        material_->setReceiveShadows(false);
        material_->getTechnique(0)->setLightingEnabled(false);
        material_->setDepthBias(0.0f, 0.0f);
        material_->setCullingMode(Ogre::CULL_NONE);
        material_->setDepthWriteEnabled(true);
        // material_->setSeparateSceneBlending(Ogre::SceneBlendFactor::SBF_SOURCE_ALPHA, Ogre::SceneBlendFactor::SBF_ONE_MINUS_SOURCE_ALPHA, Ogre::SceneBlendFactor::SBF_ONE, Ogre::SceneBlendFactor::SBF_ZERO);

        // manual_object_ = scene_manager_->createManualObject("ChunkManual" + index_name);

        scene_node_ = parent->createChildSceneNode("ChunkNode" + index_name);
        // scene_node_->attachObject(manual_object_);
        entity_ = scene_manager_->createEntity(mesh);
        entity_->setMaterial(material_);
        scene_node_->attachObject(entity_);

        // manual_object_->begin(material_->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);
        // for (int y = 0; y < chunk_size; y++)
        // {
        //     for (int x = 0; x < chunk_size; x++)
        //     {
        //         manual_object_->position((x * resolution) - base_offset, (y * resolution) - base_offset, 0.0f);
        //         manual_object_->textureCoord(x / (float)chunk_size, y / (float)chunk_size);
        //         manual_object_->normal(0.0f, 0.0f, 1.0f);

        //         manual_object_->position((x * resolution) + base_offset, (y * resolution) - base_offset, 0.0f);
        //         manual_object_->textureCoord(x / (float)chunk_size, y / (float)chunk_size);
        //         manual_object_->normal(0.0f, 0.0f, 1.0f);

        //         manual_object_->position((x * resolution) - base_offset, (y * resolution) + base_offset, 0.0f);
        //         manual_object_->textureCoord(x / (float)chunk_size, y / (float)chunk_size);
        //         manual_object_->normal(0.0f, 0.0f, 1.0f);

        //         manual_object_->position((x * resolution) + base_offset, (y * resolution) + base_offset, 0.0f);
        //         manual_object_->textureCoord(x / (float)chunk_size, y / (float)chunk_size);
        //         manual_object_->normal(0.0f, 0.0f, 1.0f);

        //         int id = (y * chunk_size + x) * 4;
        //         manual_object_->quad(id, id + 1, id + 3, id + 2);
        //     }
        // }
        // manual_object_->end();

        scene_node_->setPosition(Ogre::Vector3(index.x * chunk_base, index.y * chunk_base, 0));

        elevation_texture_ = Ogre::TextureManager::getSingleton().createManual(
            "ChunkElevationTex" + index_name, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
            Ogre::TEX_TYPE_2D, chunk_size, chunk_size, 0, Ogre::PF_BYTE_L);
        occupancy_texture_ = Ogre::TextureManager::getSingleton().createManual(
            "ChunkOccupancyTex" + index_name, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
            Ogre::TEX_TYPE_2D, chunk_size, chunk_size, 0, Ogre::PF_BYTE_LA);

        Ogre::Pass *const pass = material_->getTechnique(0)->getPass(0);
        // pass->setSceneBlending(Ogre::SBF_SOURCE_ALPHA, Ogre::SBF_ONE_MINUS_SOURCE_ALPHA);
        // pass->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);

        Ogre::TextureUnitState *const elevation_texture_unit = pass->createTextureUnitState();
        elevation_texture_unit->setTextureName(elevation_texture_->getName());
        elevation_texture_unit->setTextureFiltering(Ogre::TFO_NONE);
        Ogre::TextureUnitState *const occupancy_texture_unit = pass->createTextureUnitState();
        occupancy_texture_unit->setTextureName(occupancy_texture_->getName());
        occupancy_texture_unit->setTextureFiltering(Ogre::TFO_NONE);
    }

    ChunkVisual::~ChunkVisual()
    {
        scene_manager_->destroySceneNode(scene_node_);
        scene_manager_->destroyEntity(entity_);
        // scene_manager_->destroyManualObject(manual_object_);
        Ogre::MaterialManager::getSingleton().remove(material_->getHandle());
        Ogre::TextureManager::getSingleton().remove(elevation_texture_->getHandle());
        Ogre::TextureManager::getSingleton().remove(occupancy_texture_->getHandle());
    }

    void ChunkVisual::updateTexture(const std::vector<uint8_t> &elevation, const std::vector<uint8_t> &occupancy, float elevation_alpha, float elevation_beta)
    {
        // Ogre::DataStreamPtr pixel_stream_elevation;
        // pixel_stream_elevation.bind(new Ogre::MemoryDataStream((void *)elevation.data(), elevation.size() * sizeof(uint8_t)));
        // elevation_texture_->unload();
        // elevation_texture_->loadRawData(pixel_stream_elevation, chunk_size_, chunk_size_, Ogre::PF_BYTE_L);

        // Ogre::DataStreamPtr pixel_stream_occupancy;
        // pixel_stream_occupancy.bind(new Ogre::MemoryDataStream((void *)occupancy.data(), occupancy.size() * sizeof(uint8_t)));
        // occupancy_texture_->unload();
        // occupancy_texture_->loadRawData(pixel_stream_occupancy, chunk_size_, chunk_size_, Ogre::PF_BYTE_LA);

        size_t img_size = chunk_size_ * chunk_size_;

        Ogre::HardwarePixelBufferSharedPtr elevation_buffer = elevation_texture_->getBuffer();
        elevation_buffer->lock(Ogre::Image::Box(0, 0, chunk_size_, chunk_size_), Ogre::HardwareBuffer::HBL_DISCARD);
        const Ogre::PixelBox &elevation_pb = elevation_buffer->getCurrentLock();
        uint8_t *elevation_data = static_cast<uint8_t *>(elevation_pb.data);
        const uint8_t *elevation_img = reinterpret_cast<const uint8_t *>(elevation.data());
        for (int i = 0; i < img_size; i++)
            elevation_data[i] = elevation_img[i];
        elevation_buffer->unlock();

        Ogre::HardwarePixelBufferSharedPtr occupancy_buffer = occupancy_texture_->getBuffer();
        occupancy_buffer->lock(Ogre::Image::Box(0, 0, chunk_size_, chunk_size_), Ogre::HardwareBuffer::HBL_DISCARD);
        const Ogre::PixelBox &occupancy_pb = occupancy_buffer->getCurrentLock();
        uint16_t *occupancy_data = static_cast<uint16_t *>(occupancy_pb.data);
        const uint16_t *occupancy_img = reinterpret_cast<const uint16_t *>(occupancy.data());
        for (int i = 0; i < img_size; i++)
        {
            occupancy_data[i] = occupancy_img[i];
            // if ((occupancy_img[i] & 0xff) < 126)
            //     occupancy_data[i] = occupancy_img[i] & 0xff00;
            // else if ((occupancy_img[i] & 0xff) > 130)
            //     occupancy_data[i] = occupancy_img[i] | 0x00ff;
            // else
            //     occupancy_data[i] = occupancy_img[i] & 0xff00 | 0x0080;
        }
        occupancy_buffer->unlock();

        const Ogre::GpuProgramParametersSharedPtr parameters = material_->getTechnique(0)->getPass(0)->getVertexProgramParameters();
        parameters->setNamedConstant("u_elevation_alpha", elevation_alpha);
        parameters->setNamedConstant("u_elevation_beta", elevation_beta);
    }
}
