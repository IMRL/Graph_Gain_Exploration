#ifndef _CHUNK_MAP_CHUNK_VISUAL_H_
#define _CHUNK_MAP_CHUNK_VISUAL_H_

#include <memory>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreEntity.h>
#include <chunkmap/chunk_map.h>

namespace IMRL
{
    class ChunkVisual
    {
    public:
        using Ptr = std::shared_ptr<ChunkVisual>;
        ChunkVisual(Ogre::SceneManager *scene_manager, Ogre::SceneNode *parent, Ogre::MeshPtr &mesh, float resolution, uint32_t chunk_size, const ChunkMap::Index &index, size_t layer_id);
        ~ChunkVisual();

        void updateTexture(const std::vector<uint8_t> &elevation, const std::vector<uint8_t> &occupancy, float elevation_alpha, float elevation_beta);

    private:
        uint32_t chunk_size_;
        Ogre::SceneManager *scene_manager_;
        Ogre::SceneNode *scene_node_;
        Ogre::MaterialPtr material_;
        Ogre::Entity *entity_;
        // Ogre::ManualObject *manual_object_;
        Ogre::TexturePtr elevation_texture_;
        Ogre::TexturePtr occupancy_texture_;
    };
}

#endif // _CHUNK_MAP_CHUNK_VISUAL_H_
