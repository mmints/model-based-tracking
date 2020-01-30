#ifndef MT_PARTICLEGRID_H
#define MT_PARTICLEGRID_H

#include "ModelTracker.h"

namespace mt
{

class ParticleGrid
{
    std::vector<mt::Particle> particles;
    GLuint texture;
};

}

#endif //MT_PARTICLEGRID_H
