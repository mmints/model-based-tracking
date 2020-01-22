#ifndef MT_PARTICLEGRID_H
#define MT_PARTICLEGRID_H

#include <vector>
#include "Particle.h"

// TODO: Turn the ParticleGrid Struct into a Class
// TODO: Combine with ParticleGenerator

namespace mt
{
 /**
  * The ParticleGrid is the main container that holds the all
  * data of the particles used for the Tracking process.
  */
struct ParticleGrid
{
    std::vector<mt::Particle> particles;
    GLuint texture;
};

}

#endif //MT_PARTICLEGRID_H
