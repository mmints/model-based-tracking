#ifndef CVK_2_PARTICLEGENERATOR_H
#define CVK_2_PARTICLEGENERATOR_H

#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>

#include "Particle.h"
#include <CVK_2/CVK_Framework.h> // TODO: Temporary us for getting access to Geometries

namespace mt
{

class ParticleGenerator
{
private:
    CVK::Geometry m_geometry; // Select the geometry of the model that should be tracked
    int m_particleTextureHeight;
    int m_particleTextureWidth;
    int m_particleCount;

public:
    ParticleGenerator(CVK::Geometry &geometry, int particleTextureHeight, int particleTextureWidth);

    void initializeParticles(std::vector<Particle> &particles);

    void renderParticleTexture(std::vector<Particle> &particles, int resolutionX, int resolutionY,  GLuint shaderID);
};
}



#endif //CVK_2_PARTICLEGENERATOR_H
