#ifndef MT_PARTICLEGENERATOR_H
#define MT_PARTICLEGENERATOR_H

#include <vector>
#include <cmath>
#include <GL/glew.h>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include "Particle.h"
#include <CVK_2/CVK_Framework.h> // TODO: Temporary us for getting access to Geometries

namespace mt
{

class ParticleGenerator
{
private:
    CVK::Node *m_model = nullptr;
    int m_particle_count;
    int m_frame_resolution_width;
    int m_frame_resolution_height;

    void generateLinearDistributedRotationMatrix(glm::vec3 &random_angle);
    void generateGaussianDistributedRotationMatrix(glm::mat4 &rotation_matrix);

public:
    ParticleGenerator(std::string path_to_model, int particle_count, int frame_resolution_width, int frame_resolution_height);

    void initializeParticles(std::vector<Particle> &particles, float distribution_radius);

    void renderParticleTextureGrid(std::vector<Particle> &particles);

    void updateParticles(std::vector<Particle> &particles);
};
}

#endif //MT_PARTICLEGENERATOR_H