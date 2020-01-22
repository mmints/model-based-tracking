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

class ParticleGenerator     // TODO: Combine with ParticleGrid
{
private:
    CVK::Node *m_model = nullptr;
    int m_particle_count;
    int m_particle_width;
    int m_particle_height;
    int m_grid_resolution_width;
    int m_grid_resolution_height;

    void generateLinearDistributedRotationMatrix(glm::vec3 &random_angle);
    void generateGaussianDistributedRotationMatrix(glm::mat4 &rotation_matrix);

public:
    ParticleGenerator(std::string path_to_model, int particle_count, int particle_width, int particle_height);
    ParticleGenerator(int particle_count, int particle_width, int particle_height);

    ParticleGenerator();

    void initializeParticles(std::vector<Particle> &particles, float distribution_radius);

    void renderParticleTextureGrid(std::vector<Particle> &particles);

    void updateParticles(std::vector<Particle> &particles);
    void updateParticles(std::vector<Particle> &particles, float translation_deviation, float rotation_deviation);
};
}

#endif //MT_PARTICLEGENERATOR_H