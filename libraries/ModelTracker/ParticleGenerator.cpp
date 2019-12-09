#include "ParticleGenerator.h"
#include <iostream>

mt::ParticleGenerator::ParticleGenerator(CVK::Geometry &geometry, int particleTextureHeight, int particleTextureWidth) {
    m_geometry = geometry;
    m_particleTextureHeight = particleTextureHeight;
    m_particleTextureWidth = particleTextureWidth;
    m_particleCount = particleTextureHeight * particleTextureWidth;
}

void mt::ParticleGenerator::initializeParticles(std::vector<Particle> &particles) {
    // TODO: Distribute Particles around an initial position
    for (int i = 0; i < m_particleCount; i++)
    {
        mt::Particle particle(i, 0.f, glm::mat4(1.f));
        particles.push_back(particle);
    }
}

void mt::ParticleGenerator::renderParticleTexture(std::vector<Particle> &particles, int resolutionX, int resolutionY, GLuint shaderID) {
    //TODO: Implement
    GLuint modelMatrixHandle = glGetUniformLocation(shaderID, "modelMatrix");
    int particleIdx = 0;
    for (int y = 0; y < m_particleTextureHeight; y++)
    {
        for (int x = 0; x < m_particleTextureWidth; x++)
        {
            glViewport(resolutionX * m_particleTextureWidth, resolutionY * m_particleTextureHeight, resolutionX, resolutionY);
        }
    }
}