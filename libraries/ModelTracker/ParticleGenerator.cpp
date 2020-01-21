#include "ParticleGenerator.h"

mt::ParticleGenerator::ParticleGenerator(std::string path_to_model, int particle_count, int particle_width, int particle_height) {
    srand(time(0)); // Set a seed for evey new instantiated ParticleGenerator
    if ((int)std::pow((int)std::sqrt(particle_count), 2) != particle_count) {
        throw std::invalid_argument("[mt::ParticleGenerator::ParticleGenerator] Invalid Particle Count. "
                                    "Make sure that it is a result of a square operation.");
    }

    m_particle_count = particle_count;
    m_model = new CVK::Node("model", path_to_model);

    m_particle_width = particle_width;
    m_particle_height = particle_height;

    int dimension = (int)std::sqrt(m_particle_count);
    m_grid_resolution_width = m_particle_width * dimension;
    m_grid_resolution_height = m_particle_height * dimension;
}

void mt::ParticleGenerator::generateLinearDistributedRotationMatrix(glm::vec3 &random_angle) {
    glm::vec3 min_angle = glm::vec3(0.f);
    glm::vec3 max_angle = glm::vec3(2 * M_PI);
    random_angle = glm::linearRand(min_angle, max_angle);
}

void mt::ParticleGenerator::initializeParticles(std::vector<Particle> &particles, float distribution_radius) {
    glm::vec3 scene_center = glm::vec3(0.f);
    glm::vec3 rotation_angles;
    glm::vec3 translation_vector;

    for (int i = 0; i < m_particle_count; i++)
    {
        mt::ParticleGenerator::generateLinearDistributedRotationMatrix(rotation_angles);
        translation_vector = glm::gaussRand(scene_center, glm::vec3(distribution_radius));
        mt::Particle particle(i, 0.f, translation_vector, rotation_angles);
        particles.push_back(particle);
    }
}

void mt::ParticleGenerator::renderParticleTextureGrid(std::vector<Particle> &particles) {
    int dimension = (int)std::sqrt(m_particle_count);
    int i = 0;
    for (int x = 0; x < dimension; x++) {
        for (int y = 0; y < dimension; y++) {
            glViewport(m_particle_width * x, m_particle_height * y, m_particle_width, m_particle_height);
            m_model->setModelMatrix(particles[i].getModelMatrix());
            m_model->render();
            i++;
        }
    }
}

void mt::ParticleGenerator::updateParticles(std::vector<Particle> &particles) {
    glm::vec3 translation;
    glm::vec3 rotation_angles;

    for (int i = 0; i < m_particle_count; i++)
    {
        // TODO: What is the correct Deviation for the update function?
        translation = glm::gaussRand(particles[i].getTranslation(), glm::vec3(0.8f));
        rotation_angles = glm::gaussRand(particles[i].getRotation(), glm::vec3(0.2f));
        particles[i].setModelMatrix(translation, rotation_angles);
    }
}