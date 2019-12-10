#include "ParticleGenerator.h"
#include <iostream>


mt::ParticleGenerator::ParticleGenerator(std::string path_to_model, int particle_count, glm::vec2 frame_resolution) {
    m_model = new CVK::Node("model", path_to_model);
    m_particle_count = particle_count;
    m_frame_resolution = frame_resolution;
}

void mt::ParticleGenerator::generateLinearRotation(glm::mat4 &rotation_matrix) {
    glm::vec3 axis_x = glm::vec3(1, 0, 0);
    glm::vec3 axis_y = glm::vec3(0, 1, 0);
    glm::vec3 axis_z = glm::vec3(0, 0, 1);

    glm::mat4 rotation_x;
    glm::mat4 rotation_y;
    glm::mat4 rotation_z;

    glm::vec3 random_angle;
    glm::vec3 min_angle = glm::vec3(0.f);
    glm::vec3 max_angle = glm::vec3(2 * M_PI);

    random_angle = glm::linearRand(min_angle, max_angle);

    rotation_x = glm::rotate(glm::mat4(1.f), random_angle.x, axis_x);
    rotation_y = glm::rotate(glm::mat4(1.f), random_angle.y, axis_y);
    rotation_z = glm::rotate(glm::mat4(1.f), random_angle.z, axis_z);

    rotation_matrix = rotation_x * rotation_y * rotation_z;
}

void mt::ParticleGenerator::generateDistributedModelMatrices(glm::vec3 &mean_pos, float radius, glm::mat4 &model_matrix) {
    glm::mat4 rotation_matrix;
    glm::vec3 translation_vector = glm::gaussRand(mean_pos, glm::vec3(radius));
    mt::ParticleGenerator::generateLinearRotation(rotation_matrix);
    model_matrix = glm::translate(rotation_matrix, translation_vector);
}

void mt::ParticleGenerator::initializeParticles(std::vector<Particle> &particles) {
    glm::vec3 scene_center = glm::vec3(0.f);
    glm::mat4 model_matrix;

    for (int i = 0; i < m_particle_count; i++)
    {
        mt::ParticleGenerator::generateDistributedModelMatrices(scene_center, 1.f, model_matrix);
        mt::Particle particle(i, 0.f, model_matrix);
        particles.push_back(particle);
    }
}

void mt::ParticleGenerator::renderParticleTextureGrid(int size_x, int size_y, std::vector<Particle> &particles) {
    int vpWidth = m_frame_resolution.x/size_x;
    int vpHeight = m_frame_resolution.y/size_y;

    int i = 0;
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            glViewport(vpWidth * x, vpHeight * y, vpWidth, vpHeight);
            m_model->setModelMatrix(particles[i].getModelMatrix());
            m_model->render();
            i++;
        }
    }
}

void mt::ParticleGenerator::updateParticles(std::vector<Particle> &particles) {
    glm::vec3 translation;
    glm::mat4 model_matrix;

    for (int i = 0; i < m_particle_count; i++)
    {
        translation = particles[i].getModelMatrix()[3];
        mt::ParticleGenerator::generateDistributedModelMatrices(translation, 0.2f, model_matrix);
        particles[i].setModelMatrix(model_matrix);
    }

}
