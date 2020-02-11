#include "Particle.h"

mt::Particle::Particle(int width, int height)
{
    m_width = width;
    m_height = height;
    m_weight = 0.f;
    m_translation_vector = glm::vec3(1.f);
    glm::vec3 m_rotation_angles = glm::vec3(0.f);
}

mt::Particle::Particle(int width, int height, float weight, glm::vec3 translation_vector, glm::vec3 rotation_angles) {
    m_width = width;
    m_height = height;
    m_weight = weight;
    m_translation_vector = translation_vector;
    m_rotation_angles = rotation_angles;
}

void mt::Particle::setWeight(float weight) {
    m_weight = weight;
}

float mt::Particle::getWeight() {
    return m_weight;
}

int mt::Particle::getWidth() {
    return m_width;
}

int mt::Particle::getHeight() {
    return m_height;
}

glm::mat4 mt::Particle::getModelMatrix() {
    glm::mat4 rotation_x;
    glm::mat4 rotation_y;
    glm::mat4 rotation_z;

    rotation_x = glm::rotate(glm::mat4(1.f), m_rotation_angles.x, glm::vec3(1, 0, 0));
    rotation_y = glm::rotate(glm::mat4(1.f), m_rotation_angles.y, glm::vec3(0, 1, 0));
    rotation_z = glm::rotate(glm::mat4(1.f), m_rotation_angles.z, glm::vec3(0, 0, 1));

    glm::mat4 rotation_matrix = rotation_x * rotation_y * rotation_z;
    glm::mat4 translation_matrix = glm::translate(glm::mat4(1.f), m_translation_vector);
    return translation_matrix * rotation_matrix * glm::scale(glm::mat4(1.f), glm::vec3(1.2f));
}

glm::vec3 mt::Particle::getTranslation() {
    return m_translation_vector;
}

glm::vec3 mt::Particle::getRotation() {
    return m_rotation_angles;
}

void mt::Particle::setModelMatrix(glm::vec3 translation_vector, glm::vec3 rotation_angles) {
    m_translation_vector = translation_vector;
    m_rotation_angles = rotation_angles;
}
