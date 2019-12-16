#include "Particle.h"

mt::Particle::Particle(int index)
{
    m_index = index;
    m_weight = 0.f;
    m_translation_vector = glm::vec3(1.f);
    m_rotation_matrix = glm::mat4(1.f);
}

mt::Particle::Particle(int index, float weight, glm::vec3 translation_vector, glm::mat4 rotation_matrix) {
    m_index = index;
    m_weight = weight;
    m_translation_vector = m_translation_vector;
    m_rotation_matrix = rotation_matrix;
}

mt::Particle::~Particle() { }

float mt::Particle::getWeight() {
    return m_weight;
}

int mt::Particle::getIndex() {
    return m_index;
}

glm::mat4 mt::Particle::getModelMatrix() {
    return glm::translate(m_rotation_matrix, m_translation_vector);
}