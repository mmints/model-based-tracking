#include "Particle.h"

mt::Particle::Particle(int index)
{
    m_index = index;
    m_weight = 0.f;
    m_modelMatrix = glm::mat4(1.f);
}

mt::Particle::Particle(int index, float weight, glm::mat4 modelMatrix) {
    m_index = index;
    m_weight = weight;
    m_modelMatrix = modelMatrix;
}

mt::Particle::~Particle() { }

float mt::Particle::getWeight() {
    return m_weight;
}

int mt::Particle::getIndex() {
    return m_index;
}

glm::mat4 mt::Particle::getModelMatrix() {
    return m_modelMatrix;
}

void mt::Particle::setModelMatrix(glm::mat4 model_matrix) {
    m_modelMatrix = model_matrix;
}
