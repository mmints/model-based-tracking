#ifndef CVK_2_PARTICLE_H
#define CVK_2_PARTICLE_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace mt
{

class Particle
{
private:
    float m_weight;
    int m_index;
    glm::vec3 m_translation_vector;
    glm::vec3 m_rotation_angles;

public:
    Particle(int index);
    Particle(int index, float weight, glm::vec3 translation_vector, glm::vec3 rotation_angles);

    void setWeight(float weight);
    float getWeight();

    int getIndex();

    glm::vec3 getTranslation();
    glm::vec3 getRotation();

    void setModelMatrix(glm::vec3 translation_vector, glm::vec3 rotation_angles);
    glm::mat4 getModelMatrix();
};

}

#endif //CVK_2_PARTICLE_H