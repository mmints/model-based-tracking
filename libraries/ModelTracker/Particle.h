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
    glm::mat4 m_rotation_matrix;

public:
    Particle(int index);
    Particle(int index, float weight, glm::vec3 m_translation_vector, glm::mat4 m_roatition_matrix);
    ~Particle();

    float getWeight();
    int getIndex();

    glm::mat4 getModelMatrix();
    void setModelMatrix(glm::mat4 model_matrix);
};

}

#endif //CVK_2_PARTICLE_H