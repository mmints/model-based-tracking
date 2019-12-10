#ifndef CVK_2_PARTICLE_H
#define CVK_2_PARTICLE_H

#include <glm/glm.hpp>

namespace mt
{

class Particle
{
private:
    float m_weight;
    int m_index;
    glm::mat4 m_modelMatrix;

public:
    Particle(int index);
    Particle(int index, float weight, glm::mat4 modelMatrix);
    ~Particle();

    float getWeight();
    int getIndex();

    glm::mat4 getModelMatrix();
    void setModelMatrix(glm::mat4 model_matrix);
};

}



#endif //CVK_2_PARTICLE_H
