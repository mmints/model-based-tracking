#ifndef MT_PARTICLEGRID_H
#define MT_PARTICLEGRID_H

#include <Shader/ShaderSimple.h>
#include <vector>
#include <cmath>
#include <GL/glew.h>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include "Particle.h"
#include <CVK_2/CVK_Framework.h>

namespace mt
{

class ParticleGrid
{
private:
    // Particles and ParticleGrid Parameters
    std::vector<mt::Particle> m_particles;
    CVK::Node *m_model = nullptr;
    int m_particle_grid_dimension; // particle_grid_rows == particle_gird_columns == particle_dimension

    // Shader paths
    const char *m_shader_simple_paths[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};

    // Shader
    ShaderSimple *m_shader_simple = nullptr;

    // Matrices
    glm::mat4 m_view_matrix;
    glm::mat4 m_projection_matrix;

    // Matrix Handler
    GLuint m_view_matrix_handler;
    GLuint m_projection_matrix_handler;

    // FBOs
    CVK::FBO *m_color_fbo = nullptr;

    // Private Functions
    void initializeParticles(int particle_count, int width, int height);

public:
    ParticleGrid(std::string path_to_model, int particle_width, int particle_height, int particle_count);

    // Color FBO
    void renderColorTexture();
    GLint getColorTexture();
};

}

#endif //MT_PARTICLEGRID_H
