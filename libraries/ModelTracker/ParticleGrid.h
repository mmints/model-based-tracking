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
    const char *m_color_shader_paths[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/Simple.frag"};
    const char *m_normals_shader_paths[2] = {SHADERS_PATH "/PassNormals.vert", SHADERS_PATH "/PassNormals.frag"};

    // Shader
    ShaderSimple *m_color_shader = nullptr;
    ShaderSimple *m_normals_shader = nullptr;

    // Matrices
    glm::mat4 m_view_matrix;
    glm::mat4 m_projection_matrix;

    // Matrix Handler
    GLuint m_view_matrix_handler_color;
    GLuint m_projection_matrix_handler_color;
    GLuint m_view_matrix_handler_normals;
    GLuint m_projection_matrix_handler_normals;

    // FBOs
    CVK::FBO *m_color_fbo = nullptr;
    CVK::FBO *m_normals_fbo = nullptr;

    // Private Functions
    void initializeParticles(int particle_count, int width, int height);
    void renderParticleGrid();

public:
    ParticleGrid(std::string path_to_model, int particle_width, int particle_height, int particle_count);

    /**
     * Bind the Color FBO and render the particle grid with the
     * color shader into this buffer.
     */
    void renderColorTexture();

    /**
     * Retruns the texture ID of the color texture.
     * @return color texture id
     */
    GLint getColorTexture();

    /**
     * Bind the Normal FBO and render the particle grid with the
     * normals shader into this buffer.
     */
    void renderNormalTexture();

    /**
     * Retruns the texture ID of the color texture.
     * @return color texture id
     */
    GLint getNormalTexture();
};

}

#endif //MT_PARTICLEGRID_H
