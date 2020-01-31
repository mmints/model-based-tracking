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
#include <Shader/ShaderSobel.h>

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
    const char *m_depth_shader_paths[2] = {SHADERS_PATH "/Simple.vert", SHADERS_PATH "/PassDepth.frag"};
    const char *m_sobel_shader_paths[2] = {SHADERS_PATH "/ScreenFill.vert", SHADERS_PATH "/SobelFilter.frag"};

    // Shader
    ShaderSimple *m_color_shader = nullptr;
    ShaderSimple *m_normals_shader = nullptr;
    ShaderSimple *m_depth_shader = nullptr;
    ShaderSobel *m_sobel_shader = nullptr;

    // Matrices
    glm::mat4 m_view_matrix;
    glm::mat4 m_projection_matrix;

    // Matrix Handler
    GLuint m_view_matrix_handler_color;
    GLuint m_projection_matrix_handler_color;

    GLuint m_view_matrix_handler_normals;
    GLuint m_projection_matrix_handler_normals;

    GLuint m_view_matrix_handler_depth;
    GLuint m_projection_matrix_handler_depth;

    // FBOs
    CVK::FBO *m_color_fbo = nullptr;
    CVK::FBO *m_normals_fbo = nullptr;
    CVK::FBO *m_depth_fbo = nullptr;
    CVK::FBO *m_edge_fbo = nullptr;

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
     * Returns the texture ID of the color texture.
     * @return color texture id
     */
    GLuint getColorTexture();

    /**
     * Bind the Normal FBO and render the particle grid with the
     * normals shader into this buffer.
     */
    void renderNormalTexture();

    /**
     * Returns the texture ID of the normal texture.
     * @return normal texture id
     */
    GLuint getNormalTexture();

    /**
     * Bind the Depth FBO and render the particle grid with the
     * depth shader into this buffer.
     */
    void renderDepthTexture();

    /**
     * Returns the texture ID of the depth texture.
     * @return depth texture id
     */
    GLuint getDepthTexture();

    /**
     * Bind the Edge FBO and perform sobel post processing on
     * the color texture with the sobel shader and render into this buffer.
     */
    void renderEdgeTexture();

    /**
     * Returns the texture ID of the edge texture.
     * @return edge texture id
     */
    GLuint getEdgeTexture();

};

}

#endif //MT_PARTICLEGRID_Hk
