#include "ParticleGrid.h"

using namespace mt;

ParticleGrid::ParticleGrid(std::string path_to_model, int particle_width, int particle_height, int particle_count)
{
    srand(time(0)); // Set a seed for evey new instantiated ParticleGenerator

    if ((int)std::pow((int)std::sqrt(particle_count), 2) != particle_count) {
        throw std::invalid_argument("[mt::ParticleGenerator::ParticleGenerator] Invalid Particle Count. "
                                    "Make sure that it is a result of a square operation.");
    }
    // Set Parameters
    m_model = new CVK::Node("model", path_to_model);
    m_particle_grid_dimension = (int)std::sqrt(particle_count);

    // Set Shader
    m_color_shader = new ShaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_color_shader_paths);
    m_normals_shader = new ShaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_normals_shader_paths);
    m_depth_shader = new ShaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_depth_shader_paths);
    m_sobel_shader = new ShaderSobel( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_sobel_shader_paths);
    m_sobel_shader->setResolution(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height);

    // Set Matrices
    m_view_matrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    m_projection_matrix = glm::perspective(glm::radians(40.0f), (float) particle_width/particle_height, 1.0f, 100.0f);

    // Set Matrix Handler
    m_view_matrix_handler_color = glGetUniformLocation(m_color_shader->getProgramID(), "viewMatrix");
    m_projection_matrix_handler_color = glGetUniformLocation(m_color_shader->getProgramID(), "projectionMatrix");

    m_view_matrix_handler_normals = glGetUniformLocation(m_normals_shader->getProgramID(), "viewMatrix");
    m_projection_matrix_handler_normals = glGetUniformLocation(m_normals_shader->getProgramID(), "projectionMatrix");

    m_view_matrix_handler_depth = glGetUniformLocation(m_depth_shader->getProgramID(), "viewMatrix");
    m_projection_matrix_handler_depth = glGetUniformLocation(m_depth_shader->getProgramID(), "projectionMatrix");

    // Set FBOs
    m_color_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);
    m_normals_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);
    m_depth_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);
    m_edge_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);

    // Init Particles
    initializeParticles(particle_count, particle_width, particle_height);
}

// *** Color FBO *** //

void ParticleGrid::renderColorTexture()
{
    m_color_fbo->bind();
    CVK::State::getInstance()->setShader(m_color_shader);
    m_color_shader->useProgram();

    glUniformMatrix4fv(m_view_matrix_handler_color, 1, GL_FALSE, value_ptr(m_view_matrix));
    glUniformMatrix4fv(m_projection_matrix_handler_color, 1, GL_FALSE, value_ptr(m_projection_matrix));

    renderParticleGrid();

    m_color_fbo->unbind();
}

GLuint ParticleGrid::getColorTexture()
{
    return m_color_fbo->getColorTexture(0);
}

// *** Normal FBO *** //

void ParticleGrid::renderNormalTexture()
{
    m_normals_fbo->bind();
    CVK::State::getInstance()->setShader(m_normals_shader);
    m_normals_shader->useProgram();

    glUniformMatrix4fv(m_view_matrix_handler_normals, 1, GL_FALSE, value_ptr(m_view_matrix));
    glUniformMatrix4fv(m_projection_matrix_handler_normals, 1, GL_FALSE, value_ptr(m_projection_matrix));

    renderParticleGrid();

    m_normals_fbo->unbind();
}

GLuint ParticleGrid::getNormalTexture()
{
    return m_normals_fbo->getColorTexture(0);
}

// *** Depth FBO *** //

void ParticleGrid::renderDepthTexture()
{
    m_depth_fbo->bind();
    CVK::State::getInstance()->setShader(m_depth_shader);
    m_depth_shader->useProgram();

    glUniformMatrix4fv(m_view_matrix_handler_depth, 1, GL_FALSE, value_ptr(m_view_matrix));
    glUniformMatrix4fv(m_projection_matrix_handler_depth, 1, GL_FALSE, value_ptr(m_projection_matrix));

    renderParticleGrid();

    m_depth_fbo->unbind();
}

GLuint ParticleGrid::getDepthTexture()
{
    return m_depth_fbo->getColorTexture(0);
}

// *** Edge FBO *** //

void ParticleGrid::renderEdgeTexture()
{
    m_edge_fbo->bind();
    m_sobel_shader->setTextureInput(0, m_color_fbo->getColorTexture(0));
    m_sobel_shader->useProgram();
    m_sobel_shader->update();
    m_sobel_shader->render();
    m_edge_fbo->unbind();
}

GLuint ParticleGrid::getEdgeTexture()
{
    return m_edge_fbo->getColorTexture(0);
}


// *** Private Functions *** //

void ParticleGrid::initializeParticles(int particle_count, int width, int height)
{
    for (int i = 0; i < particle_count; i++)
    {
        mt::Particle particle(width, height);
        m_particles.push_back(particle);
    }
}

void ParticleGrid::renderParticleGrid()
{
    int width = m_particles[0].getWidth();
    int height = m_particles[0].getHeight();

    int i = 0;
    for (int x = 0; x < m_particle_grid_dimension; x++)
    {
        for (int y = 0; y < m_particle_grid_dimension; y++)
        {
            glViewport(width * x, height * y, width, height);
            m_model->setModelMatrix(m_particles[0].getModelMatrix());
            m_model->render();
            i++;
        }
    }
}