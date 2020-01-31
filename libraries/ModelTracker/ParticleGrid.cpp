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

    // Set Matrices
    m_view_matrix = glm::lookAt(glm::vec3(0.0, 0.0, 25.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    m_projection_matrix = glm::perspective(glm::radians(40.0f), (float) particle_width/particle_height, 1.0f, 100.0f);

    // Set Matrix Handler
    m_view_matrix_handler = glGetUniformLocation(m_color_shader->getProgramID(), "viewMatrix");
    m_projection_matrix_handler = glGetUniformLocation(m_color_shader->getProgramID(), "projectionMatrix");

    // Set FBOs
    m_color_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);

    // Init Particles
    initializeParticles(particle_count, particle_width, particle_height);
}

/**
 * Bind the Color FBO and iterate over the ParticleGrid structure with an view port.
 * Render at every position one particle.
 */
void ParticleGrid::renderColorTexture()
{
    m_color_fbo->bind();
    CVK::State::getInstance()->setShader(m_color_shader);
    m_color_shader->useProgram();
    glUniformMatrix4fv(m_view_matrix_handler, 1, GL_FALSE, value_ptr(m_view_matrix));
    glUniformMatrix4fv(m_projection_matrix_handler, 1, GL_FALSE, value_ptr(m_projection_matrix));

    renderParticleGrid();

    m_color_fbo->unbind();
}

/**
 * Retruns the texture ID of the color texture.
 * @return color texture id
 */
GLint ParticleGrid::getColorTexture()
{
    return m_color_fbo->getColorTexture(0);
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
    for (int x = 0; x < m_particle_grid_dimension; x++) {
        for (int y = 0; y < m_particle_grid_dimension; y++) {
            glViewport(width * x, height * y, width, height);
            m_model->setModelMatrix(m_particles[0].getModelMatrix());
            m_model->render();
            i++;
        }
    }
}
