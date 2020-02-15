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
    m_particle_count = particle_count;
    m_particle_grid_dimension = (int)std::sqrt(particle_count);

    m_particle_width = particle_width;
    m_particle_height = particle_height;

    // Set Shader
    m_color_shader = new ShaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_color_shader_paths);
    m_normals_shader = new ShaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_normals_shader_paths);
    m_depth_shader = new ShaderSimple( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_depth_shader_paths);
    m_sobel_shader = new ShaderSobel( VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT, m_sobel_shader_paths);
    m_sobel_shader->setResolution(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height);

    // Set Matrices || view_matrix.z -> Distance Camera to object
    m_view_matrix = glm::lookAt(glm::vec3(0.0, 0.0, 10.0f), glm::vec3(0.0f, 0.0, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    m_projection_matrix = glm::perspective(glm::radians(86.411667f), (float) particle_width/particle_height, 1.0f, 100.0f);

    // Set Matrix Handler
    m_view_matrix_handler_color = glGetUniformLocation(m_color_shader->getProgramID(), "viewMatrix");
    m_projection_matrix_handler_color = glGetUniformLocation(m_color_shader->getProgramID(), "projectionMatrix");

    m_view_matrix_handler_normals = glGetUniformLocation(m_normals_shader->getProgramID(), "viewMatrix");
    m_projection_matrix_handler_normals = glGetUniformLocation(m_normals_shader->getProgramID(), "projectionMatrix");

    m_view_matrix_handler_depth = glGetUniformLocation(m_depth_shader->getProgramID(), "viewMatrix");
    m_projection_matrix_handler_depth = glGetUniformLocation(m_depth_shader->getProgramID(), "projectionMatrix");

    // Full Screen Rendering
    m_fullscreen_projection_matrix = glm::perspective(glm::radians(86.411667f), (float) 1280/720, 1.0f, 100.0f);
    m_fullscreen_projection_matrix_handler = glGetUniformLocation(m_color_shader->getProgramID(), "projectionMatrix");

    // Set FBOs
    m_color_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);
    m_normals_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);
    m_depth_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);
    m_edge_fbo = new CVK::FBO(m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height, 1, true);

    printf("[ParticleFilter] FBO Resolution: %i x %i \n", m_particle_grid_dimension * particle_width, m_particle_grid_dimension * particle_height);

    // Init Particles
    initializeParticles(particle_count, particle_width, particle_height);

    // Set Back Ground Color of the Current GL Instance
    CVK::State::getInstance()->setBackgroundColor(BLACK);
    glm::vec3 BgCol = CVK::State::getInstance()->getBackgroundColor();
    glClearColor( BgCol.r, BgCol.g, BgCol.b, 0.0);
}

void ParticleGrid::update(float rotation_deviation, float translation_deviation)
{
    glm::vec3 rotation_angles;
    glm::vec3 translation;
    for (int i = 0; i < m_particle_count; i++)
    {
        rotation_angles = glm::gaussRand(m_particles[i].getRotation(), glm::vec3(rotation_deviation));
        translation = glm::gaussRand(m_particles[i].getTranslation(), glm::vec3(translation_deviation));
        m_particles[i].setModelMatrix(translation, rotation_angles);
    }
}

// *** All FBOs *** //

void ParticleGrid::renderAllTextures()
{
    renderColorTexture();
    renderNormalTexture();
    renderDepthTexture();
    renderEdgeTexture();
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
    m_sobel_shader->setTextureInput(0, m_color_fbo->getColorTexture(0));
    m_sobel_shader->useProgram();
    m_sobel_shader->update();

    m_edge_fbo->bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_sobel_shader->render();
    m_edge_fbo->unbind();
}

GLuint ParticleGrid::getEdgeTexture()
{
    return m_edge_fbo->getColorTexture(0);
}

// *** Properties *** //

int ParticleGrid::getParticleWidth() {
    return m_particle_width;
}

int ParticleGrid::getParticleHeight() {
    return m_particle_height;
}

int ParticleGrid::getParticleCount() {
    return m_particle_count;
}

int ParticleGrid::getParticleGridDimension() {
    return m_particle_grid_dimension;
}

void ParticleGrid::sortParticlesByWeight()
{
    sort( m_particles.begin( ), m_particles.end( ), [ ](Particle lhs, Particle& rhs )
    {
        return lhs.getWeight() > rhs.getWeight();
    });
}

void ParticleGrid::renderFirstParticleToScreen()
{
    glClear(GL_DEPTH_BUFFER_BIT);

    CVK::State::getInstance()->setShader(m_color_shader);
    m_color_shader->useProgram();

    glUniformMatrix4fv(m_view_matrix_handler_color, 1, GL_FALSE, value_ptr(m_view_matrix));
    glUniformMatrix4fv(m_fullscreen_projection_matrix_handler, 1, GL_FALSE, value_ptr(m_fullscreen_projection_matrix));

    glViewport(0, 0, 1280, 720);

    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    for (int i = 0; i < 10; i++)
    {
        m_model->setModelMatrix(m_particles[i].getModelMatrix());
        m_model->render();

    }
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

// *** Private Functions *** //

void ParticleGrid::initializeParticles(int particle_count, int width, int height)
{
    glm::vec3 scene_center = glm::vec3(0.f);
    glm::vec3 rotation_angles= glm::vec3(0.f);;

    for (int i = 0; i < particle_count; i++)
    {
        mt::Particle particle(width, height, 0.f, scene_center, rotation_angles);
        m_particles.push_back(particle);
    }
}

void ParticleGrid::renderParticleGrid()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    int width = m_particles[0].getWidth();
    int height = m_particles[0].getHeight();

    int i = 0;
    for (int x = 0; x < m_particle_grid_dimension; x++)
    {
        for (int y = 0; y < m_particle_grid_dimension; y++)
        {
            glViewport(width * x, height * y, width, height);
            m_model->setModelMatrix(m_particles[i].getModelMatrix());
            m_model->render();
            i++;
        }
    }
}