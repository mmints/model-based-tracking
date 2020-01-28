#include "ShaderSobel.h"

ShaderSobel::ShaderSobel(GLuint shader_mask, const char** shaderPaths) : CVK::ShaderSimpleTexture(shader_mask,shaderPaths)
{
    m_widthID = glGetUniformLocation(m_ProgramID, "width");
    m_heightID = glGetUniformLocation(m_ProgramID, "height");
}

void ShaderSobel::update()
{
    glUniform1i(m_widthID, m_width);
    glUniform1i(m_heightID, m_height);
    ShaderSimpleTexture::update();
}

void ShaderSobel::setResolution(int width, int height)
{
    m_width = width;
    m_height = height;
}