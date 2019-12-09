#include "ShaderSimple.h"

ShaderSimple::ShaderSimple(GLuint shader_mask, const char** shaderPaths) : CVK::ShaderMinimal(shader_mask,shaderPaths)
{
    m_colorID = glGetUniformLocation(m_ProgramID, "color");
}

void ShaderSimple::update()
{
    ShaderMinimal::update();
}

void ShaderSimple::update(CVK::Node *node)
{
    if (node->hasMaterial()) {
        CVK::Material *mat = node->getMaterial();
        glUniform3fv(m_colorID, 1, glm::value_ptr(*mat->getdiffColor()));
    }
}

