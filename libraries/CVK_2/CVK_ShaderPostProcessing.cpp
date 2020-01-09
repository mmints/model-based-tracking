#include "CVK_ShaderPostProcessing.h"

CVK::ShaderPostProcessing::ShaderPostProcessing( GLuint shader_mask, const char** shaderPaths)
{	
	// generate shader program
	GenerateShaderProgramm(shader_mask, shaderPaths);


	m_screenFillingQuad.set_Points(
		glm::vec3( -1.f, 1.f, 0.f),
		glm::vec3( -1.f, -1.f, 0.f),
		glm::vec3( 1.f, -1.f, 0.f),
		glm::vec3( 1.f, 1.f, 0.f));

	m_screenFillingQuadZED.set_Points(
	        glm::vec3( -1.f, -1.f, 0.f),
	        glm::vec3( -1.f, 1.f, 0.f),
	        glm::vec3( 1.f, 1.f, 0.f),
	        glm::vec3( 1.f, -1.f, 0.f));
}

void CVK::ShaderPostProcessing::update()
{
	
}

void CVK::ShaderPostProcessing::render()
{
	m_screenFillingQuad.render();
}

void CVK::ShaderPostProcessing::renderZED()
{
    m_screenFillingQuadZED.render();
}