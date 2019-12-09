#include "CVK_ShaderMinimal.h"

CVK::ShaderMinimal::ShaderMinimal( GLuint shader_mask, const char** shaderPaths)
{	
	// generate shader program
	GenerateShaderProgramm(shader_mask, shaderPaths);
		
	// matrices
	m_modelMatrixID = glGetUniformLocation(m_ProgramID, "modelMatrix");
	m_viewMatrixID = glGetUniformLocation(m_ProgramID, "viewMatrix");
	m_projectionMatrixID = glGetUniformLocation(m_ProgramID, "projectionMatrix");

}

void CVK::ShaderMinimal::updateModelMatrix(glm::mat4 modelmatrix) const
{
	glUniformMatrix4fv(m_modelMatrixID, 1, GL_FALSE, glm::value_ptr(modelmatrix));
}

void CVK::ShaderMinimal::update()
{
	CVK::Camera* cam = CVK::State::getInstance()->getCamera();
	glm::mat4 projection, viewmatrix;

	if (cam != nullptr)
	{
		viewmatrix = *cam->getView();
		projection = *cam->getProjection()->getProjMatrix();
	}

	glUniformMatrix4fv(m_viewMatrixID, 1, GL_FALSE, glm::value_ptr(viewmatrix));
	glUniformMatrix4fv(m_projectionMatrixID, 1, GL_FALSE, glm::value_ptr(projection));
}

void CVK::ShaderMinimal::update( CVK::Node* node)
{
	
}
