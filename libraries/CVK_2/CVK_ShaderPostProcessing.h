#ifndef __CVK_SHADER_POST_PROCESSING_H
#define __CVK_SHADER_POST_PROCESSING_H

#include "CVK_Defs.h"
#include "CVK_ShaderSet.h"
#include "CVK_Plane.h"

namespace CVK
{

/**
* Minimal Shader class implementation for Post Processing using the ShaderSet. Uses texture input
* information from ShaderSet. Has own render-Method which renders a screen filling quad.
* @brief Shader for post processing
*/
class ShaderPostProcessing : public CVK::ShaderSet
{
public:
	/**
	* Constructor for ShaderPostProcessing with given parameters. Collects uniform locations for
	* all used variables from Shader Program.
	* @param shader_mask Describes which shader files are used
	* @param shaderPaths Array of paths to shader files
	*/
	ShaderPostProcessing(GLuint shader_mask, const char** shaderPaths);

	/**
	* Sets rendering dependent variables in Shader. None used in ShaderPostProcessing, but needed for subclasses.
	* @brief Sets rendering variables
	*/
	virtual void update();

	/**
	 * Renders screen filling quad using this shader for coloring
	 * @brief Renders screen filling quad
	 */
	void render();



    /**
     * Renders a on X-axis mirrored screen filling quad using this shader for coloring
     * @brief Renders screen filling quad
     */
    void renderNormalMap();

    /**
 * Renders a on Y and X-axis mirrored screen filling quad using this shader for coloring
 * @brief Renders screen filling quad
 */
    void renderZED();

private:
	CVK::Plane m_screenFillingQuad; //!< Used geometry for rendering
    CVK::Plane m_screenFillingQuadZED; //!< Used geometry for rendering --- MIRRORED on Y and X-Axis
    CVK::Plane m_screenFillingQuadNormalMap; //!< Used geometry for rendering --- MIRRORED on X-Axis

};

}

#endif /* __CVK_SHADER_POST_PROCESSING_H */
