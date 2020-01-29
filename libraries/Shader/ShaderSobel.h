#ifndef CVK_2_SHADERSOBEL_H
#define CVK_2_SHADERSOBEL_H

#include <CVK_2/CVK_Framework.h>

class ShaderSobel : public CVK::ShaderSimpleTexture
{
public:
    ShaderSobel( GLuint shader_mask, const char** shaderPaths);
    void setResolution(int width, int height);
    void update();

private:
    GLuint m_widthID;
    GLuint m_heightID;
    GLuint m_width;
    GLuint m_height;

};


#endif //CVK_2_SHADERSOBEL_H
