#ifndef CVK_2_SHADERSIMPLE_H
#define CVK_2_SHADERSIMPLE_H

#include <CVK_2/CVK_ShaderMinimal.h>

class ShaderSimple : public CVK::ShaderMinimal
{
public:
    ShaderSimple( GLuint shader_mask, const char** shaderPaths);
    void update() override;
    void update( CVK::Node* node) override;

private:
    GLuint m_colorID;
};


#endif //CVK_2_SHADERSIMPLE_H
