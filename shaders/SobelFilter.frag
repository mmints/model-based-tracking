#version 330

in vec2 passUVCoord;

uniform sampler2D colortexture;
out vec4 fragmentColor;

float width = 512.f;
float height = width;

// Use these parameters to fiddle with settings
float step = 1.0;

float intensity(in vec4 color){
    return sqrt((color.x*color.x)+(color.y*color.y)+(color.z*color.z));
}

vec3 sobel(float stepx, float stepy, vec2 center){
    // get samples around pixel
    float tleft = intensity(texture(colortexture,center + vec2(-stepx,stepy)));
    float left = intensity(texture(colortexture,center + vec2(-stepx,0)));
    float bleft = intensity(texture(colortexture,center + vec2(-stepx,-stepy)));

    float top = intensity(texture(colortexture,center + vec2(0,stepy)));
    float bottom = intensity(texture(colortexture,center + vec2(0,-stepy)));

    float tright = intensity(texture(colortexture,center + vec2(stepx,stepy)));
    float right = intensity(texture(colortexture,center + vec2(stepx,0)));
    float bright = intensity(texture(colortexture,center + vec2(stepx,-stepy)));

    // Sobel masks (see http://en.wikipedia.org/wiki/Sobel_operator)
    //        1 0 -1     -1 -2 -1
    //    X = 2 0 -2  Y = 0  0  0
    //        1 0 -1      1  2  1

    float x = tleft + 2.0*left + bleft - tright - 2.0*right - bright;
    float y = -tleft - 2.0*top - tright + bleft + 2.0 * bottom + bright;
    float color = sqrt((x*x) + (y*y));
    if (color > 0.1f)
        color = 1.f;
    return vec3(color,color,color);
}

void main(void)
{
    fragmentColor.xyz = sobel(step/width, step/height, passUVCoord);
}
