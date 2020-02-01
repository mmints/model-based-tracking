#version 330

uniform int width;
uniform int height;

in vec2 passUVCoord;

uniform sampler2D colortexture;

out vec4 fragmentColor;

// Use these parameters to fiddle with settings
float step = 1.0;

float intensity(vec4 color){
    // Transform RGB values of texture into gray scale befor calculation the itensity
    float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114)); // Values from Priese Buch S. 53
    return dot(vec3(gray),vec3(gray));
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

    float x =  tleft + 2.0 * left + bleft  - tright - 2.0 * right  - bright;
    float y = -tleft - 2.0 * top  - tright + bleft  + 2.0 * bottom + bright;
    float color = dot(vec2(x,y), vec2(x,y));
    if (color > 0.01f)
    color = 1.f;
    return vec3(color,color,color);
}

void main(void)
{
    fragmentColor.xyz = sobel(step/width, step/height, passUVCoord);
}
