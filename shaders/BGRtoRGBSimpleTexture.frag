#version 330

in vec2 passUVCoord;

uniform sampler2D colortexture;

vec4 tempColor;
out vec4 fragmentColor;

void main() {
    tempColor = texture(colortexture, passUVCoord);
    fragmentColor = vec4(tempColor.b, tempColor.g, tempColor.r, tempColor.a);
}