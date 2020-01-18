#version 330

in vec2 passUVCoord;

uniform sampler2D colortexture;
out vec4 fragmentColor;

void main() {

    float gray = dot(texture(colortexture, passUVCoord).rgb, vec3(0.2126, 0.7152, 0.0722));
    fragmentColor = vec4(vec3(gray), 1.0);
}