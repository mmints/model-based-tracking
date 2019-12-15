#version 330

layout (location = 0) in vec4 position;
layout (location = 2) in vec2 uv;

out vec2 passUVCoord;

void main() {
    gl_Position = position;
    passUVCoord = uv;
}