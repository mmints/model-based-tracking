#version 330 core

uniform vec3 color;

out vec4 fragmentColor;

void main(){
    fragmentColor.rgb = color;
    fragmentColor.a = 1.0;
}