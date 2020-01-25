#version 330 core

out vec4 fragmentColor;
in vec3 passNormal;


void main(){
    fragmentColor.rgb = passNormal;
    fragmentColor.a = 1.0;
}