#version 330 core

layout (location = 0) in vec4 positionAttribute;
layout (location = 1) in vec3 normal;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 passNormal;

void main(){
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * positionAttribute;

    // Transform the normal correctly into view space
    // and pass it to the fragment shader
    mat3 normalMatrix = mat3(transpose(inverse(viewMatrix * modelMatrix)));
    passNormal = normalize(normalMatrix * normal);
}