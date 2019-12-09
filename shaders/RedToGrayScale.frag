uniform sampler2D texImage;

void main() {
    vec4 color = texture2D(texImage, gl_TexCoord[0].st);
    gl_FragColor = vec4(color.r, color.r, color.r, color.a);
}