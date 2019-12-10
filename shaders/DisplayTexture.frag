uniform sampler2D texImage;

void main() {
    gl_FragColor = texture2D(texImage, gl_TexCoord[0].st);
}