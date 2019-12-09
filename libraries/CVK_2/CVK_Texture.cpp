#include <string>
#include <iostream>
#include "CVK_Texture.h"
#include "stb_image.h"

CVK::Texture::Texture(const std::string fileName)
{
	m_textureID = INVALID_GL_VALUE;

	createTexture();
	load( fileName);
}

CVK::Texture::Texture( int width, int height, int bytesPerPixel, unsigned char *data)
{
	m_textureID = INVALID_GL_VALUE;

	createTexture();
	setTexture(width, height, bytesPerPixel, data);
}

CVK::Texture::Texture( GLuint texture)
{
	setTexture(texture);
}

CVK::Texture::~Texture()
{
	if (m_textureID != INVALID_GL_VALUE) glDeleteTextures(1, &m_textureID);
}

void CVK::Texture::createTexture()
{
	glGenTextures( 1, &m_textureID);
	glBindTexture( GL_TEXTURE_2D, m_textureID);
	glTexParameterf( GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameterf( GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameterf( GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameterf( GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
}

bool CVK::Texture::load(const std::string fileName)
{
	int bytesPerPixel = 0;

	unsigned char *data = stbi_load(fileName.c_str(), &m_width, &m_height, &bytesPerPixel, 0);

	//flip image vertically
	unsigned char* s = data;
	for (int y=0; y<m_height/2; y++)
	{
		unsigned char* e = data+(m_height-y-1)*m_width*bytesPerPixel;
		for(int x=0; x<m_width*bytesPerPixel; x++)
		{
			unsigned char temp = *s;
			*s = *e;
			*e = temp;
			s++;
			e++;
		}
	}

	//send image data to the new texture
	if (bytesPerPixel < 3)
	{
		std::cout << "ERROR: Unable to load texture image " << fileName << std::endl;
		return false;
	}
	else 
	{
		setTexture( m_width, m_height, bytesPerPixel, data);
	}

	//stbi_image_free(data);	//keep copy e.g. for CPU ray tracing
	glGenerateMipmap(GL_TEXTURE_2D);          

	std::cout << "SUCCESS: Loaded texture image " << fileName << std::endl;
	return true; 
}

void CVK::Texture::bind() const
{
	if (m_textureID != INVALID_GL_VALUE)  glBindTexture( GL_TEXTURE_2D, m_textureID);
}

void CVK::Texture::setTexture( GLuint texture)
{
	m_textureID = texture;
}

void CVK::Texture::setTexture( int width, int height, int bytesPerPixel, unsigned char *data)
{
	m_width = width;
	m_height = height;
	m_bytesPerPixel = bytesPerPixel;
	m_data = data;

	if (m_textureID == INVALID_GL_VALUE) createTexture();

	glBindTexture( GL_TEXTURE_2D, m_textureID);
	if (m_bytesPerPixel == 3)
	{
		glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, m_width, m_height, 0, GL_RGB, GL_UNSIGNED_BYTE, m_data);
	} 
	else if (m_bytesPerPixel == 4) 
	{
		glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_data);
	} 
	else 
	{
		std::cout << "RESOLVED: Unknown format for bytes per pixel in texture, changed to 4" << std::endl;
		glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_data);
	}
	glGenerateMipmap(GL_TEXTURE_2D);    
}

unsigned int CVK::Texture::getTexture() const
{
	return m_textureID;
}

glm::vec3 CVK::Texture::getValue( glm::vec2 tcoord) const
{
	int x = (int) (tcoord.x * m_width);
	int y = (int) (tcoord.y * m_height);
	unsigned char *texel = m_data + m_bytesPerPixel * (y * m_width + x);
	return (glm::vec3( *texel / 255.f, *(texel+1) / 255.f, *(texel+2) / 255.f));
}
