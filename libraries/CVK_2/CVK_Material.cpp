#include "CVK_Material.h"

CVK::Material::Material( glm::vec3 diffuse, glm::vec3 specular, float shininess)
{
	init( 1.f, diffuse, 1.f, specular, shininess);
}

CVK::Material::Material( float kd, glm::vec3 diffuse)
{
	init( kd, diffuse, 0.f, BLACK, 0.f);
}

CVK::Material::Material( float kd, glm::vec3 diffuse, float ks, glm::vec3 specular, float shininess)
{
	init( kd, diffuse, ks, specular, shininess);
}

CVK::Material::Material(const std::string colorTexturePath, float kd)
{
	init( kd, WHITE, 0.f, BLACK, 0.f);
	setTexture(COLOR_TEXTURE, colorTexturePath);
}

CVK::Material::Material( GLuint colorTextureID, float kd)
{
	init( kd, WHITE, 0.f, BLACK, 0.f);
	setTexture(COLOR_TEXTURE, colorTextureID);
}

CVK::Material::Material(const std::string colorTexturePath, float kd, float ks, glm::vec3 specular, float shininess)
{
	init( kd, WHITE, ks, specular, shininess);
	setTexture(COLOR_TEXTURE, colorTexturePath);
}

CVK::Material::Material( GLuint colorTextureID, float kd, float ks, glm::vec3 specular, float shininess)
{
	init( kd, WHITE, ks, specular, shininess);
	setTexture(COLOR_TEXTURE, colorTextureID);
}
void CVK::Material::init( float kd, glm::vec3 diffuse, float ks, glm::vec3 specular, float shininess)
{
	m_kd = kd;
	m_ks = ks;
	m_diffColor = diffuse;
	m_specColor = specular;
	m_shininess = shininess;

	m_kt = 0.f; 
	m_ior = 0;
}

CVK::Material::~Material()
{	
}

void CVK::Material::setKd( float kd)
{
	m_kd = kd;
}

float CVK::Material::getKd() const
{
	return m_kd;
}

void CVK::Material::setKs( float ks)
{
	m_ks = ks;
}

float CVK::Material::getKs() const
{
	return m_ks;
}
void CVK::Material::setKt( float kt)
{
	m_kt = kt;
}

float CVK::Material::getKt() const
{
	return m_kt;
}

void CVK::Material::setdiffColor( glm::vec3 col)
{
	m_diffColor = col;
}

glm::vec3* CVK::Material::getdiffColor()
{
	return &m_diffColor;
}

void CVK::Material::setspecColor( glm::vec3 col)
{
	m_specColor = col;
}

glm::vec3* CVK::Material::getspecColor()
{
	return &m_specColor;
}

void CVK::Material::setShininess( float shininess)
{
	m_shininess = shininess;
}

float CVK::Material::getShininess() const
{
	return m_shininess;
}

void CVK::Material::setIor( float ior)
{
	m_ior = ior;
}

float CVK::Material::getIor() const
{
	return m_ior;
}

void CVK::Material::setTexture( TextureType type, const std::string fileName)
{
	switch(type)
	{
		case COLOR_TEXTURE:
			if(m_colorTexture)
				m_colorTexture->load( fileName);
			else
				m_colorTexture = new Texture( fileName); 
			break;
		case NORMAL_TEXTURE:
			if(m_normalTexture)
				m_normalTexture->load( fileName);
			else
				m_normalTexture = new Texture( fileName);
			break;

	}
}

void CVK::Material::setTexture( TextureType type, GLuint textureID)
{
	switch(type)
	{
		case COLOR_TEXTURE:
			if(m_colorTexture)
				m_colorTexture->setTexture(textureID);
			else
				m_colorTexture = new Texture(textureID); 
			break;
		case NORMAL_TEXTURE:
			if(m_normalTexture)
				m_normalTexture->setTexture(textureID);
			else
				m_normalTexture = new Texture(textureID);
			break;

	}
}

bool CVK::Material::hasTexture( TextureType type) const
{
	switch(type)
	{
		case COLOR_TEXTURE:
			return m_colorTexture != nullptr;
		case NORMAL_TEXTURE:
			return m_normalTexture != nullptr;
		default:
			return false;
	}
}

CVK::Texture* CVK::Material::getTexture( TextureType type) const
{
	switch(type)
	{
		case COLOR_TEXTURE:
			return m_colorTexture;
		case NORMAL_TEXTURE:
			return m_normalTexture;
		default:
			return 0;
	}
}
