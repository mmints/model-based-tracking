#include "CVK_Projection.h"

glm::mat4 *CVK::Projection::getProjMatrix( )
{
	return &m_projection;
	
}
void CVK::Projection::setProjMatrix( glm::mat4 *projection)
{
	m_projection = *projection;
}

void CVK::Projection::getNearFar( float *near, float *far) const
{
	*near = m_znear;
	*far = m_zfar;
}

float CVK::Projection::getNear() const
{
	return m_znear;
}
