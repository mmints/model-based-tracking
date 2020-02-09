#include "ParticleFilter.h"

mt::ParticleFilter::ParticleFilter(mt::ParticleGrid &particleGrid)
{
    m_particle_count = particleGrid.getParticleCount();

    // Allocate weight memory on host
    m_color_weight_memory   = new float[m_particle_count];
    m_depth_weight_memory   = new float[m_particle_count];
    m_normals_weight_memory = new float[m_particle_count];
    m_edge_weight_memory = new float[m_particle_count]; // TODO: Not implemented jet!
    m_sum_weight_memory = new float[m_particle_count];


    // Allocate weight memory on device
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_color_weight_memory, m_particle_count * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_depth_weight_memory, m_particle_count * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_normals_weight_memory, m_particle_count * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_edge_weight_memory, m_particle_count * sizeof(float)));  // TODO: Not implemented jet!
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_sum_weight_memory, m_particle_count * sizeof(float)));

    setWeightsToZero();

    // Register and map texture to CudaArray
    mapGLTextureToCudaArray(particleGrid.getColorTexture(), m_color_texture_array);
    mapGLTextureToCudaArray(particleGrid.getDepthTexture(), m_depth_texture_array);
    mapGLTextureToCudaArray(particleGrid.getNormalTexture(), m_normals_texture_array);
}

void mt::ParticleFilter::mapGLTextureToCudaArray(GLuint texture_id, cudaArray_t &texture_array)
{
    HANDLE_CUDA_ERROR(cudaGraphicsGLRegisterImage(&m_texture_resource, texture_id, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
    HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, &m_texture_resource));

    HANDLE_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_array, m_texture_resource, 0, 0));
    HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_texture_resource));
}

void mt::ParticleFilter::calculateWeightColor(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_color_weight_memory, m_color_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeight(in, dev_color_weight_memory, m_color_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_color_weight_memory, dev_color_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));
}

void mt::ParticleFilter::calculateWeightDepth(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_depth_weight_memory, m_depth_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeight(in, dev_depth_weight_memory, m_depth_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_depth_weight_memory, dev_depth_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));
}

void mt::ParticleFilter::calculateWeightNormals(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_normals_weight_memory, m_normals_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeight(in, dev_normals_weight_memory, m_normals_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_normals_weight_memory, dev_normals_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));
}


void mt::ParticleFilter::setParticleWeight(mt::ParticleGrid &particleGrid)
{
    for (int i = 0; i < m_particle_count; i++)
    {
        m_sum_weight_memory[i] = 0.f;
        m_sum_weight_memory[i] += m_color_weight_memory[i];
        m_sum_weight_memory[i] += m_depth_weight_memory[i];
        m_sum_weight_memory[i] += m_normals_weight_memory[i];
        m_sum_weight_memory[i] += m_edge_weight_memory[i];

        particleGrid.m_particles[i].setWeight(m_sum_weight_memory[i]);

        m_color_weight_memory[i] = 0.f;
        m_depth_weight_memory[i] = 0.f;
        m_normals_weight_memory[i] = 0.f;
        m_edge_weight_memory[i] = 0.f;
    }
}

void mt::ParticleFilter::resample(mt::ParticleGrid &particleGrid, int threshold)
{
    //printf("RESAMPLE DEBUG\n");

    particleGrid.sortParticlesByWeight();
    mt::Particle heaviest_particle = particleGrid.m_particles[0]; // Get the heaviest particle for rendering before refilling particleGrid.m_particles
    //printf("HEAVIEST PARTICLE: %f \n", heaviest_particle.getWeight());

    for (int i = 0; i < threshold; i++)
    {
        m_top_particles.push_back(particleGrid.m_particles[i]);
        //printf("Set [%i] Particle W: %f \t", i, m_top_particles[i].getWeight());

        // Normalize
        m_top_particles[i].setWeight(m_top_particles[i].getWeight() / heaviest_particle.getWeight());
        //printf("NORMAL [%i] Particle W: %f \n", i, m_top_particles[i].getWeight());
    }

    float pick = 0.2;
    for (int i = 0; i < m_particle_count; i++)
    {
        if (pick > 1.0)
            pick = 0.2;

        if (m_top_particles[i % threshold].getWeight() < pick) {
            particleGrid.m_particles[i] = m_top_particles[(i % threshold) + 1];
            pick += pick;
            continue;
        }
        else {
            particleGrid.m_particles[i] = m_top_particles[(i % threshold)];
            pick += pick;
        }
    }

    m_top_particles.clear();
}

// *** This implementation of the setParticleWeight is done with kernel functions and less momory allocation and copying
//     But it does not work jet. Probably because sumWeights is buggy.
//     If there are large performance issues I should have a look back here.
// TODO: Try to fix this

/*void mt::ParticleFilter::setParticleWeight(mt::ParticleGrid &particleGrid)
{
    sumWeights();
    HANDLE_CUDA_ERROR(cudaMemcpy(m_sum_weight_memory, dev_sum_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m_particle_count; i++)
    {
        particleGrid.m_particles[i].setWeight(m_sum_weight_memory[i]);
    }
    setWeightsToZero();
}*/

// **** Private Functions **** //


// TODO: Probably does not work
void mt::ParticleFilter::sumWeights()
{
    mt::sumWeights(dev_color_weight_memory, dev_depth_weight_memory, dev_normals_weight_memory, dev_edge_weight_memory, dev_sum_weight_memory, m_particle_count);
}


void mt::ParticleFilter::setWeightsToZero()
{
    mt::setZeroArray(dev_color_weight_memory, m_particle_count);
    mt::setZeroArray(dev_depth_weight_memory, m_particle_count);
    mt::setZeroArray(dev_normals_weight_memory, m_particle_count);
    mt::setZeroArray(dev_edge_weight_memory, m_particle_count);

    mt::setZeroArray(dev_sum_weight_memory, m_particle_count);
}
