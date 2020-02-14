#include <random>
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

    // Set array to 0.f
    for (int i = 0; i < m_particle_count; i++)
    {
        m_sum_weight_memory[i] = 0.f;
        m_color_weight_memory[i] = 0.f;
        m_depth_weight_memory[i] = 0.f;
        m_normals_weight_memory[i] = 0.f;
        m_edge_weight_memory[i] = 0.f;
    }
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_color_weight_memory, m_color_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_depth_weight_memory, m_depth_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_normals_weight_memory, m_normals_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_edge_weight_memory, m_edge_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));

    // Register and map texture to CudaArray
    mapGLTextureToCudaArray(particleGrid.getColorTexture(), m_color_texture_array);
    mapGLTextureToCudaArray(particleGrid.getDepthTexture(), m_depth_texture_array);
    mapGLTextureToCudaArray(particleGrid.getNormalTexture(), m_normals_texture_array);
    mapGLTextureToCudaArray(particleGrid.getEdgeTexture(), m_edge_texture_array);
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

void mt::ParticleFilter::calculateWeightEdge(sl::Mat in, mt::ParticleGrid &particleGrid)
{
    HANDLE_CUDA_ERROR(cudaMemcpy(dev_edge_weight_memory, m_edge_weight_memory, m_particle_count * sizeof(float), cudaMemcpyHostToDevice));
    mt::calculateWeightEdge(in, dev_edge_weight_memory, m_edge_texture_array, particleGrid);
    HANDLE_CUDA_ERROR(cudaMemcpy(m_edge_weight_memory, dev_edge_weight_memory, m_particle_count * sizeof(float), cudaMemcpyDeviceToHost));
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

    particleGrid.sortParticlesByWeight();     // DO IT IN MAIN CODE
    //printf("HEAVIEST PARTICLE: %f \n", heaviest_particle.getWeight());
/*
    for (int i = 0; i < threshold; i++)
    {
        m_top_particles.push_back(particleGrid.m_particles[i]);
        //printf("Set [%i] Particle W: %f \t", i, m_top_particles[i].getWeight());

        // Normalize
        if ( particleGrid.m_particles[0].getWeight() > 0.f) {
            m_top_particles[i].setWeight(m_top_particles[i].getWeight() /  particleGrid.m_particles[0].getWeight());
        } else{
            m_top_particles[i].setWeight(0.f);
        }
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
*/
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, threshold-1);

    std::vector<Particle> tmp;
    for (int n=0; n<m_particle_count; ++n)
        tmp.push_back(particleGrid.m_particles.at(dis(gen)));

    particleGrid.m_particles = tmp;


   // m_top_particles.clear();
}
