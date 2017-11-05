#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#define threads_per_block 256

using namespace std;

__global__ void clustering_coefficient(const float* adj_matrix, const uint num_nodes, float* d_sum)
{
     __shared__ float temp[threads_per_block];
     const uint u_id = blockDim.x * blockIdx.x + threadIdx.x;
     const uint u_index = u_id * num_nodes;
     float n_u = 0.0f; //number of vertices in the neighborhood of vertex u
     float m_u = 0.0f; //number of edges in the neighborhood of vertex u
     if (u_id < num_nodes)
     {
          for (uint v_id = 0; v_id < num_nodes; v_id++)
          {
               const uint v_index = u_index + v_id;
               if (adj_matrix[v_index] == 1.0f)
               {
                    n_u++;
                    for (uint w_id = v_id + 1; w_id < num_nodes; w_id++)
                    {
                         const uint w_index = u_index + w_id;
                         if (adj_matrix[w_index] == 1.0f &&
                             adj_matrix[v_id * num_nodes + w_id] == 1.0f)
                         {
                              m_u++;
                         }
                    }
               }
          }
          float numerator = 2.0f * m_u;
          float denominator = n_u * (n_u - 1.0f);
          temp[threadIdx.x] = numerator / denominator;
     }

     __syncthreads();

     if (threadIdx.x == 0)
     {
          float sum = 0.0f;
          for (uint i = 0; i < threads_per_block; i++)
          {
               uint current_index = u_index + i;
               if ( current_index < num_nodes)
               {
                    sum += temp[i];
               }
          }
          atomicAdd(d_sum, sum);

     }
}

uint NUM_NODES;

float* allocate_adj_matrix(uint num_nodes);
float* read_graph(char filename[]);
float clustering_coefficient(const uint u, const float* adj_matrix, const uint num_nodes);


float* allocate_adj_matrix(uint num_nodes)
{
     uint size = num_nodes * num_nodes;
     float* graph = new float[size];
     for (uint i = 0; i < size; i++)
     {
          graph[i] = 0;
     }
     return graph;
}


float* read_graph(char filename[])
{
     fstream f(filename, std::ios_base::in);
     uint u,v;
     vector<pair<float,float> > all_edges;
     uint max_node = 0;
     while (f >> u >> v)
     {
          all_edges.push_back(make_pair(u,v));
          if (u > max_node)
          {
               max_node = u;
          }
          if (v > max_node)
          {
               max_node = v;
          }
     }
     f.close();
     NUM_NODES = max_node + 1;
     float* graph = allocate_adj_matrix(NUM_NODES);
     for (uint i = 0; i < all_edges.size(); i++)
     {
          u = all_edges[i].first;
          v = all_edges[i].second;
          graph[u * NUM_NODES + v] = 1.0f;
          graph[v * NUM_NODES + u] = 1.0f;
     }
     return graph;
}


/**
 * Host main routine
 */
int main(int argc, char* argv[]){
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    float* h_adj_matrix = read_graph(argv[1]);
    const uint num_nodes = NUM_NODES;

    // Verify that allocations succeeded
    if (h_adj_matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    size_t size = num_nodes * num_nodes * sizeof(float);
    // Allocate the device input vector A
    float* d_adj_matrix = NULL;
    err = cudaMalloc((void **)&d_adj_matrix, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_adj_matrix, h_adj_matrix, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    uint max_blocks = prop.maxGridSize[0];
    uint blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    // hardware limit
    if (blocks > max_blocks)
    {
         blocks = max_blocks;
    }

    float* d_sum = NULL;
    err = cudaMalloc((void **)&d_sum, sizeof(float));
    if (err != cudaSuccess)
    {
         fprintf(stderr, "Failed to allocate for a global variable (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }
    cudaMemset(d_sum, 0, sizeof(float));

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks, threads_per_block);
    clustering_coefficient<<<blocks, threads_per_block>>>(d_adj_matrix, num_nodes, d_sum);
    if (err != cudaSuccess)
    {
         fprintf(stderr, "Failed to launch vectorDot kernel (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
    }

    float h_sum;
    printf("Copy the CUDA device to the host memory\n");
    err = cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float C_G = h_sum / num_nodes;

    // Free device global memory
    err = cudaFree(d_adj_matrix);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_sum);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cout << C_G << endl;

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

