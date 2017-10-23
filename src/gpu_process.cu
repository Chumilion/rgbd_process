#include <rgbd_process/gpu_process.hu>

namespace gproc
{
  const unsigned int THREADS_PER_BLOCK_1D = 16u;
  const unsigned int THREADS_PER_LINEBLOCK = 512u;

  void resizeAndMerge(float* target_ptr, const float* const source_ptr,
                      const int target_width, const int target_height,
                      const int source_width, const int source_height, const int channels)
  {
    //Resizing heat maps
    const dim3 threads_per_block(THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D);
    const dim3 num_blocks(get_num_blocks(target_width, threads_per_block.x),
                          get_num_blocks(target_height, threads_per_block.y));
    const long source_channel_offset = source_height*source_width;
    const long target_channel_offset = target_height*target_width;

    for(int c = 0; c < channels; c++)
    {
      resizeKernel<<<num_blocks, threads_per_block>>>(target_ptr + c*target_channel_offset,
                                                      source_ptr + c*source_channel_offset,
                                                      target_width, target_height,
                                                      source_width, source_height);
    }
  }

  __global__ void resizeKernel(float* target_ptr, const float* const source_ptr,
                               const int target_width, const int target_height,
                               const int source_width, const int source_height)
  {
    //Figure out (x, y) size conversion, then bicubic interpolate
    const int x = (blockIdx.x*blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y*blockDim.y) + threadIdx.y;

    if(x < target_width && y < target_height)
    {
      const float scale_width = target_width/float(source_width);
      const float scale_height = target_height/float(source_height);
      const float x_source = (x + 0.5f)/scale_width - 0.5f;
      const float y_source = (y + 0.5f)/scale_height - 0.5f;

      target_ptr[y*target_width + x] = 
        bicubicInterpolate(source_ptr, x_source, y_source, source_width, source_height);
    }
  }

  void findPeaks(float* target_ptr, int* marking_ptr, const float* const source_ptr,
                 const int source_width, const int source_height, const int channels,
                 const int max_peaks, const float threshold)
  {
    //Finds local maxima per channel
    const long image_offset = source_height*source_width;
    const int target_offset = (max_peaks + 1);

    const dim3 threads_per_2Dblock(THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D);
    const dim3 num_2Dblocks(get_num_blocks(source_width, threads_per_2Dblock.x),
                            get_num_blocks(source_height, threads_per_2Dblock.y));
    const dim3 threads_per_1Dblock(THREADS_PER_LINEBLOCK);
    const dim3 num_1Dblocks(get_num_blocks(image_offset, threads_per_1Dblock.x));

    int* target_count;
    cudaMalloc((void **)&target_count, sizeof(int));

    for(int c = 0; c < channels; c++)
    {
      //Figure out offset
      int* marking_ptr_offsetted = marking_ptr + c*image_offset;
      const float* const source_ptr_offsetted = source_ptr + c*image_offset;
      float* target_ptr_offsetted = target_ptr + c*target_offset;

      markPeaks<<<num_2Dblocks, threads_per_2Dblock>>>(marking_ptr_offsetted,
        source_ptr_offsetted, source_width, source_height, threshold);

      cudaMemset(target_count, 0, sizeof(int));
      writePeaks<<<num_1Dblocks, threads_per_1Dblock>>>(target_ptr_offsetted,
        marking_ptr_offsetted, source_ptr_offsetted, target_count, image_offset,
        source_width, source_height, max_peaks);

      cudaMemcpy(target_ptr_offsetted, target_count, sizeof(float),
                 cudaMemcpyDeviceToDevice);
    }
  }

  __global__ void markPeaks(int* marking_ptr, const float* const source_ptr,
                            const int width, const int height, const float threshold)
  {
    //Compares with neighbors, marks with 1;
    const int x = (blockIdx.x*blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y*blockDim.y) + threadIdx.y;
    const long index = y*width + x;

    if(0 < x && x < (width - 1) && 0 < y && y < (height - 1))
    {
      const float value = source_ptr[index];
      if(value > threshold)
      {
        const int neighbors[8] = {index - width - 1, index - width, index - width + 1,
                                  index - 1, index + 1, index + width - 1,
                                  index + width, index + width + 1};
        for(int n = 0; n < 8; n++)
        {
          if(value <= source_ptr[neighbors[n]])
          {
            marking_ptr[index] = 1;
            return;
          }
        }
      }
      marking_ptr[index] = 0;
    }
    else if(x == 0 || x == (width - 1) || y == 0 || y == (height - 1))
      marking_ptr[index] = 0;
  }

  __global__ void writePeaks(float* target_ptr, const int* const marking_ptr,
                             const float* const source_ptr, int* target_count,
                             const int length, const int width, const int height,
                             const int max_peaks)
  {
    //Uses atomic adds to assign an index to peaks in parallel, then stores peak info
    //at that index
    const long index = blockIdx.x*blockDim.x + threadIdx.x;

    if(index < length && marking_ptr[index] == 1)
    {
      int i = atomicAdd(target_count, 1);
      if(i < max_peaks)
      {
        const int loc_x = (int)(index%width);
        const int loc_y = (int)(index/width);
        
        float x_acc = 0.f;
        float y_acc = 0.f;
        float score_acc = 0.f;
  
        for(int dy = -3; dy < 4; dy++)
        {
          const int y = loc_y + dy;
          if(0 <= y && y < height)
          {
            for(int dx = -3; dx < 4; dx++)
            {
              const int x = loc_x + dx;
              if( 0 <= x && x < width)
              {
                const float score = source_ptr[y*width + x];
                if(score > 0)
                {
                  x_acc += x*score;
                  y_acc += y*score;
                  score_acc += score;
                }
              }
            }
          }
        }
        const int target_index = (i + 1)*3;
        target_ptr[target_index] = x_acc/score_acc;
        target_ptr[target_index + 1] = y_acc/score_acc;
        target_ptr[target_index + 2] = source_ptr[loc_y*width + loc_x];
      }
    }
  }
}
