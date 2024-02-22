#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timers.h"
#include <hip/hip_runtime.h>
#include <iostream>

#define COUNT_MAX   5000
#define MIN(x,y)    ((x) < (y) ? (x) : (y))


#define CHECK_HIP(error) checkHipError((error), __FILE__, __LINE__)
inline void checkHipError(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(result) << " at " << file << ":" << line << std::endl;
        exit(result);
    }
}


#define numStreams 8
static int *d_r, *d_g, *d_b;

static int offset_list[numStreams];
static hipStream_t run_kernel_stream_list[numStreams];
static hipStream_t copy_stream_list[numStreams];

#ifdef SAVE_JPG
void save_jpeg_image(const char* filename, int* r, int* g, int* b, int image_width, int image_height);
#endif

__global__ void mandelbrot_kernel(int *r, int *g, int *b, int m, int n, float x_max, float x_min, float y_max, float y_min, int count_max, int offset) {
  int x_index_global = blockIdx.x * blockDim.x + threadIdx.x;
  int y_index_global = blockIdx.y * blockDim.y + threadIdx.y + offset;

  float x, x1, x2;
  float y, y1, y2;
  int c, k;

  int j = x_index_global;
  int i = y_index_global;

  x = ( ( float ) (     j - 1 ) * x_max  
      + ( float ) ( m - j     ) * x_min )
      / ( float ) ( m     - 1 );

  y = ( ( float ) (     i - 1 ) * y_max  
      + ( float ) ( n - i     ) * y_min )
      / ( float ) ( n     - 1 );

  int count = 0;

  x1 = x;
  y1 = y;

  for ( k = 1; k <= count_max; k++ )
  {
    x2 = x1 * x1 - y1 * y1 + x;
    y2 = 2.0 * x1 * y1 + y;

    if ( x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2 )
    {
      count = k;
      break;
    }
    x1 = x2;
    y1 = y2;
  }

  if ( ( count % 2 ) == 1 )
  {
    r[i * n + j] = 255;
    g[i * n + j] = 255;
    b[i * n + j] = 255;
  }
  else
  {
    c = ( int ) ( 255.0 * sqrtf ( sqrtf ( sqrtf (
      ( ( float ) ( count ) / ( float ) ( count_max ) ) ) ) ) );
    r[i * n + j] = 3 * c / 5;
    g[i * n + j] = 3 * c / 5;
    b[i * n + j] = c;
  }
}

void mandelbrot_gpu_stream(int *r, int *g, int *b, int m, int n, float x_max, float x_min, float y_max, float y_min, int count_max) {
  for (int i = 0; i<numStreams; i++) {
    dim3 block(16, 16);
    dim3 grid;
    if (i == numStreams - 1 && m % numStreams != 0) {
      grid = dim3((n + block.x - 1) / block.x, ( m % numStreams + block.y - 1) / block.y);
    } else {
      grid = dim3((n + block.x - 1) / block.x, (m / numStreams + block.y - 1) / block.y);
    }
    mandelbrot_kernel<<<grid, block, 0, run_kernel_stream_list[i]>>>(d_r, d_g, d_b, m, n, x_max, x_min, y_max, y_min, count_max, offset_list[i]);
  }
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpyAsync(r, d_r, m * n * sizeof(int), hipMemcpyDeviceToHost, copy_stream_list[0]));
  CHECK_HIP(hipMemcpyAsync(g, d_g, m * n * sizeof(int), hipMemcpyDeviceToHost, copy_stream_list[1]));
  CHECK_HIP(hipMemcpyAsync(b, d_b, m * n * sizeof(int), hipMemcpyDeviceToHost, copy_stream_list[2]));
}


void mandelbrot(int m, int n, char* output_filename) {
  // int c;
  int count_max = COUNT_MAX;
  // int i, j, k;
#ifndef SAVE_JPG
  int jhi, jlo;
  FILE *output_unit;
#endif
  double wtime;

  float x_max =   1.25;
  float x_min = - 2.25;
  float y_max =   1.75;
  float y_min = - 1.75;

  int *r, *g, *b;

  CHECK_HIP(hipHostMalloc((void**)&r, m * n * sizeof(int), hipHostMallocMapped));
  CHECK_HIP(hipHostMalloc((void**)&g, m * n * sizeof(int), hipHostMallocMapped));
  CHECK_HIP(hipHostMalloc((void**)&b, m * n * sizeof(int), hipHostMallocMapped));

  // wait for the memory to be allocated
  CHECK_HIP(hipDeviceSynchronize());

  // device initialization
  CHECK_HIP(hipMalloc((void**)&d_r, m * n * sizeof(int)));
  CHECK_HIP(hipMalloc((void**)&d_g, m * n * sizeof(int)));
  CHECK_HIP(hipMalloc((void**)&d_b, m * n * sizeof(int)));

  // stream initialization
  for (size_t i = 0; i<numStreams; i++) {
    CHECK_HIP(hipStreamCreate(&run_kernel_stream_list[i]));
    CHECK_HIP(hipStreamCreate(&copy_stream_list[i]));
  }

  // set the offsets
  for (size_t i = 0; i<numStreams; i++) {
    offset_list[i] = i * (m / numStreams);
  }
  if (m % numStreams != 0) {
    offset_list[numStreams - 1] = m - (m / numStreams) * (numStreams - 1);
  }

  printf( "  Sequential C version\n" );
  printf( "\n" );
  printf( "  Create an ASCII PPM image of the Mandelbrot set.\n" );
  printf( "\n" );
  printf( "  For each point C = X + i*Y\n" );
  printf( "  with X range [%g,%g]\n", x_min, x_max );
  printf( "  and  Y range [%g,%g]\n", y_min, y_max );
  printf( "  carry out %d iterations of the map\n", count_max );
  printf( "  Z(n+1) = Z(n)^2 + C.\n" );
  printf( "  If the iterates stay bounded (norm less than 2)\n" );
  printf( "  then C is taken to be a member of the set.\n" );
  printf( "\n" );
  printf( "  An image of the set is created using\n" );
  printf( "    M = %d pixels in the X direction and\n", m );
  printf( "    N = %d pixels in the Y direction.\n", n );

  timer_init();
  timer_start(0);
  mandelbrot_gpu_stream(r, g, b, m, n, x_max, x_min, y_max, y_min, count_max);
  timer_stop(0);
  
  wtime = timer_read(0);
  printf( "\n" );
  printf( "  Time = %lf seconds.\n", wtime );

  // destroy the device memory
  CHECK_HIP(hipFree(d_r));
  CHECK_HIP(hipFree(d_g));
  CHECK_HIP(hipFree(d_b));

  // destroy the streams
  for (size_t i = 0; i<numStreams; i++) {
    CHECK_HIP(hipStreamDestroy(run_kernel_stream_list[i]));
    CHECK_HIP(hipStreamDestroy(copy_stream_list[i]));
  }

#ifdef SAVE_JPG
  // Write data to an JPEG file.
  save_jpeg_image(output_filename, r, g, b, n, m);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen( output_filename, "wt" );

  fprintf( output_unit, "P3\n" );
  fprintf( output_unit, "%d  %d\n", n, m );
  fprintf( output_unit, "%d\n", 255 );
  for ( i = 0; i < m; i++ )
  {
    for ( jlo = 0; jlo < n; jlo = jlo + 4 )
    {
      jhi = MIN( jlo + 4, n );
      for ( j = jlo; j < jhi; j++ )
      {
        fprintf( output_unit, "  %d  %d  %d", r[i * n + j], g[i * n + j], b[i * n + j] );
      }
      fprintf( output_unit, "\n" );
    }
  }
  fclose( output_unit );
#endif

  printf( "\n" );
  printf( "  Graphics data written to \"%s\".\n\n", output_filename );

  CHECK_HIP(hipHostFree(r));
  CHECK_HIP(hipHostFree(g));
  CHECK_HIP(hipHostFree(b));
}

