#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timers.h"
#include <iostream>
#include <hip/hip_runtime.h>

#define COUNT_MAX   2000
#define MIN(x,y)    ((x) < (y) ? (x) : (y))

#ifdef SAVE_JPG
void save_jpeg_image(const char* filename, int* r, int* g, int* b, int image_width, int image_height);
#endif


#define CHECK_HIP(error) checkHipError((error), __FILE__, __LINE__)
inline void checkHipError(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(result) << " at " << file << ":" << line << std::endl;
        exit(result);
    }
}

typedef struct {
  unsigned int r;
  unsigned int g;
  unsigned int b;
} RgbColor;


__device__ void HSVtoRGB(RgbColor *rgb, unsigned h, unsigned s, unsigned v);

#define numStreams 8
static int *d_r, *d_g, *d_b;
static int offset_list[numStreams];
static hipStream_t running_stream_list[numStreams], copy_stream_list[3];

// Main part of the below code is originated from Lode Vandevenne's code.
// Please refer to http://lodev.org/cgtutor/juliamandelbrot.html
__global__ void julia_kernel(int *r, int *g, int *b, int w, int h, double cRe, double cIm, double zoom, double moveX, double moveY, int maxIterations, int offset) {
  int x_index_global = blockIdx.x * blockDim.x + threadIdx.x;
  int y_index_global = blockIdx.y * blockDim.y + threadIdx.y + offset;

  int x = x_index_global;
  int y = y_index_global;
  // real and imaginary parts of new and old
  double newRe, newIm, oldRe, oldIm;

  newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
  newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;
  // i will represent the number of iterations
  int i;
  //start the iteration process
  for (i = 0; i < maxIterations; i++)
  {
    // remember value of previous iteration
    oldRe = newRe;
    oldIm = newIm;
    // the actual iteration, the real and imaginary part are calculated
    newRe = oldRe * oldRe - oldIm * oldIm + cRe;
    newIm = 2 * oldRe * oldIm + cIm;
    // if the point is outside the circle with radius 2: stop
    if ((newRe * newRe + newIm * newIm) > 4) break;
  }
  // use color model conversion to get rainbow palette, 
  // make brightness black if maxIterations reached
  RgbColor color;
  HSVtoRGB(&color, i % 256, 255, 255 * (i < maxIterations));
  
  r[y*w+x] = color.r;
  g[y*w+x] = color.g;
  b[y*w+x] = color.b;     

}


void julia_gpu_stream(int *r, int *g, int *b, int w, int h, double cRe, double cIm, double zoom, double moveX, double moveY, int maxIterations) {
  
  
  for (int i = 0; i<numStreams; i++) {
    dim3 block(16, 16);
    dim3 grid;
    if (i == numStreams - 1 && h % numStreams != 0) {
      grid = dim3((w + block.x - 1) / block.x, ( h % numStreams + block.y - 1) / block.y);
    } else {
      grid = dim3((w + block.x - 1) / block.x, (h / numStreams + block.y - 1) / block.y);
    }

    julia_kernel<<<grid, block, 0, running_stream_list[i]>>>(d_r, d_g, d_b, w, h, cRe, cIm, zoom, moveX, moveY, maxIterations, offset_list[i]);
  }
  CHECK_HIP(hipDeviceSynchronize());
  
  // copy data back to host
  CHECK_HIP(hipMemcpyAsync(r, d_r, w * h * sizeof(int), hipMemcpyDeviceToHost, copy_stream_list[0]));
  CHECK_HIP(hipMemcpyAsync(g, d_g, w * h * sizeof(int), hipMemcpyDeviceToHost, copy_stream_list[1]));
  CHECK_HIP(hipMemcpyAsync(b, d_b, w * h * sizeof(int), hipMemcpyDeviceToHost, copy_stream_list[2]));
}

void julia(int w, int h, char* output_filename) {
  // each iteration, it calculates: new = old*old + c,
  // where c is a constant and old starts at current pixel

  // real and imaginary part of the constant c
  // determinate shape of the Julia Set
  double cRe, cIm;

  // you can change these to zoom and change position
  double zoom = 1, moveX = 0, moveY = 0;

  // after how much iterations the function should stop
  int maxIterations = COUNT_MAX;

#ifndef SAVE_JPG
  FILE *output_unit;
#endif

  double wtime;

  // pick some values for the constant c
  // this determines the shape of the Julia Set
  cRe = -0.7;
  cIm = 0.27015;

  int *r, *g, *b;
  CHECK_HIP(hipHostMalloc((void**)&r, w * h * sizeof(int), hipHostMallocMapped));
  CHECK_HIP(hipHostMalloc((void**)&g, w * h * sizeof(int), hipHostMallocMapped));
  CHECK_HIP(hipHostMalloc((void**)&b, w * h * sizeof(int), hipHostMallocMapped));
  // synchronize the device
  CHECK_HIP(hipDeviceSynchronize());

  // device init
  CHECK_HIP(hipMalloc((void**)&d_r, w * h * sizeof(int)));
  CHECK_HIP(hipMalloc((void**)&d_g, w * h * sizeof(int)));
  CHECK_HIP(hipMalloc((void**)&d_b, w * h * sizeof(int)));

  // stream init
  for (size_t i = 0; i<numStreams; i++) {
    CHECK_HIP(hipStreamCreate(&running_stream_list[i]));
    CHECK_HIP(hipStreamCreate(&copy_stream_list[i]));
  }

  
  // create offset list 
  for (int i = 0; i<numStreams; i++) {
    offset_list[i] = i * h / numStreams;
  }
  if (h % numStreams != 0) {
    offset_list[numStreams - 1] = h - (h / numStreams) * (numStreams - 1);
  }



  printf( "  Sequential C version\n" );
  printf( "\n" );
  printf( "  Create an ASCII PPM image of the Julia set.\n" );
  printf( "\n" );
  printf( "  An image of the set is created using\n" );
  printf( "    W = %d pixels in the X direction and\n", w );
  printf( "    H = %d pixels in the Y direction.\n", h );




  timer_init();
  timer_start(0);

  julia_gpu_stream(r, g, b, w, h, cRe, cIm, zoom, moveX, moveY, maxIterations);

  timer_stop(0);
  wtime = timer_read(0);
  printf( "\n" );
  printf( "  Time = %lf seconds.\n", wtime );

  // clean device 
  CHECK_HIP(hipFree(d_r));
  CHECK_HIP(hipFree(d_g));
  CHECK_HIP(hipFree(d_b));

  // clean stream
  for (size_t i = 0; i<numStreams; i++) {
    CHECK_HIP(hipStreamSynchronize(running_stream_list[i]));
    CHECK_HIP(hipStreamSynchronize(copy_stream_list[i]));
    CHECK_HIP(hipStreamDestroy(running_stream_list[i]));
    CHECK_HIP(hipStreamDestroy(copy_stream_list[i]));
  }

#ifdef SAVE_JPG
  save_jpeg_image(output_filename, r, g, b, w, h);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen( output_filename, "wt" );

  fprintf( output_unit, "P3\n" );
  fprintf( output_unit, "%d  %d\n", h, w );
  fprintf( output_unit, "%d\n", 255 );
  for ( int i = 0; i < h; i++ )
  {
    for ( int jlo = 0; jlo < w; jlo = jlo + 4 )
    {
      int jhi = MIN( jlo + 4, w );
      for ( int j = jlo; j < jhi; j++ )
      {
        fprintf( output_unit, "  %d  %d  %d", r[i*w+j], g[i*w+j], b[i*w+j] );
      }
      fprintf( output_unit, "\n" );
    }
  }

  fclose( output_unit );
#endif
  printf( "\n" );
  printf( "  Graphics data written to \"%s\".\n\n", output_filename );

  // Terminate.
  CHECK_HIP(hipHostFree(r));
  CHECK_HIP(hipHostFree(g));
  CHECK_HIP(hipHostFree(b));
}

__device__ void HSVtoRGB(RgbColor *rgb, unsigned h, unsigned s, unsigned v){
  unsigned char region, remainder, p, q, t;

  if (s == 0)
  {
    rgb->r = v;
    rgb->g = v;
    rgb->b = v;
  } else {
    region = h / 43;
    remainder = (h - (region * 43)) * 6; 

    p = (v * (255 - s)) >> 8;
    q = (v * (255 - ((s * remainder) >> 8))) >> 8;
    t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

    switch (region)
    {
      case 0:
        rgb->r = v; rgb->g = t; rgb->b = p;
        break;
      case 1:
        rgb->r = q; rgb->g = v; rgb->b = p;
        break;
      case 2:
        rgb->r = p; rgb->g = v; rgb->b = t;
        break;
      case 3:
        rgb->r = p; rgb->g = q; rgb->b = v;
        break;
      case 4:
        rgb->r = t; rgb->g = p; rgb->b = v;
        break;
      default:
        rgb->r = v; rgb->g = p; rgb->b = q;
        break;
    }
  }
}
