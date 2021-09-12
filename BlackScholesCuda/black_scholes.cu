#include "black_scholes.h"
#include "util.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

__managed__ double stddev;

__global__ void black_scholes_stddev (void* the_args)
{

  black_scholes_args_t* args = (black_scholes_args_t*) the_args;
  const double mean = args->mean;
  const int M = args->M;
  double variance = 0.0;
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if(k<M)
  {
   const double diff = args->trials[k] - mean;
   variance += diff * diff / (double) M;
  }

  args->variance = variance;
  stddev=sqrt(variance);

}


__global__ void black_scholes_iterate (void* the_args)
{

  black_scholes_args_t* args = (black_scholes_args_t*) the_args;

  const int S = args->S;
  const int E = args->E;
  const int M = args->M;
  const double r = args->r;
  const double sigma = args->sigma;
  const double T = args->T;
 // curandStateMtgp32* state=args->state;
  double* trials = args->trials;
  double mean = 0.0;
  //double* devResults=args

int i = blockIdx.x * blockDim.x + threadIdx.x;
int k = blockIdx.x * blockDim.x + threadIdx.x;
if(i<M)
{
 //  devResults[i] = curand(&state[blockIdx.x]);
  const double current_value = S * exp ( (r - (sigma*sigma) / 2.0) * T + sigma * sqrt (T) *1);
  trials[k] = exp (-r * T) * ((current_value - E < 0.0) ? 0.0 : current_value - E);
   mean += trials[k]/ (double) M; 
   __syncthreads();
  args->mean = mean;
}  
}



void myfunction(confidence_interval_t* interval,
   const double S, const double E, const double r, const double sigma, const double T, const int M,const int n)
{

  black_scholes_args_t args;
  double mean = 0.0;
  double conf_width = 0.0;
  double* trials = NULL;

  assert (M > 0);
  trials = (double*) malloc (M * sizeof (double));
  assert (trials != NULL);

  args.S = S;
  args.E = E;
  args.r = r;
  args.sigma = sigma;
  args.T = T;
  args.M = M;
  args.trials = trials;
  args.mean = 0.0;
  args.variance = 0.0;

    printf("00");

  curandStateMtgp32 *devMTGPStates;
  mtgp32_kernel_params *devKernelParams;

    printf("0");
  int *hostResults;
  int probsize=M;
  int nthreads=n;
  int nblocks=(probsize/nthreads)+1;

  printf("1");
  
  hostResults = (int *)calloc(nblocks * nthreads, sizeof(int));
    printf("2");
  cudaMalloc((void **)&devResults, nblocks * nthreads * sizeof(int));
  cudaMemset(devResults, 0, nblocks * nthreads *  sizeof(int));
  cudaMalloc((void **)&devMTGPStates, nthreads * sizeof(curandStateMtgp32));
  cudaMalloc((void**)&devKernelParams, sizeof(mtgp32_kernel_params));
    printf("3");
  curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
  curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, nthreads, 1234);

     printf("4"); 
  //args.devResults=devResults;
  //args.state=devMTGPStates;


  printf("5");
  (void)black_scholes_iterate<<<nblocks,nthreads>>>(&args);
    printf("6");
  mean = args.mean;
    printf("7");
  cudaDeviceSynchronize();
  black_scholes_stddev<<<nblocks,nthreads>>> (&args);
    printf("8");
  cudaDeviceSynchronize();
  conf_width = 1.96 * stddev / sqrt ((double) M);
    printf("9");
  interval->min = mean - conf_width;
    printf("10");
  interval->max = mean + conf_width;
    printf("11");
   cudaMemcpy(hostResults, devResults, nblocks*blocks * sizeof(int), cudaMemcpyDeviceToHost);
   cudaFree(devMTGPStates);
   free(hostResults);
   cudaFree(devResults);
   deinit_black_scholes_args (&args);
}


void deinit_black_scholes_args (black_scholes_args_t* args)
{
  if (args != NULL)
    if (args->trials != NULL)
      {
	free (args->trials);
	args->trials = NULL;
      }
}

