#include "black_scholes.h"
#include "gaussian.h"
#include "random.h" 
#include "util.h"
#include <omp.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double
black_scholes_stddev (void* the_args)
{

  black_scholes_args_t* args = (black_scholes_args_t*) the_args;
  const double mean = args->mean;
  const int M = args->M;
  double variance = 0.0;
  int k;


   #pragma acc parallel loop reduction(+:variance) 
  for (k = 0; k < M; k++)
    {
      const double diff = args->trials[k] - mean;
      variance += diff * diff / (double) M;
    }

  args->variance = variance;
  return sqrt (variance);
}


static void* black_scholes_iterate (void* the_args)
{

  black_scholes_args_t* args = (black_scholes_args_t*) the_args;

  const int S = args->S;
  const int E = args->E;
  const int M = args->M;
  const double r = args->r;
  const double sigma = args->sigma;
  const double T = args->T;

  double* trials = args->trials;
  double mean = 0.0;

  gaussrand_state_t gaussrand_state;
  void* prng_stream = NULL; 
  int k;


double *randnumbs;
randnumbs = malloc(4* M * sizeof (double));

init_gaussrand_state (&gaussrand_state);

for (int i = 0; i < M; i++)
{
  prng_stream = spawn_prng_stream(i%4);
  const double gaussian_random_number = gaussrand1 (&uniform_random_double, prng_stream, &gaussrand_state);
  randnumbs[i]=gaussian_random_number;
}

  
 #pragma acc parallel loop reduction(+:variance) 
for (k = 0; k < M; k++)
    {
     const double current_value = S * exp ( (r - (sigma*sigma) / 2.0) * T + sigma * sqrt (T) * randnumbs[k]);
     trials[k] = exp (-r * T) * ((current_value - E < 0.0) ? 0.0 : current_value - E);
      mean += trials[k] / (double) M;
    }

  args->mean = mean;
  free_prng_stream (prng_stream);
  return NULL;
}




void black_scholes (confidence_interval_t* interval,
	       const double S,
	       const double E,
	       const double r,
	       const double sigma,
	       const double T,
	       const int M,
         const int n)
{
  black_scholes_args_t args;
  double mean = 0.0;
  double stddev = 0.0;
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

  (void) black_scholes_iterate (&args);
  mean = args.mean;
  stddev = black_scholes_stddev (&args);

  conf_width = 1.96 * stddev / sqrt ((double) M);
  interval->min = mean - conf_width;
  interval->max = mean + conf_width;

  deinit_black_scholes_args (&args);
}


void
deinit_black_scholes_args (black_scholes_args_t* args)
{
  if (args != NULL)
    if (args->trials != NULL)
      {
	free (args->trials);
	args->trials = NULL;
      }
}

