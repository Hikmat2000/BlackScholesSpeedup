#include "black_scholes.h"
#include "parser.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>


int main (int argc, char* argv[])
{
  confidence_interval_t interval;
  double S, E, r, sigma, T;
  int M = 0;
  char* filename = NULL;
  int nthreads = 1;
  double t1, t2;
  
  if (argc < 3)
    {
      fprintf (stderr, 
	       "Usage: ./hw1.x <filename> <nthreads>\n\n");
      exit (EXIT_FAILURE);
    }
  filename = argv[1];
  nthreads = to_int (argv[2]);
  parse_parameters (&S, &E, &r, &sigma, &T, &M, filename);

  
  myfunction(&interval, S, E, r, sigma, T, M, nthreads);

  printf ("Black-Scholes benchmark:\n"
	  "------------------------\n"
	  "S        %g\n"
	  "E        %g\n"
	  "r        %g\n"
	  "sigma    %g\n"
	  "T        %g\n"
	  "M        %d\n",
	  S, E, r, sigma, T, M);
  printf ("Confidence interval: (%g, %g)\n", interval.min, interval.max);

  return 0;
}



