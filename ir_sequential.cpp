#include "OpenCLSetup.hpp"
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <algorithm>

#ifndef queen
#define queen cl_int
#endif
/*
#define BOARDSIZE 100
#define MAX_ITERS 100000
*/
int BOARDSIZE;
int MAX_ITERS;

using namespace std;

queen * random_board(int BOARDSIZE); //Allocates space
void print_board(queen * q, int BOARDSIZE);

int main(int argc, char *argv[]){
  //int seed = time(NULL);
  int seed = 1337626291;
  srand(seed);
  printf("Seed value: %d\n", seed);

  int BOARDSIZE, MAX_ITERS;
  std::cout << "argc: " << argc << endl;
  //std::cout << "argv[1]: " << argv[1] << endl;
  //std::cout << "argv[2]: " << argv[2] << endl;
  if ((argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) || argc == 1) 
  {
    std::cout << "Usage: ir_sequential BOARDSIZE MAX_ITERS" << std::endl;
    exit(0);
  }

  if ( argc == 3 )
  {
    BOARDSIZE = atoi(argv[1]);
    MAX_ITERS = atoi(argv[2]);
  }

  std::cout << "BOARDSIZE: " << BOARDSIZE << std::endl;
  std::cout << "MAX_ITERS: " << MAX_ITERS << std::endl;
  queen * queens = random_board(BOARDSIZE),* curr_q;
  //queen conflicts[BOARDSIZE] = {0};
  queen conflicts[BOARDSIZE];
  memset(conflicts, 0, BOARDSIZE*sizeof(int));
  //queen zero[BOARDSIZE] = {0};
  queen zero[BOARDSIZE];
  memset(zero, 0, BOARDSIZE*sizeof(int));
  int curr = 0,cf_iters = 0,iters = 0,min_con,min_c;
  int nqueens = BOARDSIZE;
  int event_id = 0;
  OpenCLWrapper w;
  w.enableProfiling = true;
 
  print_board(queens, BOARDSIZE);
  printf("\n");
  fflush(stdout);

  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};
 
  try{
    
    w.createContext();
    w.createCommandQueue();
    w.createProgram("sequential.cl");
    
    w.createKernel("seq_solve");
    
    //Queen array
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
				  sizeof(queen)*BOARDSIZE*2,
				  queens,NULL));
    //Iters taken
    w.addMemObject(clCreateBuffer(w.context,
				  CL_MEM_READ_WRITE,
				  sizeof(queen),
				  NULL,NULL));
    
    w.check(clSetKernelArg(w.kernels["seq_solve"],0,sizeof(cl_mem),
			   &w.memObjects[0]), "Error setting kernel arg 0");

    w.check(clSetKernelArg(w.kernels["seq_solve"],1,sizeof(cl_int),
			   &nqueens), "Error setting kernel arg 1");

    w.check(clSetKernelArg(w.kernels["seq_solve"],2,sizeof(cl_int),
          &MAX_ITERS), "Error setting kernel arg 2");
    
    w.check(clSetKernelArg(w.kernels["seq_solve"],3,sizeof(cl_int),
          &seed), "Error setting kernel arg 3");

    w.check(clSetKernelArg(w.kernels["seq_solve"],4,sizeof(cl_mem),
			   &w.memObjects[1]), "Error setting kernel arg 4");



    printf("Max iterations: %d\n", MAX_ITERS);
    fflush(stdout);

    w.check(clEnqueueWriteBuffer(w.commandQueue,w.memObjects[0],
          CL_FALSE,0,2*BOARDSIZE*sizeof(queen),
          queens,0,NULL,NULL),
          "Error enqueueing write buffer");
    
    w.check(clEnqueueNDRangeKernel(w.commandQueue,
          w.kernels["seq_solve"],
          1,0,globalWorkSize,
          localWorkSize,0,NULL,&w.events[event_id++]),
          "Error enqueueing kernel");
    
    w.check(clEnqueueReadBuffer(w.commandQueue,w.memObjects[0],CL_TRUE,0,
          BOARDSIZE*2*sizeof(cl_int),queens,0,NULL,
          NULL),
          "Error enqueueing read buffer");

    w.check(clEnqueueReadBuffer(w.commandQueue,w.memObjects[1],CL_TRUE,0,
          sizeof(cl_int),&iters,0,NULL,
          &w.events[event_id++]),
          "Error enqueueing read buffer");    

    std::cout << "Final board: " << std::endl;
    print_board(queens, BOARDSIZE);
    fflush(stdout);
    
    std::cout << "Solved in " << iters << " iters" << std::endl;
    fflush(stdout);
    
    cl_ulong run_start;
    cl_ulong run_end;
    cl_ulong total = 0;
    w.check(clGetEventProfilingInfo(w.events[0], CL_PROFILING_COMMAND_START,
          sizeof(cl_ulong), &run_start, NULL), 
        "Error getting event profile information");
    w.check(clGetEventProfilingInfo(w.events[0], CL_PROFILING_COMMAND_END,
          sizeof(cl_ulong), &run_end, NULL),
        "Error getting event profile information");

    std::cout << "Profiling time sequential execution on GPU:" << ": " << (run_end - run_start) << std::endl;
    std::cout << "Average time/iter:" << ": " << (run_end - run_start)/iters << std::endl;


    /*
       print_board(queens);
       for(int i(0); i<BOARDSIZE; i++){
       printf("%i,",conflicts[i]);
       }
       printf("\n");
       */

  }
  catch ( runtime_error& e) {
    std::cerr << e.what() << std::endl;
    w.cleanup();
    return 1;    
  }   

  //printf("Iters: %i\n",iters);

}

queen * random_board(int BOARDSIZE){
  vector<queen> b(BOARDSIZE);
  for(int i(0); i<BOARDSIZE; i++){b[i] = i;}
  random_shuffle(b.begin(),b.end());

  queen * queens = new queen[2*BOARDSIZE];
  for(int q(0), j(0); q<BOARDSIZE; q++, j+=2){
    queens[j] = q;
    queens[j+1] = b[q];
  }

  return queens;
}


void print_board(queen * queens, int BOARDSIZE){
  fflush(stdout);
  char tmp[BOARDSIZE][BOARDSIZE+1];
  memset(tmp,'.',BOARDSIZE*(BOARDSIZE+1));

  for(int i(0); i<BOARDSIZE; i++){
    tmp[queens[2*i]][queens[2*i+1]] = 'Q';
    tmp[i][BOARDSIZE] = '\0';
  }

  for(int i(0); i<BOARDSIZE; i++){
    printf("%s\n",&tmp[i]);
  } 
}
