// -*- c++ -*-
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#ifndef queen
#define queen int
#endif

/*
__kernel void count_conflicts(const __global queen * queens,
			      __global queen * result,
			      __local queen * ws,
			      const int row, const int nqueens, 
			      const int group_size ){

  int q = get_global_id(0);
  int col = get_global_id(1);

  int i,j,conflicts = 0,curr[2];

  curr[0] = queens[2*q];
  curr[1] = queens[2*q+1];

  conflicts += ((row == curr[0]) || (col == curr[1]) || 
		(abs_diff(row,curr[0]) == abs_diff(col,curr[1])));
  conflicts += ((row == curr[0]) && (col == curr[1]))*nqueens;

  if(conflicts > 0){
    atom_add(&result[col],conflicts);
  }

}
*/

__kernel void count_conflicts(const __global queen * queens,
			      __global queen * result,
			      __local queen * ws,
			      const int row, const int nqueens, 
			      const int group_size ){

  int q = get_global_id(0);
  int col = get_global_id(1);
  int block = get_group_id(0);
  int size = get_local_size(0);

  int i,j,conflicts = 0,curr[2];

  event_t e;
  e = async_work_group_copy(ws,&queens[2*size*block],2*size,0);
  wait_group_events(1,&e);

  curr[0] = ws[2*(q%size)];
  curr[1] = ws[2*(q%size)+1];

  conflicts += ((row == curr[0]) || (col == curr[1]) || 
		(abs_diff(row,curr[0]) == abs_diff(col,curr[1])));
  conflicts += ((row == curr[0]) && (col == curr[1]))*nqueens;

  if(conflicts > 0){
    atom_add(&result[col],conflicts);
  }

}


//Number of work-items should be a power of 2
//Inspired by/borrowed from http://developer.amd.com/documentation/articles/Pages/OpenCL-Optimization-Case-Study-Simple-Reductions_3.aspx
__kernel void reduce(__global queen * conflicts,
		     const int nqueens,const int which,
		     __local queen * scratch,
		     __local queen * indexes,
		     __global queen * result,
		     __global int * result_i,
		     __global int * cf_iters,
		     const int col,const int rand){

  int gid = get_global_id(0);
  if(gid == 0){
    if( (conflicts[col]) == nqueens+1){
      cf_iters[0] = (cf_iters[0]+1);
      conflicts[col] = -1;
    }
    else{
      cf_iters[0] = 0;
    }
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  int lid = get_local_id(0);
  int ind = gid < nqueens ? gid : -1;
  if(gid < nqueens){
      scratch[lid] = conflicts[gid];
  }
  else{
    scratch[lid] = nqueens*nqueens;
  }
  indexes[lid] = ind;
  barrier(CLK_LOCAL_MEM_FENCE);

  int offset, local_size = get_local_size(0)/2;
  for(offset = 1; offset > 0; offset >>=1 ){
    if(lid < offset){
      int other = lid + offset;
      int mine = lid;
      if(scratch[mine] < scratch[other] || 
	 (rand && scratch[mine] == scratch[other])){
	indexes[mine] = ind;
      }
      else{
	scratch[mine] = scratch[other];
	indexes[mine] = indexes[other];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if(lid == 0){
    int groupid = get_group_id(0);
    result[groupid] = scratch[0];
    result_i[groupid] = indexes[0];
  }
}

/*
  Only 1 of these should ever be spawned at once. Changing 1 value in global
  mem is probably faster than IO with host (read,change,write)
 
  Looks at the reduced list of conflicts, selects the lowest 
*/
__kernel void make_move(const __global queen * conflicts,
			const __global queen * indexes,
			__global queen * queens,
			const int which, const int nqueens,
			const int size, const int rand){

  int i,min_c=nqueens*nqueens,min_i;
  for(i = 0; i < size && min_c > 1; i++){
    if(conflicts[i] < min_c ||
       (conflicts[i] == min_c && rand)){
      min_c = conflicts[i];
      min_i = indexes[i];
    }
  }

  if(min_c != -1){
    queens[2*which+1] = min_i;
  }

}
