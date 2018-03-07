
__kernel void func(__global float* grid, 
				   __global float* res, 
				   __local float* local_grid)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t size   = get_global_size(0);
	size_t block_x_with_halo = get_local_size(0) + 8;
	size_t block_x = get_local_size(0);
	size_t local_x = get_local_id(0);
	size_t local_y = get_local_id(1);
	size_t global_id = global_y * size + global_x;
	size_t local_id  =  local_y * (block_x + 8) + local_x;

	local_grid[local_id + 4] = grid[global_id];
	if ((local_x - block_x) >= -4 && global_x > 3)
	{
		local_grid[local_id - block_x + 4] = grid[global_id - block_x];
	}
	if ((block_x + local_x) < size + 4 && global_x < size - 4)
	{
		local_grid[local_id + block_x + 4] = grid[global_id + block_x];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if (global_x > 3 && global_x < size - 4)
	{	
		float ax = 4.f / 5.f * size;
		float bx = - 1.f / 5.f * size;
		float cx = 4.f / 105.f * size;
		float dx = -1.f / 280.f * size;
		res[global_id] = (
			ax * (local_grid[local_id + 1 + 4] - local_grid[local_id-1+ 4]) +
			bx * (local_grid[local_id + 2 + 4] - local_grid[local_id-2+ 4]) +
			cx * (local_grid[local_id + 3 + 4] - local_grid[local_id-3+ 4]) +
			dx * (local_grid[local_id + 4 + 4] - local_grid[local_id-4+ 4])  
				);
	}
	else if (global_x <= 3)
	{
		float ax = -25.f / 12.f * size;
		float bx = 4.f  * size;
		float cx = -3.f * size;
		float dx = 4.f / 3.f * size;
		float ex = -1.f / 4.f * size;
		res[global_id] = ax * local_grid[local_id + 4] + 
							  bx * local_grid[local_id+1 + 4] + 
							  cx * local_grid[local_id+2 + 4] + 
							  dx * local_grid[local_id+3 + 4] + 
							  ex * local_grid[local_id+4 + 4];
	} else if (global_x >= size - 4)
	{
		float ax = -25.f / 12.f * size;
		float bx = 4.f  * size;
		float cx = -3.f * size;
		float dx = 4.f / 3.f * size;
		float ex = -1.f / 4.f * size;
		res[global_id] = -ax * local_grid[local_id + 4] - 
							  bx * local_grid[local_id-1 + 4] - 
							  cx * local_grid[local_id-2 + 4] - 
							  dx * local_grid[local_id-3 + 4] - 
							  ex * local_grid[local_id-4 + 4];
	}
	barrier(CLK_LOCAL_MEM_FENCE);


}