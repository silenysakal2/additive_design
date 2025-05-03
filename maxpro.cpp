#include <iostream>
#include <limits>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <unistd.h>



double pow(double base, int exponent)
{
	double res = 1.;
	for(int bin_pos = 1; bin_pos <= exponent; bin_pos <<= 1) {
		if(exponent & bin_pos)
			res *= base;
		base *= base;
	}
	return res;
}



struct MaxproComputeStackUnit
{
	int coord_i; // Index to candidates_coords
	long long int i_sum; // Indices to newpoint_deltas
	long long int maxpro; // A product till this point in the stack; not the inverted value (as I'm hoping for integer arithmetics to be slightly faster). Also, it's not squared yet.
};

extern "C" int *maxpro_design_meshgrid(int nv, int ns, int seed, bool periodic, bool rand_ini, bool rand_sel)
{
	int *ns_powers = (int*) malloc((nv+1) * sizeof(int)); // Bake these for later
	ns_powers[0] = 1;
	for(int v = 0; v < nv; v++)
		ns_powers[v+1] = ns_powers[v] * ns;

	double *newpoint_deltas = (double*) malloc(ns_powers[nv] * sizeof(double));
	for(long long int i = 0; i < ns_powers[nv]; i++) newpoint_deltas[i] = 0;

	int *picked_points_org = (int*) malloc(ns * nv * sizeof(int)); // An array of picked points; in order they were picked. This array is only ever written to, except for the final print.
	int **picked_points = (int**) malloc(ns * sizeof(int*));
	for(int s = 0; s < ns; s++)
		picked_points[s] = picked_points_org + (s*nv);

	srand(seed);
	if(rand_ini)
		for(int v = 0; v < nv; v++)
			picked_points[0][v] = rand() % ns; // As that while loop requires a previous point being picked
	else
		for(int v = 0; v < nv; v++)
			picked_points[0][v] = 0; // As that while loop requires a previous point being picked
	
	int *candidates_coords_org = (int*) malloc(nv * ns * sizeof(int)); // Stores available coords for new points; there's technically a bit more memory than needed, but I don't care
	int **candidates_coords = (int**) malloc((nv+1) * sizeof(int*));
	for(int v = 0; v < nv; v++) {
		candidates_coords[v] = candidates_coords_org + (v*ns);
		for(int s = 0; s < ns; s++)
			candidates_coords[v][s] = s + (s >= picked_points[0][v] ? 1 : 0);
	}
	candidates_coords[nv] = candidates_coords_org; // The last one just has to point to valid memory as it will be used for the additional element of the stack (for stop condition)

	MaxproComputeStackUnit *maxpro_compute_stack = (MaxproComputeStackUnit*) malloc((nv+1) * sizeof(MaxproComputeStackUnit)); // One more for easier stop condition
	for(int picked_point_count = 1; picked_point_count < ns-1; picked_point_count++) { // The last point will simply go to the remaining coords (thats why ns-1; the stack incrementation as it is -- the most optimized I could think of -- crashes otherwise)
		//std::cout << "\n\n--New point--\n";
		int *last_point = picked_points[picked_point_count-1];
		int *next_point = picked_points[picked_point_count]; // First, it'll hold indices to candidates_coords, then, they'll be replaced with the actual coords

		maxpro_compute_stack[nv] = {0, 0, 1};
		for(int v = nv-1; v >= 0; v--) {
			long long int dx = abs(candidates_coords[v][0] - last_point[v]);
			if(periodic && dx > (ns/2)) // Periodic
				dx = ns - dx;
			maxpro_compute_stack[v] = {0, candidates_coords[v][0] * ns_powers[v] + maxpro_compute_stack[v+1].i_sum, dx * maxpro_compute_stack[v+1].maxpro};
			next_point[v] = 0;
		}
		double best_newpoint_delta = std::numeric_limits<double>::infinity();
		int best_newpoint_count; // So far
		while(maxpro_compute_stack[nv].coord_i == 0) { // Loop through all the points, increment their deltas, and choose the best
			// Adding to the newpoint_deltas:
			int i = maxpro_compute_stack[0].i_sum;
			newpoint_deltas[i] += 1. / (((double) maxpro_compute_stack[0].maxpro) * maxpro_compute_stack[0].maxpro);
			if(newpoint_deltas[i] < best_newpoint_delta) {
				best_newpoint_delta = newpoint_deltas[i];
				for(int v = 0; v < nv; v++)
					next_point[v] = maxpro_compute_stack[v].coord_i;
				best_newpoint_count = 1;
			}
			else if(rand_sel && newpoint_deltas[i] == best_newpoint_delta) {
				best_newpoint_count++;
				if((rand() % best_newpoint_count) == 0)
					for(int v = 0; v < nv; v++)
						next_point[v] = maxpro_compute_stack[v].coord_i;
			}

			// Stack incrementing:
			int v;
			for(v = 0; maxpro_compute_stack[v].coord_i == (ns - picked_point_count - 1); v++); // Find stack increment depth
			maxpro_compute_stack[v].coord_i++;
			maxpro_compute_stack[v].i_sum = maxpro_compute_stack[v+1].i_sum + (candidates_coords[v][maxpro_compute_stack[v].coord_i] * ns_powers[v]);
			long long int dx = abs(last_point[v] - candidates_coords[v][maxpro_compute_stack[v].coord_i]);
			if(periodic && dx > (ns/2)) // Periodic
				dx = ns - dx;
			maxpro_compute_stack[v].maxpro = maxpro_compute_stack[v+1].maxpro * dx;
			v--;
			for(; v >= 0; v--) { // Increment the stack
				maxpro_compute_stack[v].coord_i = 0;
				maxpro_compute_stack[v].i_sum = maxpro_compute_stack[v+1].i_sum + (candidates_coords[v][maxpro_compute_stack[v].coord_i] * ns_powers[v]);
				dx = abs(last_point[v] - candidates_coords[v][maxpro_compute_stack[v].coord_i]);
				if(periodic && dx > (ns/2)) // Periodic
					dx = ns - dx;
				maxpro_compute_stack[v].maxpro = maxpro_compute_stack[v+1].maxpro * dx;
			}
		}
		
		// Adding the point:
		for(int v = 0; v < nv; v++) {
			int *cc = candidates_coords[v];
			int nextPoint_x = cc[next_point[v]];
			for(int i = next_point[v]; i < (ns - picked_point_count - 1); i++) // Shift the array rather than swapping the last element for better cache
				cc[i] = cc[i+1];
			next_point[v] = nextPoint_x;
		}
	}
	std::cout << std::flush;

	for(int v = 0; v < nv; v++)
		picked_points[ns-1][v] = candidates_coords[v][0];

	free(ns_powers);
	free(newpoint_deltas);
	free(picked_points);
	free(candidates_coords_org);
	free(candidates_coords);
	free(maxpro_compute_stack);
	return picked_points_org;
}


extern "C" void gen_design_candidates(char crit, int nv, int ns, long long int candidate_count, double *candidates, int seed, bool periodic, bool rand_sel)
{
	srand(seed);

	const double MAX_DIST_SQ = sqrt(nv);

	double *cand_deltas = (double*) malloc(candidate_count * sizeof(double));
	for(long long int i = 0; i < candidate_count; i++)
		cand_deltas[i] = 0.;

	/*const size_t vec_size = nv * sizeof(double);
	double *swap = (double*) malloc(vec_size);*/

	for(int picked_point_count = 0; picked_point_count < ns; picked_point_count++) {
		if(ns > 128) // Log progress
			std::cout << (picked_point_count * 100 / ns) << "%\r" << std::flush;
		// Determine the best candidate
		double best_cand_delta = std::numeric_limits<double>::infinity();
		long long int best_cand;
		long long int best_cand_count;
		for(long long int cand_i = picked_point_count; cand_i < candidate_count; cand_i++) {
			if(cand_deltas[cand_i] < best_cand_delta) {
				best_cand_delta = cand_deltas[cand_i];
				best_cand = cand_i;
				best_cand_count = 1;
			}
			else if(rand_sel && cand_deltas[cand_i] == best_cand_delta) {
				best_cand_count++;
				if(rand() % best_cand_count == 0)
					best_cand = cand_i;
			}
		}

		// Swap to commit to this point selected
		for(int v = 0; v < nv; v++) {
			double tmp = candidates[best_cand * nv + v];
			candidates[best_cand * nv + v] = candidates[picked_point_count * nv + v];
			candidates[picked_point_count * nv + v] = tmp;
		}
		/*double *chosen = candidates + (best_cand * nv);
		double *position = candidates + (picked_point_count * nv);
		memcpy(swap, chosen, vec_size);
		memcpy(chosen, position, vec_size);
		memcpy(position, swap, vec_size);*/
		cand_deltas[best_cand] = cand_deltas[picked_point_count]; // Discard the delta of the chosen one


		// Update the deltas accordingly
		switch(crit) {
			case 'm': // (u)Maxpro
				for(int cand_i = picked_point_count+1; cand_i < candidate_count; cand_i++) {
					double maxpro = 1.;
					for(int v = 0; v < nv; v++) {
						double dx = fabs(candidates[picked_point_count * nv + v] - candidates[cand_i * nv + v]);
						if(periodic && (dx > 0.5))
							dx = 1. - dx;
						maxpro *= dx;
					}
					maxpro = 1. / (maxpro*maxpro);
					cand_deltas[cand_i] += maxpro;
				}
				break;
			case 'M': // Maximin
				for(int cand_i = picked_point_count+1; cand_i < candidate_count; cand_i++) {
					double dist_sq = 0.;
					for(int v = 0; v < nv; v++) {
						double dx = fabs(candidates[picked_point_count * nv + v] - candidates[cand_i * nv + v]);
						if(periodic && (dx > 0.5))
							dx = 1. - dx;
						dist_sq += dx*dx;
					}
					cand_deltas[cand_i] = std::max(MAX_DIST_SQ - dist_sq, cand_deltas[cand_i]);
				}
				break;
			case 'p': // Phi_m
				for(int cand_i = picked_point_count+1; cand_i < candidate_count; cand_i++) {
					double dist_sq = 0.;
					for(int v = 0; v < nv; v++) {
						double dx = fabs(candidates[picked_point_count * nv + v] - candidates[cand_i * nv + v]);
						if(periodic && (dx > 0.5))
							dx = 1. - dx;
						dist_sq += dx*dx;
					}
					double phi = 1. / pow(dist_sq, (nv + 2) / 2); // It's +2 instead of +1, so it rounds up
					cand_deltas[cand_i] += phi;
				}
				break;
			default:
				throw std::runtime_error("Unknown criterion");
				break;
		}
	}

	free(cand_deltas);

	return;
}




int main()
{
	return 0;
}
