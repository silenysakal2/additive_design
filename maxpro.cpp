#include <iostream>
#include <limits>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
//#include <vector>
//#include <omp.h> // call omp_set_num_threads()

/*

Build flags

GCC/Clang: -O3 -fopenmp -march=native (optional)

add_compile_options(/O2 /openmp:llvm)  # or /openmp


MSVC: /O2 /openmp (or /openmp:llvm on newer MSVC)

*/


#if defined(_OPENMP)
	#include <omp.h>
#endif


#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif


#ifdef _WIN32
#include <cstdlib>  // For malloc/free
#endif


#ifdef _WIN32
#define EXTERN_C extern "C" __declspec(dllexport)
#else
#define EXTERN_C extern "C"
#endif

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


double per_dist(double a, double b) // (signed distance)
{
	double d = (a - b);
	if(d > 0.5)
		d = d - 1.;
	if(d < -0.5)
		d = d + 1.;
	return d;
}



struct MaxproComputeStackUnit
{
	int coord_i; // Index to candidates_coords
	long long int i_sum; // Indices to newpoint_deltas
	long long int maxpro; // A product till this point in the stack; not the inverted value (as I'm hoping for integer arithmetics to be slightly faster). Also, it's not squared yet.
};

/*
   This function generates a design of ns points. It chooses points from a meshgrid (therefore making an LHS design).
*/
EXTERN_C int *maxpro_design_meshgrid(int nv, int ns, int seed, bool periodic, bool rand_ini, bool rand_sel)
{
	int *ns_powers = (int*) malloc((nv+1) * sizeof(int)); // Bake these for later
	ns_powers[0] = 1;
	for(int v = 0; v < nv; v++)
		ns_powers[v+1] = ns_powers[v] * ns;

	double *newpoint_deltas = (double*) malloc(ns_powers[nv] * sizeof(double));
	for(long long int i = 0; i < ns_powers[nv]; i++) newpoint_deltas[i] = 0;

	int *picked_points = (int*) malloc(ns * nv * sizeof(int*));

	srand(seed);
	if(rand_ini)
		for(int v = 0; v < nv; v++)
			picked_points[0 * nv + v] = rand() % ns; // As that while loop requires a previous point being picked
	else
		for(int v = 0; v < nv; v++)
			picked_points[0 * nv + v] = 0; // As that while loop requires a previous point being picked
	
	int *candidates_coords = (int*) malloc((nv+1) * ns * sizeof(int)); // Stores available coords for new points; there's technically a bit more memory than needed (for the first point picked), and there's a whole ns-long segment at the end to not cause a segmentation fault on stop condition (a lazy solution), but I don't care.
	for(int v = 0; v < nv; v++)
		for(int s = 0; s < ns; s++)
			candidates_coords[v * ns + s] = s + (s >= picked_points[0 * nv + v] ? 1 : 0);

	MaxproComputeStackUnit *maxpro_compute_stack = (MaxproComputeStackUnit*) malloc((nv+1) * sizeof(MaxproComputeStackUnit)); // One more for easier stop condition
	for(int picked_point_count = 1; picked_point_count < ns-1; picked_point_count++) { // The last point will simply go to the remaining coords (thats why ns-1; the stack incrementation as it is -- the most optimized I could think of -- crashes otherwise)
		int *last_point = picked_points + ((picked_point_count-1) * nv);
		int *next_point = picked_points + (picked_point_count * nv); // First, it'll hold indices to candidates_coords, then, they'll be replaced with the actual coords

		maxpro_compute_stack[nv] = {0, 0, 1};
		for(int v = nv-1; v >= 0; v--) {
			long long int dx = abs(candidates_coords[v * ns + 0] - last_point[v]);
			if(periodic && dx > (ns/2)) // Periodic
				dx = ns - dx;
			maxpro_compute_stack[v] = {0, candidates_coords[v * ns + 0] * ns_powers[v] + maxpro_compute_stack[v+1].i_sum, dx * maxpro_compute_stack[v+1].maxpro};
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
			maxpro_compute_stack[v].i_sum = maxpro_compute_stack[v+1].i_sum + (candidates_coords[v * ns + maxpro_compute_stack[v].coord_i] * ns_powers[v]);
			long long int dx = abs(last_point[v] - candidates_coords[v * ns + maxpro_compute_stack[v].coord_i]);
			if(periodic && dx > (ns/2)) // Periodic
				dx = ns - dx;
			maxpro_compute_stack[v].maxpro = maxpro_compute_stack[v+1].maxpro * dx;
			v--;
			for(; v >= 0; v--) { // Increment the stack
				maxpro_compute_stack[v].coord_i = 0;
				maxpro_compute_stack[v].i_sum = maxpro_compute_stack[v+1].i_sum + (candidates_coords[v * ns + maxpro_compute_stack[v].coord_i] * ns_powers[v]);
				dx = abs(last_point[v] - candidates_coords[v * ns + maxpro_compute_stack[v].coord_i]);
				if(periodic && dx > (ns/2)) // Periodic
					dx = ns - dx;
				maxpro_compute_stack[v].maxpro = maxpro_compute_stack[v+1].maxpro * dx;
			}
		}
		
		// Adding the point:
		for(int v = 0; v < nv; v++) {
			int *cc = candidates_coords + (v * ns);
			int nextPoint_x = cc[next_point[v]];
			for(int i = next_point[v]; i < (ns - picked_point_count - 1); i++) // Shift the array rather than swapping the last element for better cache
				cc[i] = cc[i+1];
			next_point[v] = nextPoint_x;
		}
	}

	for(int v = 0; v < nv; v++)
		picked_points[(ns-1) * nv + v] = candidates_coords[v * ns + 0];

	free(ns_powers);
	free(newpoint_deltas);
	free(candidates_coords);
	free(maxpro_compute_stack);
	return picked_points;
}


/*
   This function generates a design of ns points.
   
   For every new point, it chooses the best from a list of candidates.
*/
EXTERN_C void gen_design_candidates(char crit, int nv, int ns, long long int candidate_count, double *candidates, int seed, bool periodic, bool rand_sel)
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


// This is only to be really used in the next function
static inline double maxpro_semiAnalytical_1DAreaMaxMul(int nv, int ns, int v, int s, double *points, const int *area_counter, const double *coords_sorted, bool periodic)
{
	double p = points[s * nv + v];
	double max_mul;
	if(periodic) {
		bool area_wrapped = area_counter[v] == (ns-1);
		double area_start = coords_sorted[v * ns + area_counter[v]];
		double area_end = coords_sorted[v * ns + (area_wrapped ? 0 : (area_counter[v] + 1))];
		double p_opposite = (p > 0.5) ? (p - 0.5) : (p + 0.5);
		if(area_wrapped ? ((area_start < p_opposite) || (area_end > p_opposite)) : ((area_start < p_opposite) && (area_end > p_opposite)))
			max_mul = 0.5;
		else
			max_mul = std::max(fabs(per_dist(p, area_start)), fabs(per_dist(p, area_end)));
	}
	else {
		double area_start = area_counter[v] ? coords_sorted[(v) * ns + area_counter[v]-1] : 0.;
		double area_end = (area_counter[v] == nv) ? 1. : coords_sorted[v * ns + area_counter[v]];
		max_mul = std::max(fabs(p - area_start), fabs(p - area_end));
	}
	return max_mul;
}

/*
   This function splits the unit hypercube into areas: along each axis, they are divided by the points' coordinates. Inside each of these areas, it iteratively finds the best next point, and then it adds it to the end of the `points` array. `ns` represents number of points BEFORE calling this function.

   It is already optimized: there is a way to, for any rectangular area, calculate the minimum maxpro. In this function however, that strategy is only employed to eliminate areas right away, not to make a tree. For a better function (it may not always be faster), look for the word "tree".
*/

/*
EXTERN_C long long int maxpro_addPoint_semiAnalytical_Par(
	int nv, int ns, double* points,
	double error_treshold, int min_iterations, int max_iterations, bool periodic)
{
	if (ns == 1) { // The counter breaks if set to 1
		for (int v = 0; v < nv; v++) {
			points[nv + v] = points[v] + 0.5;
			if (points[v + nv] > 1.) points[v + nv] -= 1.;
		}
		return -1;
	}

	long long int skipped_points = 0;

	// Build coords_sorted (independent per v)
	double* coords_sorted = (double*)malloc(nv * ns * sizeof(double));
#pragma omp parallel for schedule(static)
	for (int v = 0; v < nv; v++) {
		for (int s = 0; s < ns; s++)
			coords_sorted[v * ns + s] = points[s * nv + v];
		std::sort(coords_sorted + (v * ns), coords_sorted + ((v + 1) * ns));
	}

	// area_counter: one extra at the start for stop condition; use pointer offset like original
	int* area_counter = ((int*)malloc((nv + 1) * sizeof(int))) + 1;
	for (int v = -1; v < nv; v++) area_counter[v] = 0;

	// For computing maxpro minima in the areas
	double* maxpro_compute_stack = (double*)malloc(nv * ns * sizeof(double));
	for (int s = 0; s < ns; s++) maxpro_compute_stack[s] = 1.;

	// Each layer depends on the previous one (keep v sequential), parallelize over s.
	for (int v = 0; v < (nv - 1); v++) {
#pragma omp parallel for schedule(static)
		for (int s = 0; s < ns; s++) {
			maxpro_compute_stack[(v + 1) * ns + s] =
				maxpro_compute_stack[v * ns + s] *
				maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, v, s, points, area_counter, coords_sorted, periodic);
		}
	}

	double* best_cand = (double*)malloc(nv * sizeof(double));
	double best_cand_maxpro = std::numeric_limits<double>::infinity();

	double* curr_cand = (double*)malloc(nv * sizeof(double));
	double* curr_cand_maxpro_d = (double*)malloc(nv * sizeof(double)); // first derivative accumulators
	double* curr_cand_maxpro_dd = (double*)malloc(nv * sizeof(double)); // second derivative accumulators

	while (area_counter[-1] == 0) {
		// Calculate the last *layer* min maxpro (sum over s) — parallel reduction
		double min_maxpro = 0.;
#pragma omp parallel for reduction(+:min_maxpro) schedule(static)
		for (int s = 0; s < ns; s++) {
			double dx_prod =
				maxpro_compute_stack[(nv - 1) * ns + s] *
				maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, nv - 1, s, points, area_counter, coords_sorted, periodic);
			min_maxpro += 1. / (dx_prod * dx_prod);
		}

		if (min_maxpro < best_cand_maxpro) { // numerical root search in this area
			// Initialize candidate at area mid and set initial "error" scale
			double error = 0.;

			if (periodic) {
				for (int v = 0; v < nv; v++) {
					bool area_wrapped = area_counter[v] == (ns - 1);
					double area_start = coords_sorted[v * ns + area_counter[v]];
					double area_end = coords_sorted[v * ns + (area_wrapped ? 0 : (area_counter[v] + 1))];
					curr_cand[v] = area_wrapped ? ((area_start + area_end + 1.) * 0.5) : ((area_start + area_end) * 0.5);
					if (area_wrapped && (curr_cand[v] > 1.)) curr_cand[v] -= 1.;
					double area_dx = area_end - area_start + (area_wrapped ? 1. : 0.);
					error += area_dx * area_dx;
				}
			}
			else {
				for (int v = 0; v < nv; v++) {
					// (Kept verbatim with original indexing behavior)
					double area_start = area_counter[nv - 1] ? coords_sorted[(nv - 1) * ns + area_counter[nv - 1] - 1] : 0.;
					double area_end = (area_counter[nv - 1] == nv) ? 1. : coords_sorted[(nv - 1) * ns + area_counter[nv - 1]];
					curr_cand[v] = (area_start + area_end) * 0.5;
					double area_dx = area_end - area_start;
					error += area_dx * area_dx;
				}
			}

			error = std::sqrt(error) * 0.5;

			for (int iteration_i = 0;
				(iteration_i < max_iterations) && ((error > error_treshold) || (iteration_i < min_iterations));
				iteration_i++)
			{
				// zero accumulators
				for (int v = 0; v < nv; v++) {
					curr_cand_maxpro_d[v] = 0.;
					curr_cand_maxpro_dd[v] = 0.;
				}

				// Parallel accumulation over s with per-thread locals to avoid races
#pragma omp parallel
				{
					double* d_local = (double*)malloc(nv * sizeof(double));
					double* dd_local = (double*)malloc(nv * sizeof(double));
					for (int v = 0; v < nv; v++) { d_local[v] = 0.; dd_local[v] = 0.; }

#pragma omp for schedule(static)
					for (int s = 0; s < ns; s++) {
						double dx_prod = 1.;
#pragma omp simd reduction(*:dx_prod)
						for (int v = 0; v < nv; v++) {
							double dx = periodic ? per_dist(points[s * nv + v], curr_cand[v])
								: (points[s * nv + v] - curr_cand[v]);
							dx_prod *= dx;
						}
						double maxpro = 1. / (dx_prod * dx_prod);

						for (int v = 0; v < nv; v++) {
							double dx = periodic ? per_dist(points[s * nv + v], curr_cand[v])
								: (points[s * nv + v] - curr_cand[v]);
							double dx_inv = 1. / dx;
							d_local[v] += maxpro * dx_inv;            // -2 factor applied later
							dd_local[v] += maxpro * dx_inv * dx_inv;   // +6 factor applied later
						}
					}

					// Combine thread-local partials
#pragma omp critical
					{
						for (int v = 0; v < nv; v++) {
							curr_cand_maxpro_d[v] += d_local[v];
							curr_cand_maxpro_dd[v] += dd_local[v];
						}
					}

					free(d_local);
					free(dd_local);
				} // end parallel

				// Take a Newton-like step and compute new "error"
				double err_acc = 0.;
				for (int v = 0; v < nv; v++) {
					double dx = (-2.0 * curr_cand_maxpro_d[v]) / (6.0 * curr_cand_maxpro_dd[v]);
					err_acc += dx * dx;
					curr_cand[v] += dx;
					if (curr_cand[v] > 1.)      curr_cand[v] -= 1.;
					else if (curr_cand[v] < 0.) curr_cand[v] += 1.;
				}
				error = std::sqrt(err_acc);
			}

			// Evaluate candidate objective — parallel reduction over s
			double curr_cand_maxpro = 0.;
#pragma omp parallel for reduction(+:curr_cand_maxpro) schedule(static)
			for (int s = 0; s < ns; s++) {
				double dx_prod = 1.;
#pragma omp simd reduction(*:dx_prod)
				for (int v = 0; v < nv; v++) {
					double dx = periodic ? per_dist(points[s * nv + v], curr_cand[v])
						: (points[s * nv + v] - curr_cand[v]);
					dx_prod *= dx;
				}
				curr_cand_maxpro += 1. / (dx_prod * dx_prod);
			}

			if (curr_cand_maxpro < best_cand_maxpro) {
				double* tmp = best_cand;
				best_cand = curr_cand;
				curr_cand = tmp;
				best_cand_maxpro = curr_cand_maxpro;
			}
		}
		else {
			skipped_points++;
		}

		// Stack incrementing; parallelize layer recomputes over s
		int increment_v;
		for (increment_v = nv - 1; area_counter[increment_v] == (periodic ? (ns - 1) : ns); increment_v--);
		area_counter[increment_v]++;

		if (increment_v >= 0 && (increment_v < (nv - 1))) {
#pragma omp parallel for schedule(static)
			for (int s = 0; s < ns; s++)
				maxpro_compute_stack[(increment_v + 1) * ns + s] =
				maxpro_compute_stack[increment_v * ns + s] *
				maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, increment_v, s, points, area_counter, coords_sorted, periodic);
		}
		increment_v++;
		for (; increment_v < (nv - 1); increment_v++) {
			area_counter[increment_v] = 0;
#pragma omp parallel for schedule(static)
			for (int s = 0; s < ns; s++)
				maxpro_compute_stack[(increment_v + 1) * ns + s] =
				maxpro_compute_stack[increment_v * ns + s] *
				maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, increment_v, s, points, area_counter, coords_sorted, periodic);
		}
		if (increment_v < nv) area_counter[increment_v] = 0; // last one; stack layer not updated
	}

	for (int v = 0; v < nv; v++)
		points[ns * nv + v] = best_cand[v];

	free(coords_sorted);
	free(area_counter - 1);
	free(maxpro_compute_stack);
	free(best_cand);
	free(curr_cand);
	free(curr_cand_maxpro_d);
	free(curr_cand_maxpro_dd);

	return skipped_points;
}
*/

/*
   This function splits the unit hypercube into areas: along each axis, they are divided by the points' coordinates. Inside each of these areas, it iteratively finds the best next point, and then it adds it to the end of the `points` array. `ns` represents number of points BEFORE calling this function.

   It is already optimized: there is a way to, for any rectangular area, calculate the minimum maxpro. In this function however, that strategy is only employed to eliminate areas right away, not to make a tree. For a better function (it may not always be faster), look for the word "tree".
*/
EXTERN_C long long int maxpro_addPoint_semiAnalytical(int nv, int ns, double *points, double error_treshold, int min_iterations, int max_iterations, bool periodic)
{
	if(ns == 1) { // The counter breaks if set to 1
		for(int v = 0; v < nv; v++) {
			points[nv + v] = points[v] + 0.5;
			if(points[v + nv] > 1.)
				points[v + nv] -= 1.;
		}
		return -1;
	}

	long long int skipped_points = 0;


	// TODO: encapsulate stuff in functions/macros (macros might be better here) so there isn't as much repetitive code
	double *coords_sorted = (double*) malloc(nv * ns * sizeof(double));
	for(int v = 0; v < nv; v++) {
		for(int s = 0; s < ns; s++)
			coords_sorted[v * ns + s] = points[s * nv + v];
		std::sort(coords_sorted + (v * ns), coords_sorted + ((v+1) * ns));
	}

	int *area_counter = ((int*) malloc((nv+1) * sizeof(int))) + 1; // There's one more at the start for the stop condition
	for(int v = -1; v < nv; v++)
		area_counter[v] = 0;
	// When periodic, the area 0 is in beetween the 0th and 1st tick; the last area loops over. When not periodic, the area 0 is an edge area, with there being and if statement to check for zero-width areas

	double *maxpro_compute_stack = (double*) malloc(nv * ns * sizeof(double)); // For computing maxpro minima in the areas
	for(int s = 0; s < ns; s++)
		maxpro_compute_stack[s] = 1.;
	for(int v = 0; v < (nv-1); v++)
		for(int s = 0; s < ns; s++)
			maxpro_compute_stack[(v+1) * ns + s] = maxpro_compute_stack[v * ns + s] * maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, v, s, points, area_counter, coords_sorted, periodic);

	double *best_cand = (double*) malloc(nv * sizeof(double));
	double best_cand_maxpro = std::numeric_limits<double>::infinity();

	double *curr_cand = (double*) malloc(nv * sizeof(double));
	double *curr_cand_maxpro_d = (double*) malloc(nv * sizeof(double)); // Maxpro first derivative (the delta maxpro when adding the candidate)
	double *curr_cand_maxpro_dd = (double*) malloc(nv * sizeof(double)); // Maxpro second derivative

	while(area_counter[-1] == 0) {
		// TODO beware zero-width areas
		// Calculate that last *layer* of maxpro_compute_stack
		double min_maxpro = 0.;
		for(int s = 0; s < ns; s++) {
			double dx_prod = maxpro_compute_stack[(nv-1) * ns + s] * maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, nv-1, s, points, area_counter, coords_sorted, periodic);
			min_maxpro += 1. / (dx_prod * dx_prod);
		}
		// Thus we've now obtained the min maxpro of our current area

		if(min_maxpro < best_cand_maxpro) { // It's worth doing a numerical root search
			// TODO add the opposite-point trial

			double error = 0.; // Initialize the error at half-diagonal distance
			// Also, a bit of a disclaimer: it isn't really an estimated error, but rather the amount 

			if(periodic)
				for(int v = 0; v < nv; v++) {
					bool area_wrapped = area_counter[v] == (ns-1);
					double area_start = coords_sorted[v * ns + area_counter[v]];
					double area_end = coords_sorted[v * ns + (area_wrapped ? 0 : (area_counter[v] + 1))];
					curr_cand[v] = area_wrapped ? ((area_start + area_end + 1.) * 0.5) : ((area_start + area_end) * 0.5);
					if(area_wrapped && (curr_cand[v] > 1.)) // Adding area_wrapped to the condition speeds thing up by like one instruction (unless it's wrapped) :D
						curr_cand[v] -= 1.;

					double area_dx = area_end - area_start + (area_wrapped ? 1. : 0.);
					error += area_dx * area_dx;
				}
			else
				for(int v = 0; v < nv; v++) {
					double area_start = area_counter[nv-1] ? coords_sorted[(nv-1) * ns + area_counter[nv-1]-1] : 0.;
					double area_end = (area_counter[nv-1] == nv) ? 1. : coords_sorted[(nv-1) * ns + area_counter[nv-1]];
					curr_cand[v] = (area_start + area_end) * 0.5;

					double area_dx = area_end - area_start;
					error += area_dx * area_dx;
				}

			error = sqrt(error) * 0.5;
			for(int iteration_i = 0; (iteration_i < max_iterations) && ((error > error_treshold) || (iteration_i < min_iterations)); iteration_i++) {
				// Calculate the delta-maxpro of this candidate's first and second derivative
				for(int v = 0; v < nv; v++) {
					curr_cand_maxpro_d[v] = 0.;
					curr_cand_maxpro_dd[v] = 0.;
				}
				for(int s = 0; s < ns; s++) {
					double dx_prod = 1.;
					for(int v = 0; v < nv; v++)
						dx_prod *= periodic ? per_dist(points[s * nv + v], curr_cand[v]) : (points[s * nv + v] - curr_cand[v]);
					double maxpro = 1. / (dx_prod * dx_prod);
					for(int v = 0; v < nv; v++) {
						double dx_inv = 1. / (periodic ? per_dist(points[s * nv + v], curr_cand[v]) : (points[s * nv + v] - curr_cand[v]));
						curr_cand_maxpro_d[v] += maxpro * dx_inv; // We'll be multiplying by the -2 later
						curr_cand_maxpro_dd[v] += maxpro * dx_inv * dx_inv; // We'll be multiplying by the +6 later
					}
				}

				// Fuzz the candidate based on the derivatives (while calculating the error)
				error = 0.;
				for(int v = 0; v < nv; v++) {
					double dx = (-2 * curr_cand_maxpro_d[v]) / (6 * curr_cand_maxpro_dd[v]);
					error += dx*dx;
					curr_cand[v] += dx;
					if(curr_cand[v] > 1.)
						curr_cand[v] -= 1.;
					else if(curr_cand[v] < 0.)
						curr_cand[v] += 1.;
				}
				error = sqrt(error);
			}

			// Calculate the optimized candidate's maxpro delta and accept/reject it
			double curr_cand_maxpro = 0.;
			for(int s = 0; s < ns; s++) {
				double dx_prod = 1.;
				for(int v = 0; v < nv; v++)
					dx_prod *= periodic ? per_dist(points[s * nv + v], curr_cand[v]) : (points[s * nv + v] - curr_cand[v]);
				curr_cand_maxpro += 1. / (dx_prod*dx_prod);
			}

			if(curr_cand_maxpro < best_cand_maxpro) {
				double *tmp = best_cand;
				best_cand = curr_cand;
				curr_cand = tmp;
				best_cand_maxpro = curr_cand_maxpro;
			}
		}
		else
			skipped_points++;

		// Stack incrementing
		int increment_v;
		for(increment_v = nv-1; area_counter[increment_v] == (periodic ? (ns-1) : ns); increment_v--);
		area_counter[increment_v]++;
		if(increment_v >= 0 && (increment_v < (nv-1)))
			for(int s = 0; s < ns; s++)
				maxpro_compute_stack[(increment_v+1) * ns + s] = maxpro_compute_stack[increment_v * ns + s] * maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, increment_v, s, points, area_counter, coords_sorted, periodic);
		increment_v++;
		for(; increment_v < (nv-1); increment_v++) { // Why not do the last? I have another optimization: for the last layer of the maxpro_compute_stack, numbers only have to be added.
			area_counter[increment_v] = 0;

			for(int s = 0; s < ns; s++)
				maxpro_compute_stack[(increment_v+1) * ns + s] = maxpro_compute_stack[increment_v * ns + s] * maxpro_semiAnalytical_1DAreaMaxMul(nv, ns, increment_v, s, points, area_counter, coords_sorted, periodic);
		}
		if(increment_v < nv)
			area_counter[increment_v] = 0; // Here goes the last one, but the maxpro_compute_stack will not be updated.
	}

	for(int v = 0; v < nv; v++)
		points[ns * nv + v] = best_cand[v];

	free(coords_sorted);
	free(area_counter - 1);
	free(maxpro_compute_stack);
	free(best_cand);
	free(curr_cand);
	free(curr_cand_maxpro_d);
	free(curr_cand_maxpro_dd);

	return skipped_points;
}




int main()
{
	return 0;
}
