#include "kmeans.h"
#include <assert.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <stdio.h>

/** Retourne la différence (en secondes) entre deux timespec */
double get_delta(struct timespec begin, struct timespec end) {
	return end.tv_sec - begin.tv_sec + (end.tv_nsec - begin.tv_nsec) * 1e-9;
}

double kmeans_dist(int d, double p1[d], double p2[d]) {
	double dist = 0;
	
	for (int i = 0; i < d; i++) {
		double delta = p1[i] - p2[i];
		dist += delta * delta;
	}
	return dist;
}

int kmeans(int d, int n, int k, double points[n][d], double means[k][d], int clusters[n], int max_iter, unsigned int * rand_state) {
	struct timespec begin_init, end_init;
	struct timespec begin_assign, end_assign;
	struct timespec begin_compute, end_compute;
	
	double delta_assign = 0.0;
	double delta_compute = 0.0;
	
	assert(k <= n);
	
	clock_gettime(CLOCK_REALTIME, &begin_init);
	kmeans_init(d, n, k, points, means, rand_state);
	clock_gettime(CLOCK_REALTIME, &end_init);
	printf("Temps d'initialisation: %lf s.\n", get_delta(begin_init, end_init));
	
	for (int i=0; i < max_iter; i++) {
		clock_gettime(CLOCK_REALTIME, &begin_assign);
		bool changed = kmeans_assign_clusters(d, n, k, points, means, clusters);
		clock_gettime(CLOCK_REALTIME, &end_assign);
		delta_assign += get_delta(begin_assign, end_assign);
		
		if (!changed) {
			printf("Temps de assignment: %lf s.\n", delta_assign);
			printf("Temps de compute: %lf s.\n", delta_compute);
			return i;
		}
		clock_gettime(CLOCK_REALTIME, &begin_compute);
		kmeans_compute_means(d, n, k, points, means, clusters);
		clock_gettime(CLOCK_REALTIME, &end_compute);
		delta_compute += get_delta(begin_compute, end_compute);
	}
	printf("Temps d'assignment: %lf s.\n", delta_assign);
	printf("Temps de calcul: %lf s.\n", delta_compute);
	return max_iter;
}

void kmeans_init(int d, int n, int k, double points[n][d], double means[k][d], unsigned int * rand_state) {
	int * indices = (int *) malloc(n*sizeof(int));
	
	for (int i = 0; i < n; i++) {
		indices[i] = i;
	}
	for (int i = n-1; i > 0; i--) {
		int j = rand_r(rand_state) % (i+1);
		int temp = indices[i];
		indices[i] = indices[j];
		indices[j] = temp;
	}
	for (int i = 0; i < k; i++) {
		for (int x = 0; x < d; x++) {
			means[i][x] = points[indices[i]][x]; 
		}
	}
	free(indices);
}

bool kmeans_assign_clusters(int d, int n, int k, double points[n][d], double means[k][d], int clusters[n]) {
	bool changed = false;
	#pragma omp parallel
	{
		#pragma omp for 
		for (int i = 0; i < n; i++) {
			int min_index = 0;
			double min_dist = kmeans_dist(d, points[i], means[0]);
			for (int j = 1; j < k; j++) {
				double dist = kmeans_dist(d, points[i], means[j]);
				if (dist < min_dist) {
					min_dist = dist;
					min_index = j;
				}
			}
			if (clusters[i] != min_index) {
				clusters[i] = min_index;
				#pragma omp atomic write
				changed = true;
			}
		}
	}
	return changed;
}

void kmeans_compute_means(int d, int n, int k, double points[n][d], double means[k][d], int clusters[n]) {
	int nb_points[k];
	for (int j = 0; j < k; j++) {
		for (int x = 0; x < d; x++) {
			means[j][x] = 0;
		}
		nb_points[j] = 0;
	}

	#pragma omp parallel for default(none) shared(n,d,k,points,clusters) schedule(static) reduction(+:nb_points) reduction(+:means[:k])
	for (int i = 0; i < n; i++) {
	    int j = clusters[i];
	    for (int x = 0; x < d; x++) {
		means[j][x] += points[i][x];
	    }
	    nb_points[j]++;
	}



	for (int j = 0; j < k; j++) {
		for (int x = 0; x < d; x++) {
			means[j][x] /= nb_points[j];
		}
	}
}

