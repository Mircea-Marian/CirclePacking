#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <utility>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#include <cstdlib>
#include <cstring>

using namespace std;

#define DOUBLE_PI (2 * (double)3.14159265358979323846)
#define OMP_NUM_THREADS 4

class Circle {
	public:
		double x, y, r;
		Circle() {}
		void initialize(double aa, double bb, double cc) {
			x = aa ; y = bb; r = cc;
		}
		bool isOverlappingWith(Circle c) {
			return pow(x - c.x, 2) + pow(y - c.y, 2) < pow(r + c.r, 2);
		}
};

class Individual {
	private:
		double delta_angle;
		unsigned short ind_s, gran;
	public:
		Circle *circle_vec;
		double x_c, y_c, r_c, mut_s;

	   	/*
		*	Allocs memory and randomly initializes individual data.
		*/
		void initialize(
			pair<double, unsigned short> *input_vec,
			unsigned short N,
			unsigned short individual_size,
			double canvas_size,
			unsigned short granulation
			) {
			gran = granulation;
			ind_s = individual_size;

			delta_angle = DOUBLE_PI / granulation;

			circle_vec = new Circle [individual_size];

			Circle *circle_it = circle_vec, *a_circle_it;

			unsigned short a, circle_i = 0, b;
			bool isValidCircle;
			Circle c;

			while(N--) {
				a = input_vec->second;
				while(a--) {

					isValidCircle = false;
					while(!isValidCircle) {

						c.initialize(
							static_cast <double> (rand()) /
								(static_cast <double> (RAND_MAX/canvas_size)),
							static_cast <double> (rand()) /
								(static_cast <double> (RAND_MAX/canvas_size)),
							input_vec->first
						);

						b = circle_i;
						a_circle_it = circle_vec;
						isValidCircle = true;
						while(b--) {
							if(a_circle_it->isOverlappingWith(c)) {
								isValidCircle = false;
								break;
							}
							a_circle_it++;
						}

					}

					*circle_it++ = c;
					circle_i++;
				}

				input_vec++;
			}

			assert_fitness();
		}

		void partial_initialize(
			unsigned short individual_size,
			unsigned short granulation
		) {
			circle_vec = new Circle [individual_size];
			delta_angle = DOUBLE_PI / granulation;
			gran = granulation;
			ind_s = individual_size;
		}

		void dump_to_file_point_representation(
			char *file_name,
			ios_base::openmode mode,
			unsigned short dump_gran
		) {
			Circle *circle_it = circle_vec;
			unsigned short a = ind_s, b;
			double alpha;
			const double dump_delta_angle = DOUBLE_PI / dump_gran;
			pair<double,double> points[dump_gran * ind_s], *p_it = points;
			while(a--) {

				alpha = 0;
				b = dump_gran;
				while(b--) {

					*p_it = pair<double,double> (
						circle_it->x + circle_it->r * cos(alpha),
						circle_it->y + circle_it->r * sin(alpha)
					);

					alpha += dump_delta_angle;
					p_it++;
				}

				circle_it++;
			}

			ofstream f_out(file_name, mode);

			p_it = points;
			a = dump_gran * ind_s;
			while(a--)
				f_out << (p_it++)->first << ' ';
			f_out << endl;

			p_it = points;
			a = dump_gran * ind_s;
			while(a--)
				f_out << (p_it++)->second << ' ';
			f_out << endl;

			f_out.close();
		}

		void assert_fitness() {
			Circle *circle_it = circle_vec;
			unsigned short a = ind_s, b;
			double x_sum = 0.0, y_sum = 0.0, alpha;
			pair<double,double> points[gran * ind_s], *p_it = points;
			while(a--) {

				alpha = 0;
				b = gran;
				while(b--) {

					*p_it = pair<double,double> (
						circle_it->x + circle_it->r * cos(alpha),
						circle_it->y + circle_it->r * sin(alpha)
					);

					x_sum += p_it->first;
					y_sum += p_it->second;

					alpha += delta_angle;
					p_it++;
				}

				circle_it++;
			}

			x_c = x_sum / (gran * ind_s);
			y_c = y_sum / (gran * ind_s);

			r_c = -1;
			p_it = points;
			a = gran * ind_s;
			while(a--) {
				x_sum = pow(x_c - p_it->first, 2) + pow(y_c - p_it->second, 2);

				if(x_sum > r_c) r_c = x_sum;

				p_it++;
			}
			r_c = sqrt(r_c);
		}

		void mutate(double mutation_step) {
			Circle *a_circle_it = circle_vec, *b_circle_it;
			unsigned short a = ind_s;
			double d_AB;
			Circle c;
			bool isValidCircle;
			while(a--) {

				d_AB = sqrt(
					pow(a_circle_it->x - x_c, 2)
					+ pow(a_circle_it->y - y_c, 2)
				);

				c.x = x_c + (d_AB - mutation_step) * (a_circle_it->x - x_c) / d_AB;
				c.y = y_c + (d_AB - mutation_step) * (a_circle_it->y - y_c) / d_AB;
				c.r = a_circle_it->r;

				isValidCircle = true;
				b_circle_it = circle_vec;
				while(b_circle_it < a_circle_it) {

					if(a_circle_it->isOverlappingWith(*b_circle_it)) {
						isValidCircle = false;
						break;
					}

					b_circle_it++;
				}

				if(isValidCircle) {
					b_circle_it = a_circle_it + 1;
					while(b_circle_it < circle_vec + ind_s) {

						if(a_circle_it->isOverlappingWith(*b_circle_it)) {
							isValidCircle = false;
							break;
						}

						b_circle_it++;
					}

					if(isValidCircle) {
						a_circle_it->x = c.x;
						a_circle_it->y = c.y;
					}
				}

				a_circle_it++;
			}

			assert_fitness();
		}

		void single_point_cross_over(Individual *other_indiv, Individual *child,
			unsigned short point_index) {
			Circle *self_it = circle_vec, *other_it,
				*child_it, *a_circle_it;

			bool successfull_co = true;
			unsigned short a = point_index, b;
			while(a--) {

				b = ind_s - point_index;
				other_it = other_indiv->circle_vec + point_index;
				while(b--) {
					if(self_it->isOverlappingWith(*other_it)) {
						successfull_co = false;
						break;
					}
					other_it++;
				}

				if(!successfull_co)
					break;

				self_it++;
			}

			if(!successfull_co) {
				successfull_co = true;
				a = point_index, b;
				self_it = other_indiv->circle_vec;
				while(a--) {

					b = ind_s - point_index;
					other_it = circle_vec + point_index;
					while(b--) {
						if(self_it->isOverlappingWith(*other_it)) {
							successfull_co = false;
							break;
						}
						other_it++;
					}

					if(!successfull_co)
						break;

					self_it++;
				}
				if(successfull_co) {
					memcpy(
						child->circle_vec,
						other_indiv->circle_vec,
						sizeof(Circle) * point_index
					);
					memcpy(
						child->circle_vec + point_index,
						circle_vec + point_index,
						sizeof(Circle) * (ind_s - point_index)
					);
					child->assert_fitness();
				} else {
					if(r_c > other_indiv->r_c) {
						memcpy(
							child->circle_vec,
							circle_vec,
							sizeof(Circle) * ind_s
						);
						child->x_c = x_c;
						child->y_c = y_c;
						child->r_c = r_c;
					} else {
						memcpy(
							child->circle_vec,
							other_indiv->circle_vec,
							sizeof(Circle) * ind_s
						);
						child->x_c = other_indiv->x_c;
						child->y_c = other_indiv->y_c;
						child->r_c = other_indiv->r_c;
					}
				}
			} else {
				memcpy(
					child->circle_vec,
					circle_vec,
					sizeof(Circle) * point_index
				);
				memcpy(
					child->circle_vec + point_index,
					other_indiv->circle_vec + point_index,
					sizeof(Circle) * (ind_s - point_index)
				);
				child->assert_fitness();
			}
		}
};

int fitness_compare(const void * a, const void * b) {
	return ( ((Individual*)a)->r_c - ((Individual*)b)->r_c );
}

void fittest_individuals_selection(
	Individual *population,
	unsigned short population_size,
	Individual *breeding_population,
	unsigned short breeding_pop_size,
	unsigned short individual_size) {

	qsort(population, population_size, sizeof(Individual), fitness_compare);

	unsigned short i;
	#pragma parallel for private(i)
	for(i = 0 ; i < breeding_pop_size ; i++) {

		memcpy(breeding_population[i].circle_vec,
			population[i].circle_vec, individual_size * sizeof(Circle));
		breeding_population[i].x_c = population[i].x_c;
		breeding_population[i].y_c = population[i].y_c;
		breeding_population[i].r_c = population[i].r_c;
	}
}

void copy_from_breeding_population(
	Individual *population,
	Individual *breeding_population,
	unsigned short breeding_pop_size,
	unsigned short individual_size) {

	unsigned short i;
	#pragma parallel for private(i)
	for(i = 0 ; i < breeding_pop_size ; i++) {
		memcpy(population[i].circle_vec,
			breeding_population[i].circle_vec, individual_size * sizeof(Circle));
		breeding_population[i].x_c = population[i].x_c;
		breeding_population[i].y_c = population[i].y_c;
		breeding_population[i].r_c = population[i].r_c;
	}
}

int main(int argc, char* argv[]) {

	omp_set_num_threads(OMP_NUM_THREADS);

	ifstream f_in (argv[1]);

	unsigned short N, i, individual_size = 0;
	f_in >> N;

	pair<double, unsigned short> input_vec[N], *inp_it = input_vec;

	i = N;
	while(i--) {
		f_in >> inp_it->first >> inp_it->second;
		individual_size += inp_it->second;
		inp_it++;
	}
	f_in.close();

	unsigned short population_size = 10000;
	unsigned short epochs = 800;
	double canvas_size = 2000;
	unsigned short granulation = 10;
	double mutation_step = 1.0;
	double percentage = 0.05;
	double breeding_percentage = 0.2;
	double mutation_prob = 1;

	double conv_threshold = 0.0001;
	struct timeval start, end;
	unsigned long time_sum;
	vector<double> time_avgs;
	double avg_fitness, prev_avg_fitness;

	double a, avg_time;

	double rand_vec[population_size], *rand_it, rand_vec1[population_size], *rand_it1;

	Individual population[population_size], *pop_it = population;

	unsigned short breeding_pop_size = (unsigned short)(breeding_percentage
		* population_size);
	Individual breeding_population[breeding_pop_size], *breeding_it;
	struct drand48_data state_buffers[OMP_NUM_THREADS];
	long int rand_number[OMP_NUM_THREADS];
	int tid;

	for (i = 0; i < OMP_NUM_THREADS; ++i) {
		srand48_r(time(NULL), &state_buffers[i]);
	}

	a = 2 * canvas_size;
	i = population_size;
	while(i--) {
		pop_it->initialize(input_vec, N, individual_size, canvas_size,
			granulation);
		if(a > pop_it->r_c)
			a = pop_it->r_c;
		pop_it++;
	}

	#pragma omp parallel for private(i)
	for(i = 0 ; i < breeding_pop_size ; i++)
		breeding_population[i].partial_initialize(individual_size, granulation);

	cout << "before genetic algorithm: " << a << endl;

	unsigned short slice_dim = (unsigned short)(percentage * epochs), b = 0;
	double fitness_sum = 0;
	avg_fitness = 0;
	time_sum = 0;

	unsigned short iii = 0;

	while(epochs--) {

		if(!(iii % slice_dim) && iii != 0) {
			avg_time = 1.0 * time_sum / slice_dim;
			prev_avg_fitness = avg_fitness;
			avg_fitness = fitness_sum / slice_dim;
			cout << "average iteration time for " << b * slice_dim << " to " << (b+1) * slice_dim
				<< ": " << avg_time << " microseconds" << endl;
			cout << "average fitness: " << avg_fitness << endl;
			time_avgs.push_back(avg_time);
			time_sum = 0;
			fitness_sum = 0;
			b++;

			//algorithm converged
			if ((prev_avg_fitness > 0) && (abs(prev_avg_fitness - avg_fitness) < conv_threshold)) {
				cout << "Algorithm converged because the average fitness did not change by more than " << conv_threshold <<
						" across " << slice_dim << " iterations " << endl;
				break;
			}
		}

		iii++;

		gettimeofday(&start, NULL);

		fittest_individuals_selection(
			population,
			population_size,
			breeding_population,
			breeding_pop_size,
			individual_size
		);

		copy_from_breeding_population(
			population,
			breeding_population,
			breeding_pop_size,
			individual_size
		);

		rand_it = rand_vec + breeding_pop_size;
		rand_it1 = rand_vec1 + breeding_pop_size;
		a = breeding_pop_size - 1;
		#pragma omp parallel
		{
		#pragma omp for private(i, tid)
		for (i = 0; i < population_size - breeding_pop_size; ++i) {
			tid = omp_get_thread_num();
			lrand48_r(&state_buffers[tid], &rand_number[tid]);
			*(rand_it + i) = (double)rand_number[tid];
			lrand48_r(&state_buffers[tid], &rand_number[tid]);
		 	*(rand_it1 + i) = (double)rand_number[tid];
		}
		#pragma omp for private(i)
		for(i = breeding_pop_size; i < population_size; i++) {
		 	unsigned short aa = (unsigned short) (static_cast <double> (rand_vec[i]) /
		 		(static_cast <double> (RAND_MAX/a)));
		 	unsigned short bb = (unsigned short) (static_cast <double> (rand_vec1[i]) /
		 		(static_cast <double> (RAND_MAX/a)));

		 	breeding_population[aa].single_point_cross_over(
		 		breeding_population + bb,
		 		population + i,
		 		individual_size >> 1
		 	);
		}
		}

		rand_it = rand_vec;
		a = population_size;
		while(a--)
			*rand_it++ = rand();
		a = 2 * canvas_size;
		#pragma omp parallel for private(i) reduction(min : a)
		for(i = 0 ; i < population_size ; i++) {
			if(rand_vec[i] < mutation_prob)
				population[i].mutate(mutation_step);
			population[i].assert_fitness();
			if(a > population[i].r_c)
				a = population[i].r_c;
		}

		if(mutation_step > 0.1)
			mutation_step = mutation_step * 0.9;
		gettimeofday(&end, NULL);
		time_sum += 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
		fitness_sum += a;
	}


	a = 2 * canvas_size;
	i = population_size;
	pop_it = population;
	while(i--) {
		if(a > pop_it->r_c) {
			a = pop_it->r_c;
			breeding_it = pop_it;
		}
		pop_it++;
	}

	cout << "after genetic algorithm: " << a << endl;

	breeding_it->dump_to_file_point_representation(
		"individuals_dump.txt",
		ofstream::out,
		20
	);

	#pragma omp parallel for private(i)
	for(i = 0 ; i < population_size ; i++)
		delete[] population[i].circle_vec;

	cout << "Average time per iteration: ";
	for (i = 1; i < time_avgs.size(); ++i)
		time_avgs[0] += time_avgs[i];
	cout << 1.0 * time_avgs[0] / time_avgs.size() << " microseconds" << endl;

	#pragma omp parallel for private(i)
	for(i = 0 ; i < breeding_pop_size ; i++)
		delete[] breeding_population[i].circle_vec;

	return 0;
}
