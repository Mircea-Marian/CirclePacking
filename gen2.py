import sys
from random import random, randint
import math
from copy import deepcopy
from time import time
from genetic import roulette_selection, recombine,\
	crossover, Circle, recombine

from multiprocessing import Process, Pipe

import pygame
from pygame.locals import *

import pickle

MUTATE_OP = 0
EXIT_OP = 1
RETURN_EMPTY_OP = 2
COMBINE_OP = 3

def getData(fileName):
	with open(fileName) as f:
	    content = f.readlines()
	content = [x.strip() for x in content]
	data = []
	for line in content[1:]:
		cc = line.split(' ')
		data.append( ( int(cc[0]), int(cc[1]), ) )
	return data

class Circle:
	def __init__(self, X=None, Y=None, radius=None):
		self.X = X
		self.Y = Y
		self.radius = radius

	def isOverlappingWithCircle(self, circle):
		return (self.X - circle.X)**2 + (self.Y - circle.Y)**2\
			<= (self.radius + circle.radius)**2

	def isOverlappingWithCoords(self, X, Y, radius):
		return (self.X - X)**2 + (self.Y - Y)**2\
			<= (self.radius + radius)**2

	def isIdentic(self, circle):
		return self.X == circle.X and\
			self.Y == circle.Y and self.radius == circle.radius

	def printSelf(self):
		print('x=' + str(self.X) + ' y=' + str(self.Y) + ' radiu=' + str(self.radius))

def isIndividValid(individual):
	for i in range(len(individual) - 1):
		for j in range(i + 1, len(individual)):
			if individual[i].isOverlappingWithCircle(individual[j]):
				return False
	return True

def initIndivid(data, map_dimension=500):
	number_of_circles = sum(map( lambda el: el[1], data ))

	individ = []
	for radius, num in data:
		a = num
		while a != 0:

			while True:
				new_X = random() * map_dimension

				if new_X < radius:
					continue

				new_Y = random() * map_dimension

				didOverlap = False
				for circle in individ:
					if circle.isOverlappingWithCoords(new_X, new_Y, radius):
						didOverlap = True
						break

				if not didOverlap:
					individ.append(Circle(new_X, new_Y, radius))
					break

			a -= 1

	return individ

def getIndividFitnessSemiCircle(individ, cent_X=0, granulation=10):
	sum_Y, k_Y = 0, 0

	points_per_circle = []

	for circle in individ:

		alpha = 0

		for _ in range(granulation):

			points_per_circle.append(( circle.X + circle.radius * math.cos(alpha),\
				circle.Y + circle.radius * math.sin(alpha),))

			sum_Y += points_per_circle[-1][1]

			k_Y += 1

			alpha += (math.pi / granulation)

	enveloping_circ_Y = float(sum_Y)/k_Y

	big_radius_squared = points_per_circle[0][0]**2\
		+ (points_per_circle[0][1] - enveloping_circ_Y)**2

	for p_X, p_Y in points_per_circle[1:]:
		new_radius = p_X**2 + (p_Y - enveloping_circ_Y)**2

		if new_radius > big_radius_squared:
			big_radius_squared = new_radius

	return (big_radius_squared, 0, enveloping_circ_Y,)

def mutation(descendants, percentage=0.5, delta_dist=1, delta_angle=math.pi/10, fitnessFunc=getIndividFitnessSemiCircle,\
	single_circle_mut_prob=0.5):
	new_decendants = []
	for individ, fitness, enveloping_circ_X, enveloping_circ_Y in descendants:
		if random() < percentage:

			new_individ = deepcopy(individ)

			k = 0
			for circle in new_individ:

				# if random() >= single_circle_mut_prob:
				# 	k += 1
				# 	continue


				d_AB = math.sqrt(circle.X**2 +\
					(enveloping_circ_Y - circle.Y)**2)

				new_X = (d_AB - delta_dist) * circle.X / d_AB
				new_Y = enveloping_circ_Y + (d_AB - delta_dist) * (circle.Y - enveloping_circ_Y) / d_AB

				didOverlap = False
				hitLeftWall = False
				for m in range(len(new_individ)):
					if m != k\
						and (new_individ[m].isOverlappingWithCoords(\
								new_X, new_Y, new_individ[k].radius)\
							or new_X < circle.radius
						):
						if new_X < circle.radius:
							hitLeftWall = True
						didOverlap = True
						break

				if not didOverlap:
					circle.X, circle.Y = new_X, new_Y
				else:
					hitLFandMutated = False
					if hitLeftWall:
						new_X = circle.radius
						if circle.Y > enveloping_circ_Y:
							new_Y = circle.Y - delta_dist
						else:
							new_Y = circle.Y + delta_dist

						didOverlap = False
						for m in range(len(new_individ)):
							if m != k and new_individ[m\
								].isOverlappingWithCoords(new_X, new_Y,\
								new_individ[k].radius):
								didOverlap = True
								break

						if not didOverlap:
							circle.X, circle.Y = new_X, new_Y
							hitLFandMutated = True

					if not hitLFandMutated:
						angle = math.atan2(circle.X - enveloping_circ_X,circle.Y - enveloping_circ_Y)

						new_X = math.cos(angle - delta_angle) * d_AB
						new_Y = enveloping_circ_Y + math.sin(angle - delta_angle) * d_AB

						didOverlap = False
						for m in range(len(new_individ)):
							if m != k and (new_individ[m\
								].isOverlappingWithCoords(new_X, new_Y,\
								new_individ[k].radius) or new_X < circle.radius):
								didOverlap = True
								break

						if not didOverlap:
							circle.X, circle.Y = new_X, new_Y
						else:
							new_X = math.cos(angle + delta_angle) * d_AB
							new_Y = enveloping_circ_Y + math.sin(angle + delta_angle) * d_AB

							didOverlap = False
							for m in range(len(new_individ)):
								if m != k and (new_individ[m\
									].isOverlappingWithCoords(new_X, new_Y,\
									new_individ[k].radius) or new_X < circle.radius):
									didOverlap = True
									break

							if not didOverlap:
								circle.X, circle.Y = new_X, new_Y

				k+=1

			isNewIndivid = False
			for k in range(len(new_individ)):
				if not new_individ[k].isIdentic(individ[k]):
					isNewIndivid = True
					break

			if isNewIndivid:
				new_decendants.append((new_individ,) + fitnessFunc(new_individ))
	return new_decendants

def tighten(new_individ, delta_dist, enveloping_circ_X, enveloping_circ_Y, delta_angle=math.pi/10):
	iter_n = 0

	while iter_n < 10000:
		iter_n += 1

		did_thighten = False

		k = 0
		for circle in new_individ:

			d_AB = math.sqrt((enveloping_circ_X - circle.X)**2 +\
				(enveloping_circ_Y - circle.Y)**2)

			new_X = delta_dist * ( enveloping_circ_X - circle.X ) / d_AB\
				+ circle.X

			new_Y = delta_dist * ( enveloping_circ_Y - circle.Y ) / d_AB\
				+ circle.Y

			didOverlap = False
			hitLeftWall = False
			for m in range(len(new_individ)):
				if m != k and (new_individ[m\
					].isOverlappingWithCoords(new_X, new_Y,\
					new_individ[k].radius) or new_X < circle.radius):
					if new_X < circle.radius:
						hitLeftWall = True
					didOverlap = True
					break

			if not didOverlap:
				circle.X, circle.Y = new_X, new_Y
				did_thighten = True
			elif hitLeftWall:
				new_X = circle.radius
				if circle.Y > enveloping_circ_Y:
					new_Y = circle.Y - delta_dist
				else:
					new_Y = circle.Y + delta_dist

				didOverlap = False
				for m in range(len(new_individ)):
					if m != k and new_individ[m\
						].isOverlappingWithCoords(new_X, new_Y,\
						new_individ[k].radius):
						didOverlap = True
						break

				if not didOverlap:
					circle.X, circle.Y = new_X, new_Y
					did_thighten = True

		if not did_thighten:
			break

	print(iter_n)
	return new_individ

def tighten2(new_individ, delta_dist, enveloping_circ_X, enveloping_circ_Y, delta_angle=math.pi/10):
	iter_n = 0

	while iter_n < 60000:
		iter_n += 1

		did_thighten = False

		k = 0
		for circle in new_individ:

			d_AB = math.sqrt((enveloping_circ_X - circle.X)**2 +\
				(enveloping_circ_Y - circle.Y)**2)

			new_X = delta_dist * ( enveloping_circ_X - circle.X ) / d_AB\
				+ circle.X

			new_Y = delta_dist * ( enveloping_circ_Y - circle.Y ) / d_AB\
				+ circle.Y

			didOverlap = False
			for m in range(len(new_individ)):
				if m != k and (new_individ[m\
					].isOverlappingWithCoords(new_X, new_Y,\
					new_individ[k].radius) or new_X < circle.radius):
					didOverlap = True
					break

			if not didOverlap:
				circle.X, circle.Y = new_X, new_Y
				did_thighten = True
			else:
				new_X = circle.X
				if circle.Y > enveloping_circ_Y:
					new_Y = circle.Y - delta_dist
				else:
					new_Y = circle.Y + delta_dist

				didOverlap = False
				for m in range(len(new_individ)):
					if m != k and (new_individ[m\
						].isOverlappingWithCoords(new_X, new_Y,\
						new_individ[k].radius) or new_X < circle.radius):
						didOverlap = True
						break

				if not didOverlap:
					circle.X, circle.Y = new_X, new_Y
					did_thighten = True
				else:
					if circle.X > enveloping_circ_X:
						new_X = circle.X - delta_dist
					else:
						new_X = circle.X + delta_dist
					new_Y = circle.Y

					didOverlap = False
					for m in range(len(new_individ)):
						if m != k and (new_individ[m\
							].isOverlappingWithCoords(new_X, new_Y,\
							new_individ[k].radius) or new_X < circle.radius):
							didOverlap = True
							break

					if not didOverlap:
						circle.X, circle.Y = new_X, new_Y
						did_thighten = True

		if not did_thighten:
			break

	print(iter_n)
	return new_individ

def tighten3(new_individ, delta_dist, enveloping_circ_X, enveloping_circ_Y, delta_angle=math.pi/10):
	iter_n = 0

	new_individ = list(map(\
		lambda circle: [circle, math.sqrt((enveloping_circ_X - circle.X)**2 +\
			(enveloping_circ_Y - circle.Y)**2),],\
		new_individ\
	))
	new_individ.sort(key=lambda el: el[1])

	# while iter_n < 60000:
	while True:
		iter_n += 1

		did_thighten = False

		k = 0
		for circle, d_AB in new_individ:

			new_X = (d_AB - delta_dist) * circle.X / d_AB
			new_Y = enveloping_circ_Y + (d_AB - delta_dist) * (circle.Y - enveloping_circ_Y) / d_AB

			didOverlap = False
			hitLeftWall = False
			for m in range(len(new_individ)):
				if m != k and (new_individ[m\
					][0].isOverlappingWithCoords(new_X, new_Y,\
					new_individ[k][0].radius) or new_X < circle.radius):
					if new_X < circle.radius:
						hitLeftWall = True
					didOverlap = True
					break

			if not didOverlap:
				circle.X, circle.Y = new_X, new_Y
				did_thighten = True
			elif hitLeftWall:
				new_X = circle.radius
				if circle.Y > enveloping_circ_Y:
					new_Y = circle.Y - delta_dist
				else:
					new_Y = circle.Y + delta_dist

				didOverlap = False
				for m in range(len(new_individ)):
					if m != k and new_individ[m\
						][0].isOverlappingWithCoords(new_X, new_Y,\
						new_individ[k][0].radius):
						didOverlap = True
						break

				if not didOverlap:
					circle.X, circle.Y = new_X, new_Y
					did_thighten = True

		if not did_thighten:
			break

		for i in range(len(new_individ)):
			new_individ[i][1] = math.sqrt((enveloping_circ_X - circle.X)**2 +\
				(enveloping_circ_Y - circle.Y)**2)

		if iter_n % 10000 == 0:
			print(str(delta_dist) + ': ' + str(iter_n))
			pr = '\t'
			for circle, val in new_individ:
				pr += str(val) + ' '
			print(pr)

	print(iter_n)
	return  list(map(lambda el: el[0], new_individ))

def procJob(con):
	while True:
		op = con.recv()
		if op == MUTATE_OP:
			mutation_prob = con.recv()
			mutation_step = con.recv()
			con.send(mutation(con.recv(), mutation_prob, mutation_step))
		elif EXIT_OP == op:
			break
		elif op == RETURN_EMPTY_OP:
			con.send([])
		elif op == COMBINE_OP:
			genetic_pool = con.recv()
			rez = []
			for i, j in con.recv():
				descendant = recombine((genetic_pool[i], genetic_pool[j],))

				if descendant: rez.append((descendant,) + getIndividFitnessSemiCircle(descendant))

				descendant = recombine((genetic_pool[i], genetic_pool[j],), 1)

				if descendant: rez.append((descendant,) + getIndividFitnessSemiCircle(descendant))
			con.send(rez)

def crossover(genetic_pool, pipe_and_proc, mating_rights_percentage=0.1):
	mating_rights_len = int(len(genetic_pool) + mating_rights_percentage)
	proc_no = len(pipe_and_proc)

	parents = []

	for i in range(mating_rights_len - 1):
		for j in range(i + 1, mating_rights_len):
			parents.append((i, j,))

	quotient, remainder = divmod(len(parents), proc_no)

	assigned_nums = [quotient for _ in range(proc_no)]
	for a in range(remainder):
		assigned_nums[a] += 1

	min_gen_pool = list(map( lambda el: el[0], genetic_pool[:mating_rights_len] ))

	a = 0
	for iii in range(proc_no):

		if assigned_nums[iii] != 0:
			pipe_and_proc[iii][0].send(COMBINE_OP)
			pipe_and_proc[iii][0].send(min_gen_pool)
			pipe_and_proc[iii][0].send(parents[a:a + assigned_nums[iii]])
		else: pipe_and_proc[iii][0].send(RETURN_EMPTY_OP)

		a += assigned_nums[iii]

	rez = []
	for pipe, proc in pipe_and_proc:
		rez.extend(pipe.recv())

	return rez

def getMutationStepsLinear(intial_value, number_of_iterations, lower_limit):
	step = (intial_value - lower_limit) / number_of_iterations

	rez = []
	for i in range(number_of_iterations):
		rez.append(intial_value)
		intial_value -= step
	return rez

def drawCircles(best_individ, max_radius):
	screen_dim = 700

	y_offset = 100

	pygame.init()
	screen = pygame.display.set_mode((screen_dim, screen_dim))
	done = False

	screen_dim = 500

	offset = (max_radius / space_dim) * screen_dim

	new_screen_dim = screen_dim - offset

	while not done:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True


		pygame.draw.circle(screen, (255, 0, 0),\
			(\
				int((enveloping_circ_X / space_dim) * new_screen_dim ),\
				int(y_offset + (enveloping_circ_Y / space_dim) * new_screen_dim )\
			),\
			int( (1.1 * enveloping_circ_radius / space_dim) * new_screen_dim ))
		pygame.draw.circle(screen, (0, 0, 0),\
			(\
				int((enveloping_circ_X / space_dim) * new_screen_dim ),\
				int(y_offset + (enveloping_circ_Y / space_dim) * new_screen_dim )\
			),\
			int( (1 * enveloping_circ_radius / space_dim) * new_screen_dim ))

		pygame.draw.circle(screen, (0, 255, 0),\
			(\
				int((enveloping_circ_X / space_dim) * new_screen_dim ),\
				int(y_offset + (enveloping_circ_Y / space_dim) * new_screen_dim )\
			),\
			int( (2 / space_dim) * new_screen_dim ))

		for circle in best_individ:
			pygame.draw.circle(screen, (0, 128, 255),\
				(\
					int((circle.X / space_dim) * new_screen_dim ),\
					int(y_offset + (circle.Y / space_dim) * new_screen_dim )\
				),\
				int( (circle.radius / space_dim) * new_screen_dim ))
			pygame.draw.circle(screen, (0, 0, 0),\
				(\
					int((circle.X / space_dim) * new_screen_dim ),\
					int(y_offset + (circle.Y / space_dim) * new_screen_dim )\
				),\
				int( (0.95 * circle.radius / space_dim) * new_screen_dim ))

		pygame.display.flip()

if __name__ == "__main__" :
	start_time = time()

	circles_data = getData(sys.argv[1])

	max_radius = max(map(lambda el: el[0], circles_data))

	# Parameters definition
	genetic_pool_dim = 100
	number_of_iterations = 200
	selection_pressure = 0.5
	space_dim = 70
	mutation_step = float(5)
	lower_limit_step = 0.05
	mutation_prob = 0.8
	fitnessFunc=getIndividFitnessSemiCircle
	proc_no = 8
	mating_rights_percentage=0.1

	mutation_steps = getMutationStepsLinear(mutation_step, number_of_iterations, lower_limit_step)

	pipe_and_proc = []
	for _ in range(proc_no):
		parent_conn, child_conn = Pipe()

		p = Process(target=procJob,\
			args=(\
				child_conn,\
			))
		p.start()

		pipe_and_proc.append((parent_conn, p,))

	genetic_pool = []
	for _ in range(genetic_pool_dim):
		individ = initIndivid(circles_data, space_dim)
		genetic_pool.append((individ,) + fitnessFunc(individ))

	print('And God said: Let there be the first generation !')

	after_selection_number = int(selection_pressure * genetic_pool_dim)

	for i in range(number_of_iterations):

		while len(genetic_pool) < after_selection_number:

			ind = randint() % len(genetic_pool)

			genetic_pool.append(deepcopy(genetic_pool[ind]))


		genetic_pool.sort(key=lambda el: el[1])

		genetic_pool = genetic_pool[0:after_selection_number]
		# genetic_pool = roulette_selection(genetic_pool, after_selection_number)

		genetic_pool.sort(key=lambda el: el[1])
		if i % (number_of_iterations / 10) == 0:
			print(str(i) + ': enveloping_circle_radius=' + str(math.sqrt(genetic_pool[0][1])))

		descendants = crossover(genetic_pool, pipe_and_proc, mating_rights_percentage)

		# genetic_pool.extend(descendants)

		# genetic_pool.extend(mutation(descendants, mutation_prob, mutation_step))

		quotient, remainder = divmod(len(descendants), proc_no)

		assigned_nums = [quotient for _ in range(proc_no)]
		for a in range(remainder):
			assigned_nums[a] += 1

		a = 0
		for iii in range(proc_no):

			if assigned_nums[iii] != 0:
				pipe_and_proc[iii][0].send(MUTATE_OP)
				pipe_and_proc[iii][0].send(mutation_prob)
				pipe_and_proc[iii][0].send(mutation_steps[i])
				pipe_and_proc[iii][0].send(descendants[a:a+assigned_nums[iii]])
			else: pipe_and_proc[iii][0].send(RETURN_EMPTY_OP)

			a += assigned_nums[iii]

		for pipe, proc in pipe_and_proc:
			genetic_pool.extend(pipe.recv())

		if i % (number_of_iterations / 10) == 0:
			print(str(i) + ': mutation_step=' + str(mutation_steps[i]))
			# if mutation_step > 0.05:
			# 	mutation_step *= 0.6
			# mutation_prob *= 1.1

	for pipe, proc in pipe_and_proc:
		pipe.send(EXIT_OP)

	for pipe, proc in pipe_and_proc:
		proc.join()

	genetic_pool.sort(key=lambda el: el[1])

	best_individ = genetic_pool[0][0]

	best_individ = tighten3(best_individ, 10, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 1, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.1, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.01, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.0001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.00001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.0000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.00000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.000000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.0000000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.00000000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.000000000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten3(best_individ, 0.0000000000001, genetic_pool[0][-2], genetic_pool[0][-1])

	print(isIndividValid(best_individ))

	enveloping_circ_radius, enveloping_circ_X, enveloping_circ_Y = getIndividFitnessSemiCircle(best_individ,\
		granulation=200)
	enveloping_circ_radius = math.sqrt(enveloping_circ_radius)

	# print('enveloping circle radius: ' + str(math.sqrt(genetic_pool[0][1])))
	print('enveloping circle radius: ' + str(enveloping_circ_radius))
	print('elapsed time: ' + str((time() - start_time)/60))

	drawCircles(best_individ, max_radius)
