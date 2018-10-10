import sys
from random import random, randint
import math
from copy import deepcopy
from time import time

import pygame
from pygame.locals import *

def getData(fileName):
	with open(fileName) as f:
	    content = f.readlines()
	content = [x.strip() for x in content]
	data = []
	for line in content:
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
			< (self.radius + circle.radius)**2

	def isOverlappingWithCoords(self, X, Y, radius):
		return (self.X - X)**2 + (self.Y - Y)**2\
			< (self.radius + radius)**2

	def isIdentic(self, circle):
		return self.X == circle.X and\
			self.Y == circle.Y and self.radius == circle.radius

	def printSelf(self):
		print('x=' + str(self.X) + ' y=' + str(self.Y) + ' radiu=' + str(self.radius))

def initIndivid(data, map_dimension=500):
	number_of_circles = sum(map( lambda el: el[1], data ))

	individ = []
	for radius, num in data:
		a = num
		while a != 0:

			while True:
				new_X, new_Y = random() * map_dimension,\
					random() * map_dimension

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

def getIndividFitness(individ, granulation=10):
	sum_X, sum_Y, k_X, k_Y = 0, 0, 0, 0

	points_per_circle = []

	for circle in individ:

		alpha = 0

		for _ in range(granulation):

			points_per_circle.append(( circle.X + circle.radius * math.cos(alpha),\
				circle.Y + circle.radius * math.sin(alpha),))

			sum_X += points_per_circle[-1][0]
			sum_Y += points_per_circle[-1][1]
			k_X += 1
			k_Y += 1

			alpha += (math.pi / granulation)

	enveloping_circ_X = float(sum_X)/k_X
	enveloping_circ_Y = float(sum_Y)/k_Y

	big_radius_squared = (points_per_circle[0][0] - enveloping_circ_X)**2\
		+ (points_per_circle[0][1] - enveloping_circ_Y)**2

	for p_X, p_Y in points_per_circle[1:]:
		new_radius = (p_X - enveloping_circ_X)**2 + (p_Y - enveloping_circ_Y)**2

		if new_radius > big_radius_squared:
			big_radius_squared = new_radius

	return (big_radius_squared, enveloping_circ_X, enveloping_circ_Y,)

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

	big_radius_squared = (points_per_circle[0][0] - 0)**2\
		+ (points_per_circle[0][1] - enveloping_circ_Y)**2

	for p_X, p_Y in points_per_circle[1:]:
		new_radius = (p_X - 0)**2 + (p_Y - enveloping_circ_Y)**2

		if new_radius > big_radius_squared:
			big_radius_squared = new_radius

	return (big_radius_squared, 0, enveloping_circ_Y,)

def roulette_selection(genetic_pool, after_selection_number):
	probs = list(map(lambda el: el[1], genetic_pool))
	s = sum(probs)
	probs = list(map(lambda el: float(el)/s, probs))

	a = float(0)
	intervals = []
	for i in range(len(probs)):
		intervals.append((\
			genetic_pool[i],\
			a,\
			a + probs[i],\
		))
		a += probs[i]

	new_genetic_pool = []
	for _ in range(after_selection_number):
		r = random()
		for individ, a, b in intervals:
			if a <= r < b:
				new_genetic_pool.append(individ)
				break

	return new_genetic_pool

def recombine(parents, copy_from_index=0):
	descendant = deepcopy(parents[0])

	for k in range(len(descendant)):

		new_X, new_Y = parents[copy_from_index][k].X,\
			parents[copy_from_index][k].Y

		didOverlap = False
		for m in range(len(descendant)):
			if m != k and descendant[m\
				].isOverlappingWithCoords(new_X, new_Y,\
				descendant[k].radius):
				didOverlap = True
				break

		if not didOverlap:
			descendant[k].X = new_X
			descendant[k].Y = new_Y

		copy_from_index = (copy_from_index + 1) % 2

	isNewIndivid = False
	for k in range(len(descendant)):
		if not descendant[k].isIdentic(parents[0][k]):
			isNewIndivid = True
			break

	if isNewIndivid:
		return descendant
	return None

def crossover(genetic_pool, mating_rights_percentage=0.1, fitnessFunc=getIndividFitnessSemiCircle):
	mating_rights_len = int(len(genetic_pool) + mating_rights_percentage)

	rez = []

	for i in range(mating_rights_len - 1):
		for j in range(i + 1, mating_rights_len):

			descendant = recombine((genetic_pool[i][0], genetic_pool[j][0],))

			if descendant: rez.append((descendant,) + fitnessFunc(descendant))

			descendant = recombine((genetic_pool[i][0], genetic_pool[j][0],), 1)

			if descendant: rez.append((descendant,) + fitnessFunc(descendant))

	return rez

def mutation(descendants, percentage=0.5, delta_dist=1, delta_angle=math.pi/10, fitnessFunc=getIndividFitnessSemiCircle):
	new_decendants = []
	for individ, fitness, enveloping_circ_X, enveloping_circ_Y in descendants:
		if random() < percentage:

			new_individ = deepcopy(individ)

			k = 0
			for circle in new_individ:

				d_AB = math.sqrt((enveloping_circ_X - circle.X)**2 +\
					(enveloping_circ_Y - circle.Y)**2)

				new_X = delta_dist * ( enveloping_circ_X - circle.X ) / d_AB\
					+ circle.X

				new_Y = delta_dist * ( enveloping_circ_Y - circle.Y ) / d_AB\
					+ circle.Y

				# didOverlap = False
				# for m in range(len(new_individ)):
				# 	if m != k and new_individ[m\
				# 		].isOverlappingWithCoords(new_X, new_Y,\
				# 		new_individ[k].radius):
				# 		didOverlap = True
				# 		break

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
					angle = math.atan2(circle.X - enveloping_circ_X,circle.Y - enveloping_circ_Y)

					new_X = enveloping_circ_X + math.cos(angle - delta_angle) * d_AB
					new_Y = enveloping_circ_Y + math.sin(angle - delta_angle) * d_AB

					# didOverlap = False
					# for m in range(len(new_individ)):
					# 	if m != k and new_individ[m\
					# 		].isOverlappingWithCoords(new_X, new_Y,\
					# 		new_individ[k].radius):
					# 		didOverlap = True
					# 		break

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
						new_X = enveloping_circ_X + math.cos(angle + delta_angle) * d_AB
						new_Y = enveloping_circ_Y + math.sin(angle + delta_angle) * d_AB

						# didOverlap = False
						# for m in range(len(new_individ)):
						# 	if m != k and new_individ[m\
						# 		].isOverlappingWithCoords(new_X, new_Y,\
						# 		new_individ[k].radius):
						# 		didOverlap = True
						# 		break

						didOverlap = False
						for m in range(len(new_individ)):
							if m != k and (new_individ[m\
								].isOverlappingWithCoords(new_X, new_Y,\
								new_individ[k].radius) or new_X < circle.radius):
								didOverlap = True
								break

						if not didOverlap:
							circle.X, circle.Y = new_X, new_Y

				if circle.Y > enveloping_circ_Y:
					new_Y -= delta_dist
				elif circle.Y < enveloping_circ_Y:
					new_Y += delta_dist
				else:
					k += 1
					continue

				didOverlap = False
				for m in range(len(new_individ)):
					if m != k and (new_individ[m\
						].isOverlappingWithCoords(circle.X, new_Y,\
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

#TO DO
def tighten(new_individ, delta_dist, enveloping_circ_X, enveloping_circ_Y, delta_angle=math.pi/10):
	iter_n = 0

	while iter_n < 20000:
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

			# didOverlap = False
			# for m in range(len(new_individ)):
			# 	if m != k and new_individ[m\
			# 		].isOverlappingWithCoords(new_X, new_Y,\
			# 		new_individ[k].radius):
			# 		didOverlap = True
			# 		break

			didOverlap = False
			for m in range(len(new_individ)):
				if m != k and (new_individ[m\
					].isOverlappingWithCoords(new_X, new_Y,\
					new_individ[k].radius) or new_X < circle.radius):
					didOverlap = True
					break

			if not didOverlap:
				circle.X, circle.Y = new_X, new_Y
				if not did_thighten: did_thighten = True
			else:
				angle = math.atan2(circle.X - enveloping_circ_X,circle.Y - enveloping_circ_Y)

				new_X = enveloping_circ_X + math.cos(angle - delta_angle) * d_AB
				new_Y = enveloping_circ_Y + math.sin(angle - delta_angle) * d_AB

				# didOverlap = False
				# for m in range(len(new_individ)):
				# 	if m != k and new_individ[m\
				# 		].isOverlappingWithCoords(new_X, new_Y,\
				# 		new_individ[k].radius):
				# 		didOverlap = True
				# 		break

				didOverlap = False
				for m in range(len(new_individ)):
					if m != k and (new_individ[m\
						].isOverlappingWithCoords(new_X, new_Y,\
						new_individ[k].radius) or new_X < circle.radius):
						didOverlap = True
						break

				if not didOverlap:
					circle.X, circle.Y = new_X, new_Y
					if not did_thighten: did_thighten = True
				else:
					new_X = enveloping_circ_X + math.cos(angle + delta_angle) * d_AB
					new_Y = enveloping_circ_Y + math.sin(angle + delta_angle) * d_AB

					# didOverlap = False
					# for m in range(len(new_individ)):
					# 	if m != k and new_individ[m\
					# 		].isOverlappingWithCoords(new_X, new_Y,\
					# 		new_individ[k].radius):
					# 		didOverlap = True
					# 		break

					didOverlap = False
					for m in range(len(new_individ)):
						if m != k and (new_individ[m\
							].isOverlappingWithCoords(new_X, new_Y,\
							new_individ[k].radius) or new_X < circle.radius):
							didOverlap = True
							break

					if not didOverlap:
						circle.X, circle.Y = new_X, new_Y
						if not did_thighten: did_thighten = True


			k+=1

		if not did_thighten: break

	# print(iter_n)

	iter_n = 0

	while iter_n < 50000:
		iter_n += 1

		did_thighten = False

		k = 0
		for circle in new_individ:
			if circle.Y > enveloping_circ_Y:
				new_Y -= delta_dist
			elif circle.Y < enveloping_circ_Y:
				new_Y += delta_dist
			else:
				k += 1
				continue

			didOverlap = False
			for m in range(len(new_individ)):
				if m != k and (new_individ[m\
					].isOverlappingWithCoords(circle.X, new_Y,\
					new_individ[k].radius) or new_X < circle.radius):
					didOverlap = True
					break

			if not didOverlap:
				circle.X, circle.Y = new_X, new_Y
				if not did_thighten: did_thighten = True

		if not did_thighten: break


	return new_individ

if __name__ == "__main__" :
	start_time = time()

	circles_data = getData(sys.argv[1])

	max_radius = max(map(lambda el: el[0], circles_data))

	# Parameters definition
	genetic_pool_dim = 40
	number_of_iterations = 20
	selection_pressure = 0.5
	space_dim = 50
	screen_dim = 700
	mutation_step = float(1.5)
	mutation_prob = 0.5
	fitnessFunc=getIndividFitnessSemiCircle

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

		# genetic_pool = genetic_pool[0:after_selection_number]
		genetic_pool = roulette_selection(genetic_pool, after_selection_number)

		genetic_pool.sort(key=lambda el: el[1])
		if i % (number_of_iterations / 10) == 0:
			print(str(i) + ': enveloping_circle_radius=' + str(math.sqrt(genetic_pool[0][1])))

		descendants = crossover(genetic_pool)

		genetic_pool.extend(descendants)

		genetic_pool.extend(mutation(descendants, mutation_prob, mutation_step))

		if i % (number_of_iterations / 10) == 0:
			print(str(i) + ': mutation_step=' + str(mutation_step))
			if mutation_step > 0.05:
				mutation_step *= 0.6
			# mutation_prob *= 1.1

	genetic_pool.sort(key=lambda el: el[1])

	best_individ = tighten(genetic_pool[0][0], 0.05, genetic_pool[0][-2], genetic_pool[0][-1])

	best_individ = tighten(best_individ, 0.01, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten(best_individ, 0.0000001, genetic_pool[0][-2], genetic_pool[0][-1])
	best_individ = tighten(best_individ, 0.000000000001, genetic_pool[0][-2], genetic_pool[0][-1])

	print('enveloping circle radius: ' + str(math.sqrt(genetic_pool[0][1])))
	print('elapsed time: ' + str((time() - start_time)/60))
	print('mutation step: ' + str(mutation_step))

	pygame.init()
	screen = pygame.display.set_mode((screen_dim, screen_dim))
	done = False

	offset = (max_radius / space_dim) * screen_dim

	new_screen_dim = screen_dim - offset

	while not done:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				done = True

		for circle in best_individ:
			pygame.draw.circle(screen, (0, 128, 255),\
				(\
					int((circle.X / space_dim) * new_screen_dim ),\
					int((circle.Y / space_dim) * new_screen_dim )\
				),\
				int( (circle.radius / space_dim) * new_screen_dim ))

		pygame.display.flip()
