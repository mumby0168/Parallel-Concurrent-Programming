import random
import threading
import time

global loop
global timer_counter


class Particle:
    def __init__(self):
        self.X = 0
        self.Y = 0


def create_particle_array(rows, columns):
    particles = []

    for i in range(rows):
        row = []
        for c in range(columns):
            row.append(Particle())
        particles.append(row)

    return particles


def print_particles(particles):
    row_counter = 0
    for row in particles:
        row_counter += 1
        print("row ", row_counter)
        for particle in row:
            print(particle.X, particle.Y)


def move_particle(particle):
    x_move = random.random()
    y_move = random.random()
    particle.X += x_move
    particle.Y += y_move
    if particle.X > 100:
        print("X reset")
        particle.X = 0
    if particle.Y > 100:
        print("Y reset")
        particle.Y = 0


def pretty_print_particle(particle, row, index):
    pretty = str("row: " + str(row) + " item: " + str(index) + " [X = " + str(particle.X) + "]" + " [Y = " + str(particle.Y) + "]")
    print(pretty)


def move_particles_randomly(particles):
    row_counter = 0
    for rowOfParticles in particles:
        index_counter = 0
        row_counter += 1
        for particle in rowOfParticles:
            index_counter += 1
            move_particle(particle)
            pretty_print_particle(particle, row_counter, index_counter)


def timer_thread_func():
    global loop
    global timer_counter
    print("thread begin")
    while loop:
        print("sleep now")
        time.sleep(1)
        timer_counter += 1
        print("                                   ----------------   ", timer_counter)
        if timer_counter > 10:
            loop = False


if __name__ == "__main__":
    timer_counter = 0
    loop = True
    particles = create_particle_array(10, 10)
    move_particles_randomly(particles)

    timer = threading.Thread(target=timer_thread_func)
    timer.start()

    while loop:
        move_particles_randomly(particles)

    timer.join()
    print("done")

