import random
import threading
import time
import logging

global loop
global timer_counter


class Particle:
    def __init__(self):
        self.X = 0
        self.Y = 0                


def create_particle_array(items):
    particles = []
    for i in range(items):
        particles.append(Particle())
    return particles


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


def pretty_print_particle(particle, index):
    pretty = str("particle: " + str(index) + " [X = " + str(particle.X) + "]" + " [Y = " + str(particle.Y) + "]")
    print(pretty)


def move_particles_randomly(particles):
    particle_counter = 0
    for particle in particles:
        move_particle(particle)
        pretty_print_particle(particle, particle_counter)


def timer_thread_func():
    global loop
    global timer_counter
    print("thread begin")
    while loop:
        print("sleep now")
        time.sleep(1)
        timer_counter += 1
        if timer_counter > 10:
            loop = False


def thread_to_move_row(particle_id, particles):
    logging.info("index: %d", particle_id)
    particle = particles[particle_id]
    while loop:
        move_particle(particle)
        pretty_print_particle(particle, particle_id)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s:%(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    timer_counter = 0
    loop = True
    particles = create_particle_array(20)
    move_particles_randomly(particles)

    timer = threading.Thread(target=timer_thread_func)
    timer.start()

    threads = []
    particle_count = 0
    for particle in particles:
        t = threading.Thread(target=thread_to_move_row, args=(particle_count, particles,))
        threads.append(t)
        t.start()
        particle_count += 1

    timer.join()

    for t in threads:
        t.join()

    logging.info("___________________________________"
                 "program end "
                 "___________________________________")

