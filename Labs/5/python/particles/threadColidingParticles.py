import random
import threading
import time
import logging

global loop
global timer_counter
global collisions
global collection_lock

global particle_dimensions

class Dimension:
    def __init__(self):
        self. Height = 0.5
        self.Width = 0.5

class Particle:
    def __init__(self, id):
        self.X = 0
        self.Y = 0
        self.Id = id


def create_particle_array(items):
    particles = []
    for i in range(items):
        particles.append(Particle(i))
    return particles


def move_particle(particle):
    x_move = random.random()
    y_move = random.random()
    particle.X += x_move
    particle.Y += y_move
    if particle.X > 100:
        #print("X reset")
        particle.X = 0
    if particle.Y > 100:
        #print("Y reset")
        particle.Y = 0


def pretty_print_particle(particle, index):
    pretty = str("particle: " + str(particle.Id) + " [X = " + str(particle.X) + "]" + " [Y = " + str(particle.Y) + "]")
    logging.info(pretty)


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
    global collection_lock
    particle = particles[particle_id]
    logging.info("particle ID: %d", particle.Id)
    while loop:
        collection_lock.acquire()
        move_particle(particle)
        pretty_print_particle(particle, particle_id)
        collection_lock.release()


def start_movement_threads(particles):
    threads = []
    particle_count = 0
    for particle in particles:
        t = threading.Thread(target=thread_to_move_row, args=(particle_count, particles,))
        threads.append(t)
        t.start()
        particle_count += 1
    return threads


def basic_collision_check(particle, collider):
    global collisions
    global particle_dimensions
    if particle.X + particle_dimensions.Width < collider.X and collider.X + particle_dimensions.Width > particle.X and particle.Y + particle_dimensions.Height < collider.Y and collider.Y + particle_dimensions.Height > particle.Y:
        collisions += 1
        logging.info("Collision between particle %d and particle %d P1 = [X = %f Y = %f] P2 = [X = %f Y = %f]", particle.Id, collider.Id, particle.X, particle.Y, collider.X, collider.Y)




def check_collision(particles):
    global loop
    global collection_lock
    checks_complete = 0
    while loop:
        collection_lock.acquire()
        for checking in particles:
            for particle in particles:
                if particle.Id != checking.Id:
                    basic_collision_check(checking, particle)
        collection_lock.release()
        checks_complete += 1
    logging.info("collosion checks %d", checks_complete)


def start_collosion_threads(particles):
    threads = []
    t = threading.Thread(target=check_collision, args=(particles,))
    threads.append(t)
    t.start()
    return threads



if __name__ == "__main__":
    collection_lock = threading.Lock()
    particle_dimensions = Dimension()
    logging.basicConfig(format="%(asctime)s:%(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    timer_counter = 0
    loop = True
    collisions = 0
    particles = create_particle_array(5)

    timer = threading.Thread(target=timer_thread_func)
    timer.start()

    movement_threads = start_movement_threads(particles)
    coliding_threads = start_collosion_threads(particles)

    timer.join()

    for t in movement_threads:
        t.join()

    for t in coliding_threads:
        t.join()

    logging.info("___________________________________"
                 "program end "
                 "___________________________________")

