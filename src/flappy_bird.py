import neat
from src.classes_and_configuration.classes import *



pygame.font.init()
WIDTH = 1000
HEIGHT = 850
FLOOR = 730
STAT_FONT = pygame.font.SysFont("impact", 30)
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
bg_img = pygame.transform.scale(pygame.image.load("imgs/bg.png").convert_alpha(), (WIDTH, HEIGHT))
pipe_img = pygame.transform.scale2x(pygame.image.load("imgs/pipe.png").convert_alpha())
bird_images = [pygame.transform.scale2x(pygame.image.load("imgs/" + "bird" + str(x) + ".png")) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load("imgs/base.png").convert_alpha())
pygame.display.set_caption("Flappy Bird")
gen = 0


def draw_window(win, birds, pipes, base, score, gen):
    if gen == 0:
        gen = 1
    win.blit(bg_img, (0,0))
    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    score_label = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(score_label, (10, 10) )
    gen_label = STAT_FONT.render("Gens: " + str(gen - 1), 1, (255,255,255))
    win.blit(gen_label, (10, 45))
    pygame.display.update()

def eval_genomes(genomes, config):
    global WIN, gen
    win = WIN
    gen += 1
    network = []
    birds = []
    generation = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        network.append(net)
        birds.append(Bird(230, 350, bird_images))
        generation.append(genome)
    base = Base(base_img, FLOOR)
    pipes = [Pipe(700, pipe_img)]
    score = 0
    clock = pygame.time.Clock()
    run = True
    while run and len(birds) > 0:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        for x, bird in enumerate(birds):
            generation[x].fitness += 0.1
            bird.move()
            output = network[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()
        base.move()
        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for bird in birds:
                if pipe.collide(bird, win):
                    generation[birds.index(bird)].fitness -= 1
                    network.pop(birds.index(bird))
                    generation.pop(birds.index(bird))
                    birds.pop(birds.index(bird))
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
        if add_pipe:
            score += 1
            for genome in generation:
                genome.fitness += 5
            pipes.append(Pipe(WIDTH, pipe_img) )
        for r in rem:
            pipes.remove(r)
        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                network.pop(birds.index(bird))
                generation.pop(birds.index(bird))
                birds.pop(birds.index(bird))
        draw_window(WIN, birds, pipes, base, score, gen)

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.run(eval_genomes, 50)


if __name__ == '__main__':
    run( 'classes_and_configuration/feed_forward.txt' )