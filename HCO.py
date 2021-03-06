import numpy as np
import cifar10_multi_gpu_train as ct
import cifar10_eval as ce
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#  parameter setting
ld = np.array([50, 50, 50, 2, 2, 2, 2, 2, 2, 0, 0, 0])  # low boundary
ud = np.array([180, 180, 180, 7, 7, 7, 7, 7, 7, 1, 1, 1])  # up  boundary
dim = 12
# the num of parameter should be optimized
# convolution: feature map, pad, kernel size pool: max|average  kernel size
pop_size = 15  # population size

max_evaluation = 600
run_time = 2
max_flow = 3  # the max count of flow
p_eva = 0.2  # evaporation probability
best_record = []
best_record_inter = []
best_position = []
best_position_inter = []


def hco():
    for r in range(1, run_time):
        evaluation_now = 0
        pop_pos = np.random.rand(pop_size, dim)  # population position
        pop_pos = np.add(ld, np.multiply(pop_pos, np.subtract(ud, ld)))
        fitness = np.zeros(pop_size, dtype=float)  # initialize and calculate the fitness
        bndry_proce_pop(pop_pos)
        calc_fitness_pop(pop_pos, fitness, evaluation_now)
        best_index = np.argmax(fitness)  # get the index of the best performance individual
        evaluation_now += pop_size
        best_record.append(fitness[best_index])
        best_position.append(pop_pos[best_index])

        while evaluation_now <= max_evaluation:
            # execute the flow operation
            best_record_inter.append(fitness[best_index])
            best_position_inter.append(pop_pos[best_index])
            evaluation_now = flow(pop_pos, fitness, best_index, evaluation_now)
            # update the best index and position of population
            # execute the infiltration operation
            best_index = np.argmax(fitness)
            best_record_inter.append(fitness[best_index])
            best_position_inter.append(pop_pos[best_index])
            #
            evaluation_now = infiltration(pop_pos, fitness, evaluation_now)
            #
            best_index = np.argmax(fitness)
            best_record_inter.append(fitness[best_index])
            best_position_inter.append(pop_pos[best_index])
            # execute the evaporation and precipitation operation
            evaluation_now = eva_and_precip(pop_pos, fitness, best_index, evaluation_now)
            best_index = np.argmax(fitness)
    fo1 = open("/Users/wangjun/Documents/study/Graduation_Project/HCOCNN/result.txt", "w")
    fo2 = open("/Users/wangjun/Documents/study/Graduation_Project/HCOCNN/result_inter.txt", "w")
    fo3 = open("/Users/wangjun/Documents/study/Graduation_Project/HCOCNN/result_position.txt", "w")
    fo4 = open("/Users/wangjun/Documents/study/Graduation_Project/HCOCNN/result_position_inter.txt", "w")
    fo1.write(str(best_record))
    fo2.write(str(best_record_inter))
    fo3.write(str(best_position))
    fo4.write(str(best_position_inter))
    fo1.close()
    fo2.close()
    fo3.close()
    fo4.close()


def calc_fitness_pop(pop_pos, fitness, evaluation_now):
    # this function completes the flow operation
    # Args: pop_pos: population position, individual should be a row vector
    # return: fitness, a numpy array with [row,1] where row is the row of pop_pos
    # calculate the fitness for every individual
    for index, idv in enumerate(pop_pos):
        fitness[index] = cal_fitness(idv, evaluation_now)


def flow(pop_pos, fitness, best_index, evaluation_now):
    # this function completes the flow operation
    # Args: pop_pos: population position, individual should be a row vector
    # fitness: the fitness value of population
    # best_index: the best individual position index
    # evaluation_now: global control parameter
    row, col = np.shape(pop_pos)
    for index, idv in enumerate(pop_pos):
        # randomly choose a direction to flow, should not the same as itself
        flow_time = 0
        while flow_time < max_flow:
            if fitness[index] == fitness[best_index]:
                flow_direction = np.random.randint(0, row)
                while flow_direction == index or fitness[flow_direction] == fitness[best_index]:
                    flow_direction = np.random.randint(0, row)
            else:
                better_index = np.where(fitness > fitness[index])
                flow_direction = better_index[0][np.random.randint(0, np.size(better_index[0]))]
            # calculate new individual and its fitness
            new_idv = idv + np.multiply(np.random.rand(col),
                                        np.subtract(pop_pos[flow_direction], idv))
            new_idv = bndry_proce(new_idv)
            new_fitness = cal_fitness(new_idv, evaluation_now)
            evaluation_now += 1
            # only update the fitness when the new individual perform better than the old one
            if new_fitness > fitness[index]:
                pop_pos[index] = new_idv
                fitness[index] = new_fitness
                flow_time += 1
            else:
                flow_time = max_flow
            # update the control parameter
            best_index = np.argmax(fitness)
            best_record.append(fitness[best_index])
            best_position.append(pop_pos[best_index])
    return evaluation_now


def infiltration(pop_pos, fitness, evaluation_now):
    # this function completes the infiltration operation
    # Args: pop_pos: population position, individual should be a row vector
    # fitness: the fitness value of population
    # evaluation_now: global control parameter
    row, col = np.shape(pop_pos)
    for index, idv in enumerate(pop_pos):
        # A randomly chosen solution is used in producing a mutant solution of the solution
        # and should not be the same
        ngh_index = np.random.randint(0, row)  # neighbour index
        while ngh_index == index:
            ngh_index = np.random.randint(0, row)
        neighbour = pop_pos[ngh_index]
        cnt_change = np.random.randint(1, col)  # the dim change count
        # randomly determine cnt_change dim to change
        dim_change = np.random.permutation(np.arange(col))
        new_idv = idv.copy()
        # diff:the difference between idv and neighbour
        diff = new_idv[dim_change[0:cnt_change]] - neighbour[dim_change[0:cnt_change]]
        # calculate new individual
        new_idv[dim_change[0:cnt_change]] += np.multiply(diff, np.random.rand(cnt_change)) * 2
        # new_fitness = objfun.train(new_idv)
        new_idv = bndry_proce(new_idv)
        new_fitness = cal_fitness(new_idv, evaluation_now)
        evaluation_now += 1
        if new_fitness > fitness[index]:
            pop_pos[index] = new_idv
            fitness[index] = new_fitness
        best_index = np.argmax(fitness)
        best_record.append(fitness[best_index])
        best_position.append(pop_pos[best_index])
    return evaluation_now


def eva_and_precip(pop_pos, fitness, best_index, evaluation_now):
    # this function completes the evaporation and precipitation operation
    # Args: pop_pos: population position, individual should be a row vector
    # fitness: the fitness value of population
    # best_index: the best individual position index
    # evaluation_now: global control parameter
    row, col = np.shape(pop_pos)
    for index in range(row):
        # evaporation and precipitation with probability p_eva
        if np.random.rand() < p_eva:
            if np.random.rand() < 0.5:
                # move to another position randomly
                pop_pos[index] = np.random.rand(col)
                pop_pos[index] = np.add(ld, np.multiply(pop_pos[index], np.subtract(ud, ld)))
            else:
                pop_pos[index] = pop_pos[best_index]
                cnt_change = np.random.randint(1, col)  # the dim change count
                # randomly determine cnt_change dim to change
                dim_change = np.random.permutation(np.arange(col))
                gaussian = np.random.randn(cnt_change)
                # calculate new individual
                pop_pos[index, dim_change[0:cnt_change]] = np.multiply(pop_pos[index,
                                                                               dim_change[0:cnt_change]], gaussian)
        pop_pos[index] = bndry_proce(pop_pos[index])
        fitness[index] = cal_fitness(pop_pos[index], evaluation_now)
        evaluation_now += 1
        best_index = np.argmax(fitness)
        best_record.append(fitness[best_index])
        best_position.append(pop_pos[best_index])

        return evaluation_now


def bndry_proce(idv):
    # this function is to process the boundary
    # idv: individual of population
    # processing method just for test
    index = np.where(idv > ud)
    idv[index[0]] = ud[index[0]]
    index = np.where(idv < ld)
    idv[index[0]] = ld[index[0]]
    idv = np.around(idv)
    return idv


def bndry_proce_pop(pop_pos):
    # this function is to process the boundary
    # idv: individual of population
    # processing method just for test
    for index, idv in enumerate(pop_pos):
        pop_pos[index] = bndry_proce(idv)


def sphere(idv):
    return np.sum(np.multiply(idv, idv))


def cal_fitness(idv, evaluation_now):
    ct.main(idv)     # train
    print(evaluation_now)
    return ce.main(idv)  # eval


if __name__ == '__main__':
    hco()
