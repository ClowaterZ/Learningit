"""
    进化算法
"""
import numpy as np
import math
import random

# 初始化变量阈值
min_x = 0.0
max_x = 1.0
num_classes = 10

def fast_non_dominated_sort(acc_list):
    S = [[] for i in range(0, len(acc_list[0]))]
    front = [[]]
    n = [0 for i in range(0, len(acc_list[0]))]
    rank = [0 for i in range(0, len(acc_list[0]))]

    for p in range(0, len(acc_list[0])):
        S[p] = []
        n[p] = 0
        for q in range(0, len(acc_list[0])):
            if is_dominated(p, q, acc_list):  # 如果p支配q
                if q not in S[p]:
                    S[p].append(q)
            elif is_dominated(q, p, acc_list):  # 如果p被q支配
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    # while front[i] != []:
    while len(front[i]) != 0:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


# 判断p是否支配q
def is_dominated(p, q, acc_list):
    judge = True
    for i in range(len(acc_list)):
        if acc_list[i][p] < acc_list[i][q]:
            judge = False
            break
    if judge:
        judge = False
        for i in range(len(acc_list)):
            if acc_list[i][p] > acc_list[i][q]:
                judge = True
                break
    return judge


def index_of(a, value_list):
    for i in range(0, len(value_list)):
        if value_list[i] == a:
            return i
    return -1


def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list) != len(list1):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def crowding_distance(acc_list, front):
    distance = [0 for i in range(0, len(front))]
    sorted_list = []
    for i in range(len(acc_list)):
        sorted_list.append(sort_by_values(front, acc_list[i][:]))
    distance[0] = 100000
    distance[len(front) - 1] = 100000
    for i in range(len(acc_list)):
        for k in range(1, len(front) - 1):
            distance[k] = distance[k] + (acc_list[i][sorted_list[i][k + 1]] - acc_list[i][sorted_list[i][k - 1]]) / (
                    max(acc_list[i]) - min(acc_list[i]))
    return distance


def crossover(solutions):
    for i in range(3):
        # 随机交换某一个基因
        k = np.random.randint(0, len(solutions[0]))
        r = random.random()
        temp1 = solutions[i][k] * r + solutions[i + 3][k] * (1 - r)
        temp2 = solutions[i][k] * (1 - r) + solutions[i + 3][k] * r
        solutions[i][k] = temp1
        solutions[i + 3][k] = temp2

    # 以0.01的概率确定是否变异
    for i in range(len(solutions)):
        r = random.random()
        if r < 0.01:
            solutions[i] = mutation(solutions[i])

    return solutions


def mutation(solution):
    k = np.random.randint(0, len(solution))
    for label in range(num_classes):
        solution[k][label] = min_x + (max_x - min_x) * random.random()
    return solution


def evolution(solutions, acc_list):
    # pt是全体种群   ct是选择的父代对应的编号
    pt = np.array(solutions)
    ct = []

    # 快速非支配排序，front[i]中存储的是第i个非支配层中对应分配方案的标签
    front = fast_non_dominated_sort(acc_list)

    # 个体拥挤度距离算法，用于对同一层帕累托解进行排序
    distances = []
    for i in range(len(front)):
        distances.append(crowding_distance(acc_list, front[i]))

    # 选择父代，优先选择拥挤距离大的个体，保持种群多样性 10个
    for i in range(len(front)):
        for j in range(len(front[i])):
            index = distances[i].index(max(distances[i]))
            ct.append(front[i][index])
            distances[i][j] = -100000
    ct = ct[:6]
    best = ct[0]

    # 交叉，变异产生10个子代
    np.random.shuffle(ct)
    parents = pt[ct]
    children = crossover(parents)

    # 精英策略选择算子，父代中优良个体也进入子代
    population = np.append(parents, children)

    return population.tolist, best
