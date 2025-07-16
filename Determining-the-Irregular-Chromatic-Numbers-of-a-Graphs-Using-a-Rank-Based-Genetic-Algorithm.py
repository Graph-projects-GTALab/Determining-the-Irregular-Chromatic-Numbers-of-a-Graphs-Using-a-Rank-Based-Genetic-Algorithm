import random
import math
from collections import Counter
import networkx as nx
import time
import pandas as pd

# --- ورودی: گراف (لیست مجاورتی) ---
# مثال گراف ساده:
graph = nx.Graph()
graph = nx.cycle_graph(9)

V = list(graph.nodes())
N = len(V)  # تعداد رأس‌ها

# پارامترهای الگوریتم
POP_SIZE = 50
GENERATIONS = 10000
S = 3          # فشار انتخاب (Selection pressure)
P_MAX = 0.3  # حداکثر احتمال جهش
ALPHA = 100    # وزن تضاد رنگی در تابع شایستگی
BETA = 100        # وزن تکرار کد رنگی در تابع شایستگی
MAX_COLORS = 6     # بیشینه رنگ‌ها (حد بالا، می‌توان کمتر گرفت)
while True:
    try:
        Approximation = int(input("Approximation: "))
        break
    except ValueError:
        print("please enter natural number")

# --- تعریف تابع کد رنگی ---
def color_code(coloring, v):
    c_v = coloring[v]
    neighbor_colors = [coloring[u] for u in graph[v]]
    count = Counter(neighbor_colors)
    # مرتب کردن کلیدها برای یکتا بودن نمایش
    return (c_v, tuple(sorted(count.items())))

# --- محاسبه تابع شایستگی ---
def fitness_function(coloring):
    used_colors = set(coloring)
    k = len(used_colors)
    
    conflicts = 0
    for v in V:
        for u in graph[v]:
            if coloring[u] == coloring[v]:
                conflicts += 1
    conflicts //= 2  # هر تضاد دو بار شمرده میشود

    codes = [color_code(coloring, v) for v in V]
    duplicate_codes = len(codes) - len(set(codes))

    # تابع شایستگی منفی است چون می‌خواهیم کمینه کنیم
    fitness = -k - ALPHA * conflicts - BETA * duplicate_codes
    return fitness, conflicts, duplicate_codes, k

# --- تولید جمعیت اولیه ---
def generate_population():
    population = []
    for _ in range(POP_SIZE):
        individual = [random.randint(0, MAX_COLORS - 1) for _ in V]
        population.append(individual)
    return population

# --- رتبه بندی و نرمال سازی رتبه‌ها ---
def rank_population(population):
    scored = []
    for individual in population:
        fit, conflicts, dup, k = fitness_function(individual)
        scored.append((fit, individual))
    # مرتب سازی بر اساس شایستگی (بیشتر بهتر)
    scored.sort(key=lambda x: x[0], reverse=True)
    
    ranked_population = []
    for i, (fit, indiv) in enumerate(scored):
        r_i = i / (POP_SIZE - 1)  # نرمال سازی رتبه (0 بهترین، 1 بدترین)
        ranked_population.append((r_i, fit, indiv))
    return ranked_population

# --- انتخاب مبتنی بر رتبه ---
def rank_based_selection(ranked_population):
    clones = []
    for r_i, fit, indiv in ranked_population:
        clone_count = int(S * ((1 - r_i) ** (S - 1)))
        clones.extend([indiv.copy()] * clone_count)
    # اگر تعداد کلون‌ها کمتر یا بیشتر از POP_SIZE شد، اصلاح می‌کنیم
    while len(clones) < POP_SIZE:
        clones.append(random.choice(clones).copy())
    if len(clones) > POP_SIZE:
        clones = clones[:POP_SIZE]
    return clones

# --- ترکیب مبتنی بر رتبه ---
def rank_based_crossover(ranked_population):
    offspring = []
    # ترکیب بین زوج‌های متوالی در جمعیت رتبه‌بندی شده
    for i in range(0, POP_SIZE - 1, 2):
        _, _, parent1 = ranked_population[i]
        _, _, parent2 = ranked_population[i+1]
        point = random.randint(1, N-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        offspring.append(child1)
        offspring.append(child2)
    # اگر جمعیت فرد بود، آخرین فرد را اضافه می‌کنیم بدون تغییر
    if POP_SIZE % 2 == 1:
        offspring.append(ranked_population[-1][2].copy())
    return offspring

# --- جهش مبتنی بر رتبه ---
def rank_based_mutation(ranked_population):
    mutated = []
    for r_i, fit, indiv in ranked_population:
        if r_i == 0.0:
            p_i = 0.0
        else:
            p_i = P_MAX * (r_i ** (math.log(P_MAX * N) / math.log(POP_SIZE - 1)))
        new_indiv = indiv.copy()
        for idx in range(N):
            if random.random() < p_i:
                new_indiv[idx] = random.randint(0, MAX_COLORS - 1)
        mutated.append(new_indiv)
    return mutated


# --- اجرای الگوریتم ژنتیک رتبه بندی شده ---
def rank_ga():
    # start = time.time()
    population = generate_population()
    best_solution = None
    best_fitness = float('-inf')
    
    for gen in range(GENERATIONS):
        ranked_pop = rank_population(population)
        
        # ذخیره بهترین فرد
        if ranked_pop[0][1] > best_fitness:
            best_fitness = ranked_pop[0][1]
            best_solution = ranked_pop[0][2].copy()
            print(f"Generation {gen}: Best fitness = {best_fitness}")
            
        if best_fitness >= -1 * Approximation:
            break
        selected = rank_based_selection(ranked_pop)
        
        # برای ترکیب نیاز به مرتب بودن داریم، دوباره رتبه بندی می‌کنیم
        ranked_selected = rank_population(selected)
        
        offspring = rank_based_crossover(ranked_selected)
        
        # رتبه بندی فرزندان برای جهش
        ranked_offspring = rank_population(offspring)
        
        mutated = rank_based_mutation(ranked_offspring)
        
        population = mutated
    
    # خروجی نهایی
    fit, conflicts, dup, k = fitness_function(best_solution)
    print(f"Best solution: fitness = {fit} ,colors used = {k}, conflicts = {conflicts}, duplicate codes = {dup}")
    print(f"Coloring: {best_solution}")
    finish = time.time()
    return fit 

# --- اجرای الگوریتم ---
best_coloring = rank_ga()
