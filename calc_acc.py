import sys

from lab import *

actual_data = None
calc_data = None

def transfrom(text, skip, i):
    result = []
    temp = text.split('\n')
    j_1 = 0
    j_2 = -1
    for row in temp:
        j_1 += 1
        if j_1 <= skip:
            continue
        j_2 += 1
        if (j_2)%i != 0:
            continue
        j_2 = 0
        t_r = []
        for col in row.split(';'):
            v = eval(col)
            t_r.append(None if v == -1 or v == -2 else v)
        result.append(t_r)
    return result

with open(sys.argv[1], 'r') as temp:
    actual_data = transfrom(temp.read(),101,25)

with open(sys.argv[2], 'r') as temp:
    calc_data = transfrom(temp.read(),101,25)

print(len(calc_data),len(actual_data),20)

iterations_count = min(len(calc_data),len(actual_data),20)

[actual_data, calc_data] = [actual_data[0:iterations_count],calc_data[0:iterations_count]]

people_count = max(len(calc_data[0]),len(actual_data[0]))
people = []
max_error = 2
max_distance = 2

for i in range(people_count):
    people.append([None, 1])

agg_sum = 0
agg_count = 0

class MDPoint(DPoint):
    def __init__(self, pos_x, pos_y, color, s_i, id):        
        # pos_x & pos_y in the top view
        # s_i stands for stream index
        super().__init__(pos_x, pos_y, color, s_i)
        self.id = id

for i in range(len(actual_data)):
    
    d_points = []
    
    j = 0
    for c in actual_data[i]:
        if c is not None:
            d_points.append(MDPoint(c[0], c[1], (0,0,0), 0, j))
        j += 1

    j = 0
    for c in calc_data[i]:
        if c is not None:
            d_points.append(MDPoint(c[0], c[1], (0,0,0), 1, j))
        j += 1

    clusters = KMeans.predict(d_points, max_error)
    
    for cluster in clusters:
        try:
            [first_d_point, second_d_point] = cluster.d_points
            if first_d_point.s_i != 0:
                [first_d_point, second_d_point] = [second_d_point, first_d_point]
            if people[first_d_point.id][0] is None:
                # print('1')
                people[first_d_point.id] = [second_d_point.id, 1]
            elif people[first_d_point.id][0] != second_d_point.id:
                # print('2')
                people[first_d_point.id] = [second_d_point.id, people[first_d_point.id][1]*2]
            factor = 1
            error = (Point.calcDist(first_d_point, second_d_point) * factor / max_distance)
            # error = max(0.2, error)
            error = min(1, error)
            print(people)
            agg_sum +=  error
        except :
            agg_sum += 1
        agg_count += 1


print( agg_sum / agg_count if agg_count != 0 else 0 )