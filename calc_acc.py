import sys

from lab import *

actual_data = None
calc_data = None

def transfrom(text):
    result = []
    temp = text.split('\n')[1:]
    for row in temp:
        t_r = []
        for col in row.split(';'):
            v = eval(col)
            t_r.append(None if v == -1 or v == -2 else v)
        result.append(t_r)
    return result

with open(sys.argv[1], 'r') as temp:
    actual_data = transfrom(temp.read())

with open(sys.argv[2], 'r') as temp:
    calc_data = transfrom(temp.read())


iterations_count = min(len(calc_data),len(actual_data))

[actual_data, calc_data] = [actual_data[0:iterations_count],calc_data[0:iterations_count]]



people_count = len(calc_data[0])
people = []
max_distance_kmean = float('inf')
max_confidence = 5 ## in seconds
diagonal_distance = 860 ## in pixels
punishment = 0.2
marginal_error_radius = 100 ## in pixels


for i in range(people_count):
    people.append([None, 0, None, 0])

agg_sum = 0
agg_count = 0

class MDPoint(DPoint):
    def __init__(self, pos_x, pos_y, color, s_i, id):        
        # pos_x & pos_y in the top view
        # s_i stands for stream index
        super().__init__(pos_x, pos_y, color, s_i)
        self.id = id

for i in range(iterations_count):
    
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

    clusters = KMeans.predict(d_points)
    
    for cluster in clusters:
        try:
            [first_d_point, second_d_point] = cluster.d_points
            if first_d_point.s_i != 0:
                [first_d_point, second_d_point] = [second_d_point, first_d_point]
            if people[first_d_point.id][0] is None:
                # print('1')
                people[first_d_point.id] = [second_d_point.id, 1, None, 0]
            elif people[first_d_point.id][0] == second_d_point.id:
                # print('2')
                confidence = min(people[first_d_point.id][1] + 1, max_confidence)
                people[first_d_point.id] = [second_d_point.id, confidence , None, 0]
            else:
                confidence = people[first_d_point.id][1] - 1
                if confidence == 0:
                    people[first_d_point.id] = [second_d_point.id, 1 , None, people[first_d_point.id][3]]
                else:
                    if people[first_d_point.id][2] == None:
                        people[first_d_point.id] = [people[first_d_point.id][0], confidence , second_d_point.id, people[first_d_point.id][3] + punishment]
                    elif people[first_d_point.id][2] == second_d_point.id:
                        people[first_d_point.id] = [people[first_d_point.id][0], confidence , second_d_point.id, people[first_d_point.id][3]]
                    else:
                        people[first_d_point.id] = [people[first_d_point.id][0], confidence , second_d_point.id, people[first_d_point.id][3]+ punishment]
            distance = Point.calcDist(first_d_point, second_d_point)
            if distance >= marginal_error_radius:
                error = (distance / diagonal_distance)
                error = max(people[first_d_point.id][3], error)
                error = min(1, error)
                # print(people)
                agg_sum +=  error
        except :
            agg_sum += 1
        agg_count += 1


error = agg_sum / agg_count if agg_count != 0 else 0

print(1 - error)