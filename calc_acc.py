import math
import re

# point has the form (posX, posY)
def get_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2+(point1[1] - point2[1])**2)

def find_nearst_point(point, list_points):
    target = None
    error = float("inf")
    for cur_point in list_points:
        distance = get_distance(point[0], cur_point[0])
        if distance < error:
            error = distance
            target = cur_point
    return (target,error)

def calc_error(t_ps, d_ps, max_distance):
    t_ps = t_ps.copy()
    d_ps = d_ps.copy()
    acc_error = 0
    errors_count = 0
    while(len(t_ps) != 0):
        t_p = t_ps[0]
        (match_d_p, cur_error) = find_nearst_point(t_p,d_ps)
        if match_d_p is None:
            errors_count += 1
            acc_error += 1
            t_ps.remove(t_p)
        else:
            (match_t_p, _) = find_nearst_point(match_d_p,t_ps)
            if match_t_p != t_p:
                t_ps.remove(t_p)
                t_ps.append(t_p)
                continue
            errors_count += 1
            acc_error += cur_error / max_distance
            d_ps.remove(match_d_p)
            t_ps.remove(t_p)
    errors_count += len(d_ps)
    acc_error += len(d_ps)
    if acc_error == 0:
        return acc_error        
    else:
        return acc_error / errors_count


opt_ground_truth_file = open('output/opt_ground_truth_list.txt',"r")
cal_ground_truth_file = open('output/cal_ground_truth_list.txt',"r")

opt_ground_truth_list = opt_ground_truth_file.read().split('\n')
cal_ground_truth_list = cal_ground_truth_file.read().split('\n')

num_iterations = len(cal_ground_truth_list)

def get_list(line):
    res = []
    arr = re.findall('([0-9]+, [0-9]+)',re.split(r'[0-9]+ +', line)[1])
    for p in arr:
        res.append(eval('('+p+')'))
    return res

# t_ps points from ground truth file [points], point => [(posX,posY), ...]
t_ps = list(map(get_list, opt_ground_truth_list[0:num_iterations]))
# d_ps points detected by your program [points], point => [(posX,posY), ...]
d_ps = list(map(get_list, cal_ground_truth_list[0:num_iterations]))
# max_distance should equal diagonal length of frame to ensure 100% or lower error
max_distance = 400

f= open('output/accuracy.txt',"w+")
f.write(str(1 - calc_error(t_ps, d_ps, max_distance)))
f.close()

opt_ground_truth_file.close()
cal_ground_truth_file.close()