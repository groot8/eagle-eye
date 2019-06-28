import re
import sys

def grid_to_tv(pos, grid_width, grid_height, tv_origin_x, tv_origin_y, tv_width, tv_height):     
    tv_x = ( (pos % grid_width) + 0.5 ) * (tv_width / grid_width) + tv_origin_x
    tv_y = ( (pos / grid_width) + 0.5 ) * (tv_height / grid_height) + tv_origin_y
    return (tv_x, tv_y)

def parse(file_name):
    arr = []
    f=open(file_name, "r")
    f1 = f.readlines()
    print(f1)
    for x in f1:
        d = " ".join(x.split())
        # print(d)
        arr.append(d)

    output_text = ""
    for i in range(len(arr)):
        if(i == 0):
            continue
        elif(i == 1):
            [no_of_frames, no_of_people, grid_width, grid_height, step_size, first_frame, last_frame] = arr[i].split()
            for j in range(1, int(no_of_people) + 1):
                output_text += str(j - 1)
                if j == int(no_of_people):
                    output_text += "\n"
                else:
                    output_text += ";"                    
        elif(i > 76 and (i - 77) % 25 == 0):
            for j in range(0, int(no_of_people)):
                if(arr[i].split()[j] == "-2" or arr[i].split()[j] == "-1"):
                    output_text += arr[i].split()[j]
                else:
                    tv_x, tv_y = grid_to_tv( int(arr[i].split()[j]) , int(grid_width), int(grid_height), 0, 0, 358, 360)
                    output_text += '['+str(tv_x)+','+str(tv_y)+']'
                if j + 1 == int(no_of_people):
                    output_text += "\n"
                else:
                    output_text += ";"   
    return output_text[0:-1]

print(parse(sys.argv[-1]))