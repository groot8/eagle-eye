import re

# pos => from ground truth file
# 40, 99 => grid_width, drid_height
# 0, 38.48 => tv_origin_x, tv_origin_y
# 155, 381 => tv_width, tv_height
def grid_to_tv(pos, grid_width, grid_height, tv_origin_x, tv_origin_y, tv_width, tv_height):     
    tv_x = ( (pos % grid_width) + 0.5 ) * (tv_width / grid_width) + tv_origin_x
    tv_y = ( (pos / grid_width) + 0.5 ) * (tv_height / grid_height) + tv_origin_y
    return (tv_x, tv_y)



def parse(file_name):
    arr = []
    f=open(file_name, "r")
    f1 = f.readlines()
    for x in f1:
        d = " ".join(x.split())
        # print(d)
        arr.append(d)

    i = 0
    no_of_frames = 0
    no_of_people = 0
    grid_width = 0
    grid_height = 0
    step_size = 0
    first_frame = 0
    last_frame = 0

    f= open("ground_table.txt","w+")
    f.close()
    for el in arr:
        if(i == 0):
            i = i +1
        elif(i == 1):
            no_of_frames = el.split()[0]
            no_of_people = el.split()[1]
            grid_width = el.split()[2]
            grid_height = el.split()[3]
            step_size = el.split()[4]
            first_frame = el.split()[5]
            last_frame = el.split()[6]
            i = i +1
        else:
            if(el.split()[0] == "-1"):
                i = i +1
                continue
            else:
                for j in range(0, int(no_of_people)):
                    if(el.split()[j] == "-2"):
                        continue
                    else:
                        # print("Person number is ", j, " and person pos is ", el.split()[j])
                        tv_x, tv_y = grid_to_tv( int(el.split()[j]) , grid_width, grid_height, -500, -1500, 7500, 11000)
                        # print("In Frame number ", i-1 , " Person number ", j ," has tv_x = ", tv_x, " and tv_y =  ", tv_y )
                        print("In Frame number " +  str(i-1) + " Person number " + str(j) + " has tv_x = " + str(tv_x) + " and tv_y =  " + str(tv_y) )
                        f= open("ground_table.txt","a+")
                        f.write("frame " + str(i-1) + " => " + str(tv_x) + " , " + str(tv_y) + "\n")
                        f.close()
                i = i +1

parse("dataset/gt_terrace1.txt")