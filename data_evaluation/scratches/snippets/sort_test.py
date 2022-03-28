import numpy as np

np.random.seed(86)

a = np.random.randint(0, 10, 4)

p0, p1, p2, p3 = a

# check this instead: https://stackoverflow.com/questions/6145364/sort-4-number-with-few-comparisons : )

c0, c1, c2, c3, c4, c5 = p0 < p1, p0 < p2, p0 < p3, p1 < p2, p1 < p3, p2 < p3
c0_min, c1_min, c2_min, c3_min = c0 and c1 and c2, c3 and c4 and not c0, not c1 and not c3 and c5, not c2 and not c4 and not c5
c0_max, c1_max, c2_max, c3_max = not c0 and not c1 and not c2, not c3 and not c4 and c0, c1 and c3 and not c5, c5 and c4 and c2
print(a)
print(c0, c1, c2, c3, c4, c5)

sl = [0, 0, 0, 0]

# set first spot
if c0_min:
    sl[0] = p0
elif c1_min:
    sl[0] = p1
elif c2_min:
    sl[0] = p2
else:  # c3_min:
    sl[0] = p3

# set last spot
if c0_max:
    sl[3] = p0
elif c1_max:
    sl[3] = p1
elif c2_max:
    sl[3] = p2
else:  # c3_max
    sl[3] = p3

if (c0_min or c0_max) and (c1_min or c1_max):  # p0 and p1 are already placed
    if c5:
        sl[1] = p2
        sl[2] = p3
    else:
        sl[1] = p3
        sl[2] = p2
elif (c0_min or c0_max) and (c2_min or c2_max):  # p0 and p2 are already placed
    if c4:
        sl[1] = p1
        sl[2] = p3
    else:
        sl[1] = p3
        sl[2] = p1
elif (c0_min or c0_max) and (c3_min or c3_max):  # p0 and p3 are already placed
    if c3:
        sl[1] = p1
        sl[2] = p2
    else:
        sl[1] = p2
        sl[2] = p1
elif (c1_min or c1_max) and (c2_min or c2_max):  # p1 and p2 are already placed
    if c2:
        sl[1] = p0
        sl[2] = p3
    else:
        sl[1] = p3
        sl[2] = p0
elif (c1_min or c1_max) and (c3_min or c3_max):  # p1 and p3 are already placed
    if c1:
        sl[1] = p0
        sl[2] = p2
    else:
        sl[1] = p2
        sl[2] = p0
else:  # (c2_min or c2_max) and (c3_min or c3_max):  # p2 and p3 are already placed
    if c0:
        sl[1] = p0
        sl[2] = p1
    else:
        sl[1] = p1
        sl[2] = p0

print(sl)
