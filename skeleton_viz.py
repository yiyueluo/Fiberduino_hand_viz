import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import math
import sys
import glob
import serial

DEBUG_PRINT = 0

data_backup = [-1, 1,1,1, 1,1,1]

################################# functions #######################################
def serial_init():
    global DEBUG_PRINT
    devices = glob.glob("/dev/ttyACM*")
    if DEBUG_PRINT:
        print(devices)
    ser = serial.Serial(devices[0], 115200)
    success = ser.isOpen()
    if not success:
        print("\n!!! Error: serial device not found !!!")
        sys.exit(-1)
    return ser

def get_data(ser):
    global DEBUG_PRINT
    global data_backup
    received = ser.readline()
    ser.flush()
    data = received.split()

    if DEBUG_PRINT:
        print("\ndata =", data)

    if len(data) != 7:
        return data_backup
    else:
        data_backup = data
        return data


#         2-1-0-\
#                   \
# *   5-- 4 -- 3 -----x   < NOTE: demo on this finger
#   8 -- 7 -- 6 ----/
#   11 - 10 - 9 --/
#  14-- 13 -- 12 --/

#################################### main #########################################

df_hand = pd.read_csv('./skeleton.csv', sep=',', header=0)
keypoint = df_hand.to_numpy()

hand = np.copy(keypoint)

df_linkage = pd.read_csv('./linkage.csv', sep=',', header=0)
linkage = df_linkage.to_numpy()

linkage_length = []
for i in range(linkage.shape[0]):
    l = np.sqrt((hand[linkage[i, 1], 1] - hand[linkage[i, 2], 1]) ** 2 +
                (hand[linkage[i, 1], 2] - hand[linkage[i, 2], 2]) ** 2 +
                (hand[linkage[i, 1], 3] - hand[linkage[i, 2], 3]) ** 2)
    linkage_length.append(l)

print (hand.shape, linkage.shape)
plt.ion()
fig = plt.figure()

ser = serial_init()

while True:
    ''' grab data '''
    data = get_data(ser)
    # touch sensing (binary):
    print("touch:", data[0])

    # magnetometer (only needs y and x for now):
    heading = math.atan2(float(data[2]), float(data[1]))  # (y, x)
    heading = int(math.degrees(heading))
    print("heading:", heading, "\t", int((180 + heading) / 5) * "*")

    # accelerometer (only needs x and z for now):
    finger = math.atan2(float(data[4]), float(data[6]))
    finger = int(math.degrees(finger))
    print("finger:", finger, "\t", int((180 + finger) / 5) * "*")

    ''' lateral angle '''
    # for i in np.arange(0, pi/2, pi/100):
    #     # lateral_angle = pi/6
    #     lateral_angle = i
    #
    #     hand[6, 1] = keypoint[6, 1] + linkage_length[3] * math.sin(lateral_angle)
    #     hand[6, 2] = keypoint[6, 2] + linkage_length[3] * (1- math.cos(lateral_angle))
    #
    #     hand[7, 1] = keypoint[7, 1] +  (linkage_length[4] + linkage_length[3]) * math.sin(lateral_angle)
    #     hand[7, 2] = keypoint[7, 2] +  (linkage_length[4] + linkage_length[3]) * (1- math.cos(lateral_angle))
    #
    #     hand[8, 1] = keypoint[8, 1] +  (linkage_length[4] + linkage_length[3] + linkage_length[5]) * math.sin(lateral_angle)
    #     hand[8, 2] = keypoint[8, 2] + (linkage_length[4] + linkage_length[3] + linkage_length[5]) * (1- math.cos(lateral_angle))

    ''' bending '''
    for i in np.arange(0, pi / 2, pi / 100):
        # lateral_angle = pi/6
        lateral_angle = i

        hand[6, 3] = keypoint[6, 3] + linkage_length[3] * math.sin(lateral_angle)
        hand[6, 2] = keypoint[6, 2] +  linkage_length[3] * (1 - math.cos(lateral_angle))

        hand[7, 3] = keypoint[7, 3] + (linkage_length[4] + linkage_length[3]) * math.sin(lateral_angle + pi/12)
        hand[7, 2] = keypoint[7, 2] + (linkage_length[4] + linkage_length[3]) * (1 - math.cos(lateral_angle + pi/12))

        hand[8, 3] = keypoint[8, 3] + (linkage_length[4] + linkage_length[3] + linkage_length[5]) * math.sin(lateral_angle + pi/8)
        hand[8, 2] = keypoint[8, 2] + (linkage_length[4] + linkage_length[3] + linkage_length[5]) * (1 - math.cos(lateral_angle + pi/8))


        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(120,0)

        ax.set_xlim(-0, 200)
        ax.set_ylim(-0, 200)
        ax.set_zlim(-100, 100)

        # finger_angle0 = [0, 0, pi/6]
        # finger_angle2 = [0, 0, pi/6]

        ax.scatter(hand[:, 1], hand[:, 2], hand[:, 3], s=50, c='r', zorder=10)

        for i in range(linkage.shape[0]):
            st_index = linkage[i, 1]
            ed_index = linkage[i, 2]
            xs_line = [hand[st_index, 1], hand[ed_index, 1]]
            ys_line = [hand[st_index, 2], hand[ed_index, 2]]
            zs_line = [hand[st_index, 3], hand[ed_index, 3]]
            ax.plot(xs_line, ys_line, zs_line, color='k', linewidth=5, zorder=10)


        plt.draw()
        plt.pause(0.01)
        plt.clf()

        # plt.show(block=False)
        # plt.pause(0.1)
        # # plt.clf()
        # plt.close()