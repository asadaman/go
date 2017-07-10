import numpy as np
import cv2
import sys
import copy
import matplotlib.pyplot as plt

def set_image_size(img):
    image_size['x'] = img.shape[0]
    image_size['y'] = img.shape[1]

def get_mouse_point(event,x,y,flags,param):
    mouse_point = [0, 0]
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_point = [x, y]
        print(mouse_point)
        return mouse_point

def set_corners_point(img):
    cv2.namedWindow('setting')
    cv2.imshow('setting', img)
    has_set_corner = [False, False, False, False]
    while (1):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            corners['left_top'] = cv2.setMouseCallback('setting', get_mouse_point)
            print(corners['left_top'])
        if key == ord('2'):
            corners['right_top'] = cv2.setMouseCallback('setting', get_mouse_point)
        if key == ord('3'):
            corners['right_buttom'] = cv2.setMouseCallback('setting', get_mouse_point)
        if key == ord('4'):
            corners['left_buttom'] = cv2.setMouseCallback('setting', get_mouse_point)
        if not corners['left_buttom'] == [0, 0]:
            break

def initial_setting(img):
    set_image_size(img)
    set_corners_point(img)


def reduce_noise(img, median_box_size, kernel_size):
    cv2.medianBlur(img, median_box_size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    output_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return output_img

def perspective_transform(img, pts_input, pts_output):
    raws, cols, ch = img.shape
    M = cv2.getPerspectiveTransform(pts_input, pts_output)
    output_img = cv2.warpPerspective(img, M, (600, 600))
    return output_img

def ret_cross_points(img, board_type):
    crossPoints = []
    interval = img.shape[0]/(board_type-1)
    for i in range(board_type):
        row = []
        for j in range(board_type):
            x = int(i*interval)
            y = int(j*interval)
            if i==0:
                x += 7
            if j==0:
                y += 7
            if i==board_type-1:
                x -= 7
            if j==board_type-1:
                y -= 7
            row.append([y, x])
        crossPoints.append(row)
    return crossPoints

def draw_cross_points(img, board_type):
    output_img = img.copy()
    cross_points = ret_cross_points(img, board_type)
    for p_row in cross_points:
        for p in p_row:
            cv2.circle(output_img, (p[0],p[1]), 2, (0, 0, 255), 3)
    return output_img


did_initial_setting = False
image_size = {'x': 0, 'y': 0}
# corners_of_board = {'left_top': [0, 0], 'right_top': [0, 0], 'right_buttom': [0, 0], 'left_buttom': [0, 0]}
# corners_of_board = {'left_top': [423, 97], 'right_top': [836, 114], 'right_buttom': [837, 482], 'left_buttom': [379, 464]}
corners_of_board = np.float32([[423, 97], [836, 114], [837, 482], [379, 464]])
corners_of_img = np.float32([[0,0], [600,0], [600,600], [0, 600]])
board_type = 9

filename = sys.argv[1]
cap = cv2.VideoCapture(filename)
while(cap.isOpened()):
    ret, frame = cap.read()

    # if not did_initial_setting:
    #     initial_setting(frame)
    #     did_initial_setting = True

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('frame', gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    board_img = perspective_transform(frame, corners_of_board, corners_of_img)
    # board_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('frame', board_img)

    board_with_points_img = draw_cross_points(board_img, board_type)
    # cv2.imshow('board with cross points', board_with_points_img)

    noise_reduced_img = reduce_noise(board_img, 11, 9)
    # cv2.imshow('noise reduced image', noise_reduced_img)

    hsv_img = cv2.cvtColor(noise_reduced_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv image', hsv_img)

    # beige = np.uint8([[[175, 175, 180]]])
    # hsv_beige = cv2.cvtColor(beige,cv2.COLOR_BGR2HSV)
    # print(hsv_beige)
    lower_beige = np.array([0,100,100])
    upper_beige = np.array([30,255,255])

    mask_beige = cv2.inRange(hsv_img, lower_beige, upper_beige)
    cv2.imshow('mask image', mask_beige)

    # stones_position = check_stones_position(noise_reduced_img)

    if cv2.waitKey(100):
        continue
    if 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
