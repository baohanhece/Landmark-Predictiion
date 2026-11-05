# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import cv2
import shutil
import numpy as np
import SimpleITK as sitk
from skimage import io as sk_io
from skimage import draw as sk_draw


def check_and_make_dir(dir_path: str) -> None:
    """
    function to create a new folder, if the folder path dir_path in does not exist
    :param dir_path: folder path | 文件夹路径
    :return: None
    """
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError('Error, the provided path (%s) is a file path, not a folder path.' % dir_path)
        shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
        os.makedirs(dir_path)
    else:
        os.makedirs(dir_path)


def load_train_stack_data(file_path: str) -> np.ndarray:
    """
    function to load train_stack.mha data file | 加载train_stack.mha数据文件的函数
    :param file_path: train_stack.mha filepath | 挑战赛提供的train_stack.mha文件路径
    :return: a 4-dim array containing 400 training set cephalometric images | 一个包含了400张训练集头影图像的四维的矩阵
    """
    sitk_stack_image = sitk.ReadImage(file_path)
    np_stack_array = sitk.GetArrayFromImage(sitk_stack_image)
    return np_stack_array


def remove_zero_padding(image_array: np.ndarray) -> np.ndarray:
    """
    function to remove zero padding in an image | 去除图像中的0填充函数
    :param image_array: one cephalometric image array, shape is (2400, 2880, 3) | 一张头影图像的矩阵，形状为(2400, 2880, 3)
    :return: image matrix after removing zero padding | 去除零填充部分的图像矩阵
    """
    row = np.sum(image_array, axis=(1, 2))
    column = np.sum(image_array, axis=(0, 2))

    non_zero_row_indices = np.argwhere(row != 0)
    non_zero_column_indices = np.argwhere(column != 0)

    last_row = int(non_zero_row_indices[-1])
    last_column = int(non_zero_column_indices[-1])

    image_array = image_array[:last_row+1, :last_column+1, :]
    return image_array

from skimage import draw as sk_draw
import numpy as np


DIGITS = {
    '0': ["111", "101", "101", "101", "111"],
    '1': ["010", "110", "010", "010", "111"],
    '2': ["111", "001", "111", "100", "111"],
    '3': ["111", "001", "111", "001", "111"],
    '4': ["101", "101", "111", "001", "001"],
    '5': ["111", "100", "111", "001", "111"],
    '6': ["111", "100", "111", "101", "111"],
    '7': ["111", "001", "010", "010", "010"],
    '8': ["111", "101", "111", "101", "111"],
    '9': ["111", "101", "111", "001", "111"]
}

tlandmarks_19pts_gt = (
    'S',                    # S1
    'N',                   # S2
    'Or',                 # S3
    'Po',                   # S4
    'A',               # S5
    'B',             # S6
    'Pog',                 # S7
    'Me',                   # S8
    'Go',                   # S10
    'L1c',   # S11
    'U1c',   # S12
    'PNS',    # S17
    'ANS',     # S18

)

def draw_number(image, number, x, y, color, scale=8):
    """
    Draw a number near (x, y) using a bitmap font.
    x, y are image coordinates.
    """
    height, width = image.shape[:2]
    cursor_x = int(x)
    start_y = int(y)

    for digit in str(number):
        pattern = DIGITS[digit]

        for row, line in enumerate(pattern):
            for col, pixel in enumerate(line):
                if pixel == '1':
                    center_y = start_y + row * scale
                    center_x = cursor_x + col * scale

                    rr, cc = sk_draw.disk(
                        (center_y, center_x),
                        radius=max(1, scale // 2),
                        shape=(height, width)
                    )

                    image[rr, cc, :3] = color

        cursor_x += 4 * scale

'''
def calculate_prediction_metrics(result_dict: dict):
    
    n_landmarks = 0
    sdr_landmarks = 0
    sdr_landmarks_25 = 0
    sdr_landmarks_40 = 0
    n_landmarks_error = 0
    for file_path, landmark_dict in result_dict.items():
        scale = 0.1#landmark_dict['scale']
        #print(scale)
        landmarks, predict_landmarks = landmark_dict['gt'], landmark_dict['predict']

        # landmarks number
        n_landmarks = n_landmarks + np.shape(landmarks)[0]

        # mean radius error (MRE)
        each_landmark_error = np.sqrt(np.sum(np.square(landmarks - predict_landmarks), axis=1)) * scale
        n_landmarks_error = n_landmarks_error + np.sum(each_landmark_error)

        # 2mm success detection rate (SDR)
        sdr_landmarks = sdr_landmarks + np.sum(each_landmark_error < 2)

        # 2mm success detection rate (SDR)
        sdr_landmarks_25 = sdr_landmarks_25 + np.sum(each_landmark_error < 2.5)


        # 2mm success detection rate (SDR)
        sdr_landmarks_40 = sdr_landmarks_40 + np.sum(each_landmark_error < 4.0)


    mean_radius_error = n_landmarks_error / n_landmarks
    sdr = sdr_landmarks / n_landmarks*100
    sdr_25 = sdr_landmarks_25 / n_landmarks*100
    sdr_40 = sdr_landmarks_40 / n_landmarks*100

    print('Mean Radius Error (MRE): {}, 2mm Success Detection Rate (SDR): {}'.format(mean_radius_error, sdr))
    print(sdr_25,sdr_40)
    return mean_radius_error, sdr
'''

def visualize_prediction_landmarks(result_dict: dict, save_image_dir: str):
    
    for file_path, landmark_dict in result_dict.items():
        landmarks, predict_landmarks, predict_Unets = landmark_dict['gt'], landmark_dict['predict_our'], landmark_dict['predict_Unet']

        image = sk_io.imread(file_path)
        image_shape = np.shape(image)[:2]

        for i in range(np.shape(predict_landmarks)[0]):
            landmark, predict_landmark, predict_Unet = landmarks[i, :], predict_landmarks[i, :], predict_Unets[i, :]
            # ground truth landmark
            radius = 7
            rr, cc = sk_draw.disk(center=(int(landmark[1]), int(landmark[0])), radius=radius, shape=image_shape)
            image[rr, cc, :] = [255, 0, 0]

            if i < len(tlandmarks_19pts_gt):

                cv2.putText(
                  image,
                  tlandmarks_19pts_gt[i],
                  (int(landmark[0]) + 10, int(landmark[1]) + 20),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.8,
                  (255, 0, 0),  # red text in RGB image
                  2,
                  cv2.LINE_AA
                )	

            # UNET model prediction landmark
            radius = 7
            rr, cc = sk_draw.disk(center=(int(predict_Unet[1]), int(predict_Unet[0])), radius=radius, shape=image_shape)
            image[rr, cc, :] = [0, 0, 255]

            
            # OUR model prediction landmark
            rr, cc = sk_draw.disk(center=(int(predict_landmark[1]), int(predict_landmark[0])), radius=radius, shape=image_shape)
            image[rr, cc, :] = [0, 255, 0]
            '''
            # the line between gt landmark and prediction landmark
            line_width = 5
            rr, cc, value = sk_draw.line_aa(int(landmark[1]), int(landmark[0]), int(predict_landmark[1]), int(predict_landmark[0]))
            for offset in range(line_width):
                offset_rr, offset_cc = np.clip(rr + offset, 0, image_shape[0] - 1), np.clip(cc + offset, 0, image_shape[1] - 1)
                image[offset_rr, offset_cc, :] = [255, 255, 0]
            '''            
        
        filename = os.path.basename(file_path)
        sk_io.imsave(os.path.join(save_image_dir, filename), image)

'''
def visualize_prediction_landmarks(result_dict, save_image_dir):
    import os
    import numpy as np
    from skimage import io as sk_io
    from skimage import draw as sk_draw

    os.makedirs(save_image_dir, exist_ok=True)

    for file_path, landmark_dict in result_dict.items():

        # Define both variables before using them
        landmarks = np.asarray(landmark_dict['gt'])
        predict_landmarks = np.asarray(landmark_dict['predict'])

        image = sk_io.imread(file_path)

        # Ensure the image has 3 color channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Convert to uint8 if the image is floating-point
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = image.copy()
        image_shape = image.shape[:2]


        # Draw all 38 predicted landmarks in GREEN,
        # with numbers 1–38 next to each point.
        for i, point in enumerate(landmarks, start=1):
            x, y = np.round(point).astype(int)

            # Draw landmark point
            rr, cc = sk_draw.disk(
                (y, x),
                radius=5,
                shape=image_shape
            )
            image[rr, cc, :3] = [0, 255, 0]

            # Draw its number slightly below and to the right
            #if i - 1 ==5 or i- 1 == 11:
            draw_number(
                image,
                i-1,
                x + 8,
                y + 8,
                color=[0, 0, 255],
                scale=8
            )

        for i, point in enumerate(predict_landmarks, start=1):
            x, y = np.round(point).astype(int)

            # Draw landmark point
            rr, cc = sk_draw.disk(
                (y, x),
                radius=5,
                shape=image_shape
            )
            image[rr, cc, :3] = [0, 255, 0]
            
            # Draw its number slightly below and to the right
            #if i - 1 >=11 and i- 1 <= 12:
            draw_number(
                image,
                i-1,
                x - 8,
                y - 8,
                color=[0, 255, 255],
                scale=8
            )



        print(
            'Image:', file_path,
            '| GT points:', len(landmarks),
            '| Predicted points:', len(predict_landmarks)
        )

        # Draw all ground-truth points in RED
        for i, point in enumerate(landmarks, start=1):
            x, y = np.round(point).astype(int)

            rr, cc = sk_draw.disk(
                (y, x), radius=7, shape=image_shape
            )
            image[rr, cc, :3] = [255, 0, 0]

        # Draw all 38 predicted points in GREEN
        for i, point in enumerate(predict_landmarks, start=1):
            x, y = np.round(point).astype(int)

            rr, cc = sk_draw.disk(
                (y, x), radius=5, shape=image_shape
            )
            image[rr, cc, :3] = [0, 255, 0]

        filename = os.path.basename(file_path)
        save_path = os.path.join(save_image_dir, filename)

        sk_io.imsave(save_path, image, check_contrast=False)
        #print('Saved:', save_path)
'''


def calculate_prediction_metrics(result_dict: dict):
    """
    Calculate overall and per-landmark MRE and SDR.

    Returns:
        overall_metrics: overall MRE and SDR
        landmark_metrics: metrics for each landmark point
    """
    '''

    all_errors = []

    for file_path, landmark_dict in result_dict.items():
        scale = 0.1  # Change to landmark_dict['scale'] if appropriate

        gt = np.asarray(landmark_dict['gt'], dtype=np.float32)
        pred = np.asarray(landmark_dict['predict_our'], dtype=np.float32)

        if gt.shape != pred.shape:
            raise ValueError(
                "{}: GT shape {} != prediction shape {}".format(
                    file_path, gt.shape, pred.shape
                )
            )

        # Error for each landmark, in mm
        errors = np.sqrt(np.sum((gt - pred) ** 2, axis=1)) * scale
        all_errors.append(errors)

    if not all_errors:
        raise ValueError("result_dict is empty")

    # Shape: (number of images, number of landmarks)
    all_errors = np.asarray(all_errors)

    # Overall metrics
    overall_mre = np.mean(all_errors)
    overall_sdr_2 = np.mean(all_errors < 2.0) * 100
    overall_sdr_25 = np.mean(all_errors < 2.5) * 100
    overall_sdr_30 = np.mean(all_errors < 3.0) * 100
    overall_sdr_40 = np.mean(all_errors < 4.0) * 100

    print("\n========== Overall Metrics ==========")
    print("Overall MRE: {:.4f} mm".format(overall_mre))
    print("Overall SDR < 2 mm: {:.2f}%".format(overall_sdr_2))
    print("Overall SDR < 2.5 mm: {:.2f}%".format(overall_sdr_25))
    print("Overall SDR < 3.0 mm: {:.2f}%".format(overall_sdr_30))
    print("Overall SDR < 4 mm: {:.2f}%".format(overall_sdr_40))

    # Per-landmark metrics
    num_landmarks = all_errors.shape[1]
    landmark_metrics = {}

    print("\n========== Per-Landmark Metrics ==========")
    print("{:<12} {:>12} {:>14} {:>14} {:>14}".format(
        "Landmark", "MRE (mm)", "SDR < 2 mm",
        "SDR < 2.5 mm", "SDR < 4 mm"
    ))

    for i in range(num_landmarks):
        errors_i = all_errors[:, i]

        mre_i = np.mean(errors_i)
        sdr_2_i = np.mean(errors_i < 2.0) * 100
        sdr_25_i = np.mean(errors_i < 2.5) * 100
        sdr_30_i = np.mean(errors_i < 3.0) * 100
        sdr_40_i = np.mean(errors_i < 4.0) * 100

        landmark_metrics[i + 1] = {
            'MRE': mre_i,
            'SDR_2mm': sdr_2_i,
            'SDR_2_5mm': sdr_25_i,
            'SDR_3_0mm': sdr_30_i,
            'SDR_4mm': sdr_40_i
        }

        print("{:<12} {:>12.4f} {:>13.2f}% {:>13.2f}% {:>13.2f}%".format(
            "Point {}".format(i + 1),
            mre_i,
            sdr_2_i,
            sdr_25_i,
            sdr_30_i,
            sdr_40_i
        ))

    overall_metrics = {
        'MRE': overall_mre,
        'SDR_2mm': overall_sdr_2,
        'SDR_2_5mm': overall_sdr_25,
        'SDR_3_0mm': overall_sdr_30,
        'SDR_4mm': overall_sdr_40
    }
    '''
    all_errors = []

    for file_path, landmark_dict in result_dict.items():
        scale = 0.1  # Change to landmark_dict['scale'] if appropriate

        gt = np.asarray(landmark_dict['gt'], dtype=np.float32)
        pred = np.asarray(landmark_dict['predict_Unet_19full'], dtype=np.float32)

        if gt.shape != pred.shape:
            raise ValueError(
                "{}: GT shape {} != prediction shape {}".format(
                    file_path, gt.shape, pred.shape
                )
            )

        # Error for each landmark, in mm
        errors = np.sqrt(np.sum((gt - pred) ** 2, axis=1)) * scale
        all_errors.append(errors)

    if not all_errors:
        raise ValueError("result_dict is empty")

    # Shape: (number of images, number of landmarks)
    all_errors = np.asarray(all_errors)

    # Overall metrics
    overall_mre = np.mean(all_errors)
    overall_sdr_2 = np.mean(all_errors < 2.0) * 100
    overall_sdr_25 = np.mean(all_errors < 2.5) * 100
    overall_sdr_30 = np.mean(all_errors < 3.0) * 100
    overall_sdr_40 = np.mean(all_errors < 4.0) * 100

    print("\n========== Overall Metrics ==========")
    print("Overall MRE: {:.4f} mm".format(overall_mre))
    print("Overall SDR < 2 mm: {:.2f}%".format(overall_sdr_2))
    print("Overall SDR < 2.5 mm: {:.2f}%".format(overall_sdr_25))
    print("Overall SDR < 3.0 mm: {:.2f}%".format(overall_sdr_30))
    print("Overall SDR < 4 mm: {:.2f}%".format(overall_sdr_40))

    # Per-landmark metrics
    num_landmarks = all_errors.shape[1]
    landmark_metrics = {}

    print("\n========== Per-Landmark Metrics ==========")
    print("{:<12} {:>12} {:>14} {:>14} {:>14}".format(
        "Landmark", "MRE (mm)", "SDR < 2 mm",
        "SDR < 2.5 mm", "SDR < 4 mm"
    ))

    for i in range(num_landmarks):
        errors_i = all_errors[:, i]

        mre_i = np.mean(errors_i)
        sdr_2_i = np.mean(errors_i < 2.0) * 100
        sdr_25_i = np.mean(errors_i < 2.5) * 100
        sdr_30_i = np.mean(errors_i < 3.0) * 100
        sdr_40_i = np.mean(errors_i < 4.0) * 100

        landmark_metrics[i + 1] = {
            'MRE': mre_i,
            'SDR_2mm': sdr_2_i,
            'SDR_2_5mm': sdr_25_i,
            'SDR_3_0mm': sdr_30_i,
            'SDR_4mm': sdr_40_i
        }

        print("{:<12} {:>12.4f} {:>13.2f}% {:>13.2f}% {:>13.2f}%".format(
            "Point {}".format(i + 1),
            mre_i,
            sdr_2_i,
            sdr_25_i,
            sdr_30_i,
            sdr_40_i
        ))

    overall_metrics = {
        'MRE': overall_mre,
        'SDR_2mm': overall_sdr_2,
        'SDR_2_5mm': overall_sdr_25,
        'SDR_3_0mm': overall_sdr_30,
        'SDR_4mm': overall_sdr_40
    }


    return overall_metrics, landmark_metrics




