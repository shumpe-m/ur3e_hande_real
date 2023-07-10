import cv2
import numpy as np


def get_area_of_interest_new(
   image=None,
   pose=None,
   size_cropped = None,
   size_result = None,
   border_color=None,
   project_3d=True,
   flags=cv2.INTER_LINEAR,
   size_memory_scale=None,
):
    size_input = (image.shape[1], image.shape[0])
    center_image = (size_input[0] / 2, size_input[1] / 2)
    angle_a = 0 if pose[2] == None else pose[2]
   #  if abs(action_pose.b) > np.pi - 0.1 and abs(action_pose.c) > np.pi - 0.1:
   #      angle_a = action_pose.a - np.pi

    if size_result and size_cropped:
        scale = size_result[0] / size_cropped[0]
        assert scale == (size_result[1] / size_cropped[1])
    elif size_result:
        scale = size_result[0] / size_input[0]
        assert scale == (size_result[1] / size_input[1])
    else:
        scale = 1.0

    size_final = size_result or size_cropped or size_input

    trans = get_transformation(
        x=pose[0] / size_memory_scale,
        y=pose[1] /size_memory_scale,
        a=-angle_a,
        center=center_image,
        scale=scale,
        cropped=size_cropped,
    )
    mat_result = cv2.warpAffine(image, trans, size_final, flags=flags)  # INTERPOLATION_METHOD

    return mat_result

def get_transformation(x, y, a, center, scale = 1.0, cropped = None):
   rot_mat = cv2.getRotationMatrix2D((round(x), round(y)), a * 180.0 / np.pi, scale)  # [deg]
   rot_mat[0][2] += round(center[0] - x) + scale * cropped[0] / 2 - center[0] if cropped else round(center[0] - x)
   rot_mat[1][2] += round(center[1] - y) + scale * cropped[1] / 2 - center[1] if cropped else round(center[1] - y)
   return rot_mat