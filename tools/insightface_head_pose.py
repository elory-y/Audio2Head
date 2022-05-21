from insightface.app import FaceAnalysis
import numpy as np
import cv2
from PIL import Image


det_size = 640
ctx_id = -1
app = FaceAnalysis(allowed_modules=["detection","landmark_2d_106"], providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

class HeadPose(object):
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ], dtype="float32")

    def get_face_landmark(self, image):
        face = app.get(image)
        return face[0].landmark_2d_106

    def get_image_points_from_landmark(self,face_landmark):
        image_points = np.array([
            face_landmark[86],  # Nose tip
            face_landmark[0],  # Chin
            face_landmark[93],  # Left eye left corner
            face_landmark[35],  # Right eye right corne
            face_landmark[61],  # Left Mouth corner
            face_landmark[52]  # Right mouth corner
        ], dtype="float32")
        return image_points
    def caculate_pose_vector(self, image):
        face_2d_landmark = self.get_face_landmark(image)
        ldk_min = np.min(face_2d_landmark, axis=0)
        ldk_max = np.max(face_2d_landmark, axis=0)
        ldk_w_h = ldk_max - ldk_min  # w, h
        face_2d = face_2d_landmark - ldk_min
        image_points = self.get_image_points_from_landmark(face_2d)
        camera_matrix = np.array(
            [[ldk_w_h[0], 0, ldk_w_h[0] / 2],
             [0, ldk_w_h[0], ldk_w_h[1] / 2],
             [0, 0, 1]], dtype="float32")
        dist_coeffs = np.zeros((4, 1), dtype="float32")  # Assuming no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix,
                                                                      dist_coeffs)
        return rotation_vector.T, translation_vector.T

#
# if __name__ == "__main__":
#     def draw_annotation_box(image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
#         """Draw a 3D box as annotation of pose"""
#
#         camera_matrix = np.array(
#             [[233.333, 0, 128],
#              [0, 233.333, 128],
#              [0, 0, 1]], dtype="double")
#
#         dist_coeefs = np.zeros((4, 1))
#
#         point_3d = []
#         rear_size = 75
#         rear_depth = 0
#         point_3d.append((-rear_size, -rear_size, rear_depth))
#         point_3d.append((-rear_size, rear_size, rear_depth))
#         point_3d.append((rear_size, rear_size, rear_depth))
#         point_3d.append((rear_size, -rear_size, rear_depth))
#         point_3d.append((-rear_size, -rear_size, rear_depth))
#
#         front_size = 100
#         front_depth = 100
#         point_3d.append((-front_size, -front_size, front_depth))
#         point_3d.append((-front_size, front_size, front_depth))
#         point_3d.append((front_size, front_size, front_depth))
#         point_3d.append((front_size, -front_size, front_depth))
#         point_3d.append((-front_size, -front_size, front_depth))
#         point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
#
#         # Map to 2d image points
#         (point_2d, _) = cv2.projectPoints(point_3d,
#                                           rotation_vector,
#                                           translation_vector,
#                                           camera_matrix,
#                                           dist_coeefs)
#         point_2d = np.int32(point_2d.reshape(-1, 2))
#
#         # Draw all the lines
#         cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
#         cv2.line(image, tuple(point_2d[1]), tuple(
#             point_2d[6]), color, line_width, cv2.LINE_AA)
#         cv2.line(image, tuple(point_2d[2]), tuple(
#             point_2d[7]), color, line_width, cv2.LINE_AA)
#         cv2.line(image, tuple(point_2d[3]), tuple(
#             point_2d[8]), color, line_width, cv2.LINE_AA)
#     img_path = 'demo/img/baiden.jpg'
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     videoWriter = cv2.VideoWriter('xw-1.mp4', fourcc, 25, (256, 256))
#     # img = cv2.imread(img_path)
#     headpose = HeadPose()
#     capture = cv2.VideoCapture("/home/caopu/workspace/Audio2Head/data/2.mp4")
#
#     minv = np.array([-0.639, -0.501, -0.47, -102.6, -32.5, 184.6], dtype=np.float32)
#     maxv = np.array([0.411, 0.547, 0.433, 159.1, 116.5, 376.5], dtype=np.float32)
#     n=0
#     if capture.isOpened():
#         while True:
#             success, frame = capture.read()
#             if success:
#                 re, tra = headpose.caculate_pose_vector(frame)
#                 pose = np.zeros([256, 256])
#                 # re = np.squeeze(re, 0)
#                 # tra = np.squeeze(tra, 0)
#                 # poses = np.concatenate((re, tra))
#                 # poses = (poses + 1) / 2 * (maxv - minv) + minv
#                 # rot, trans = poses[:3].copy(), poses[3:].copy()
#                 draw_annotation_box(pose, np.array(re), np.array(tra))
#                 n+=1
#                 print(n)
#                 im = Image.fromarray(pose)
#                 im = im.convert("L")
#                 im.save("1.jpg")
#                 img_p = cv2.imread('1.jpg')
#                 videoWriter.write(img_p)
#             else:
#                 break
# videoWriter.release()
