import cv2
import numpy as np
import mediapipe as mp
from model import Net
import torch
from imutils import face_utils
from PIL import ImageGrab
import keyboard
import mouse



##########
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
##########

#
# IMG_SIZE = (34, 26)
# PATH = './weights/classifier_weights_iter_50.pt'
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
# model = Net()
# model.load_state_dict(torch.load(PATH))
# model.eval()
#
# # 최대 졸음 감지 수 = 10 / 조정 가능
# n_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#
# def crop_eye(img, eye_points):
#     x1, y1 = np.amin(eye_points, axis=0)
#     x2, y2 = np.amax(eye_points, axis=0)
#     cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#
#     w = (x2 - x1) * 1.2
#     h = w * IMG_SIZE[1] / IMG_SIZE[0]
#
#     margin_x, margin_y = w / 2, h / 2
#
#     min_x, min_y = int(cx - margin_x), int(cy - margin_y)
#     max_x, max_y = int(cx + margin_x), int(cy + margin_y)
#
#     eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
#
#     eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
#
#     return eye_img, eye_rect
#
#
# def predict(pred):
#     pred = pred.transpose(1, 3).transpose(2, 3)
#
#     outputs = model(pred)
#
#     pred_tag = torch.round(torch.sigmoid(outputs))
#
#     return pred_tag
#
#
# def set_roi():
#     global ROI_SET, x1, y1, x2, y2
#     ROI_SET = False
#     print("Select your ROI using mouse drag.")
#     while(mouse.is_pressed() == False):
#         x1, y1 = mouse.get_position()
#         while(mouse.is_pressed() == True):
#             x2, y2 = mouse.get_position()
#             while(mouse.is_pressed() == False):
#                 print("Your ROI : {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
#                 ROI_SET = True
#                 return
#
#
# keyboard.add_hotkey("ctrl+1", lambda: set_roi())
#
# # cap = cv2.VideoCapture(0)
#
#
#
# cv2.destroyAllWindows()
#
# ROI_SET = False
# x1, y1, x2, y2 = 0, 0, 0, 0
# while True:
#     if ROI_SET == True:
#         image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2))), cv2.COLOR_BGR2RGB)
#         key = cv2.waitKey(100)
#         if key == ord("q"):
#             print("Quit")
#             break
#
#         # img_ori = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5)
#
#         img = image.copy()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         faces = detector(gray)
#
#         for idx, face in enumerate(faces):
#             shapes = predictor(gray, face)
#             shapes = face_utils.shape_to_np(shapes)
#
#             eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
#             eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])
#
#             eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
#             eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
#             eye_img_r = cv2.flip(eye_img_r, flipCode=1)
#
#             # wake up 문구 출력 위치 = facial landmark [6]번
#             text_area_x, text_area_y = shapes[6].astype(np.int)
#             #
#
#             # cv2.imshow('l', eye_img_l)
#             # cv2.imshow('r', eye_img_r)
#
#             eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
#             eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
#
#             eye_input_l = torch.from_numpy(eye_input_l)
#             eye_input_r = torch.from_numpy(eye_input_r)
#
#             # 학습된 모델로 눈 뜸/감김 예측
#             pred_l = predict(eye_input_l)
#             pred_r = predict(eye_input_r)
#
#             if pred_l.item() == 0.0 and pred_r.item() == 0.0:
#                 n_count[idx] += 1
#
#             else:
#                 n_count[idx] = 0
#
#             if n_count[idx] > 50:
#                 cv2.putText(img, "Wake up", (text_area_x, text_area_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#             # visualize
#             state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
#             state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'
#
#             state_l = state_l % pred_l
#             state_r = state_r % pred_r
#
#             # 사각형 출력 원치 않을 경우 주석 처리
#             cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
#             cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)
#
#             # state 출력 원치 않을 경우 주석 처리
#             cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#             cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         #
#         # print(state_l)
#         # print(state_r)
#         cv2.imshow('result', img)
#         # cv2.waitKey(0) # 매 프레임마다 캡쳐 / 실시간으로 구현하려면 주석처리
