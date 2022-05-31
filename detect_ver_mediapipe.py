import cv2
import numpy as np
import mediapipe as mp
from model import Net
import torch
from imutils import face_utils
from PIL import ImageGrab
import keyboard
import mouse

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MAX_NUM_FACE = 10

IMG_SIZE = (34, 26)
PATH = './weights/classifier_weights_iter_50.pt'

n_count = [0 for i in range(MAX_NUM_FACE)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

mp_face_mesh = mp.solutions.face_mesh
keyboard.add_hotkey("ctrl+1", lambda: set_roi())

# camera object
camera = cv2.VideoCapture(0)


# landmark detection function
def landmarksDetection(img, face_landmarks, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [[int(point.x * img_width), int(point.y * img_height)] for point in
                  face_landmarks.landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
        print(1)

    # returning the list of tuples for each landmarks
    return mesh_coord


# eye cropping function
def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


def predict(pred):
    pred = pred.transpose(1, 3).transpose(2, 3)

    outputs = model(pred)

    pred_tag = torch.round(torch.sigmoid(outputs))

    return pred_tag


def set_roi():
    global ROI_SET, x1, y1, x2, y2
    ROI_SET = False
    print("Select your ROI using mouse drag.")
    while (mouse.is_pressed() == False):
        x1, y1 = mouse.get_position()
        while (mouse.is_pressed() == True):
            x2, y2 = mouse.get_position()
            while (mouse.is_pressed() == False):
                print("Your ROI : {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
                ROI_SET = True
                return


ROI_SET = False
x1, y1, x2, y2 = 0, 0, 0, 0

with mp_face_mesh.FaceMesh(max_num_faces=MAX_NUM_FACE,
                           refine_landmarks=True,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    while True:
        if ROI_SET:
            image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2))), cv2.COLOR_BGR2RGB)
            key = cv2.waitKey(100)
            if key == ord('q') or key == ord('Q'):
                print("Quit")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            img = image.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Draw the face mesh annotations on the image.
            # image.flags.writeable = True
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # return multiple face-landmarks
            results = face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(results.multi_face_landmarks) :
                    mesh_coords = landmarksDetection(image, face_landmarks, False)

                    eye_points_l = np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)
                    eye_points_r = np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)

                    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=eye_points_l)
                    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=eye_points_r)

                    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
                    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
                    eye_img_l = cv2.flip(eye_img_l, flipCode=1)
                    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

                    # wake up 문구 출력 위치 = facial landmark [10]번
                    text_area_x, text_area_y = (mesh_coords[10][0]-70), mesh_coords[10][1]

                    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
                    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

                    eye_input_l = torch.from_numpy(eye_input_l)
                    eye_input_r = torch.from_numpy(eye_input_r)

                    # 학습된 모델로 눈 뜸/감김 예측
                    pred_l = predict(eye_input_l)
                    pred_r = predict(eye_input_r)

                    if pred_l.item() == 0.0 and pred_r.item() == 0.0:
                        n_count[idx] += 1

                    else:
                        n_count[idx] = 0

                    if n_count[idx] > 50:
                        cv2.putText(img, "Wake up", (text_area_x, text_area_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # visualize
                    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
                    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

                    state_l = state_l % pred_l
                    state_r = state_r % pred_r

                    # 사각형 출력 원치 않을 경우 주석 처리
                    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255),
                                  thickness=2)
                    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255),
                                  thickness=2)

                    # state 출력 원치 않을 경우 주석 처리
                    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # cv2.polylines(image, eye_points_l, True,
                    #               (255, 255, 0), 1,
                    #               cv2.LINE_AA)
                    # cv2.polylines(image, eye_points_r, True,
                    #               (255, 255, 0), 1,
                    #               cv2.LINE_AA)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Mesh', img)

    cv2.destroyAllWindows()
#         # cv2.waitKey(0) # 매 프레임마다 캡쳐 / 실시간으로 구현하려면 주석처리
