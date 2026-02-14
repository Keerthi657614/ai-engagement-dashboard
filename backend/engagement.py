import cv2
import numpy as np
from collections import deque
import mediapipe as mp

# ============================
# SAFE MEDIAPIPE IMPORT
# ============================

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    from mediapipe import solutions as mp_solutions
    mp_face_mesh = mp_solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)

# ============================
# CONFIG
# ============================

EAR_THRESHOLD = 0.22
GAZE_CENTER_TOL = 0.10
HEAD_YAW_TOL = 10
BUFFER_SIZE = 5

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]
HEAD_POSE_POINTS=[1,33,61,199,263,291]

# ============================
# BUFFERS
# ============================

ear_buffer=deque(maxlen=BUFFER_SIZE)
gaze_buffer=deque(maxlen=BUFFER_SIZE)
yaw_buffer=deque(maxlen=BUFFER_SIZE)
engagement_buffer=deque(maxlen=10)

drowsy_counter=0

# ============================
# HELPERS
# ============================

def dist(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def compute_EAR(landmarks,idx,w,h):
    pts=[(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in idx]
    return (dist(pts[1],pts[5])+dist(pts[2],pts[4]))/(2*dist(pts[0],pts[3])+1e-6)

def iris_gaze(landmarks,w,h):
    iris=(int(landmarks[468].x*w),int(landmarks[468].y*h))
    left=(int(landmarks[33].x*w),int(landmarks[33].y*h))
    right=(int(landmarks[133].x*w),int(landmarks[133].y*h))
    return (iris[0]-left[0])/(right[0]-left[0]+1e-6)

def head_pose_yaw(landmarks,w,h):

    face_2d=[]
    face_3d=[]

    for idx in HEAD_POSE_POINTS:
        x,y=int(landmarks[idx].x*w),int(landmarks[idx].y*h)
        face_2d.append([x,y])
        face_3d.append([x,y,0])

    face_2d=np.array(face_2d,dtype=np.float64)
    face_3d=np.array(face_3d,dtype=np.float64)

    cam_matrix=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]])

    _,rot_vec,_=cv2.solvePnP(face_3d,face_2d,cam_matrix,np.zeros((4,1)))
    rmat,_=cv2.Rodrigues(rot_vec)
    angles,*_=cv2.RQDecomp3x3(rmat)

    return angles[1]*360

# ============================
# MAIN ANALYSIS
# ============================

def analyze_frame(frame):

    global drowsy_counter

    h,w=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)

    state="No Face"
    engagement=0.0
    coords={"x":0.5,"y":0.5}

    if results.multi_face_landmarks:

        landmarks=results.multi_face_landmarks[0].landmark

        coords={"x":landmarks[1].x,"y":landmarks[1].y}

        EAR=(compute_EAR(landmarks,LEFT_EYE,w,h)+
             compute_EAR(landmarks,RIGHT_EYE,w,h))/2

        gaze=iris_gaze(landmarks,w,h)
        yaw=head_pose_yaw(landmarks,w,h)

        ear_buffer.append(EAR)
        gaze_buffer.append(gaze)
        yaw_buffer.append(yaw)

        avg_ear=np.mean(ear_buffer)
        avg_gaze=np.mean(gaze_buffer)
        avg_yaw=np.mean(yaw_buffer)

        eye_score=1 if avg_ear>EAR_THRESHOLD else 0
        gaze_score=1 if abs(avg_gaze-0.5)<GAZE_CENTER_TOL else 0
        pose_score=1 if abs(avg_yaw)<HEAD_YAW_TOL else 0

        engagement=(0.4*pose_score+
                    0.3*gaze_score+
                    0.3*eye_score)

        engagement_buffer.append(engagement)
        engagement=np.mean(engagement_buffer)

        if avg_ear<0.18:
            drowsy_counter+=1
        else:
            drowsy_counter=0

        if drowsy_counter>8:
            state="Drowsy"
        elif gaze_score==0 or pose_score==0:
            state="Looking Away"
        elif engagement>0.75:
            state="Focused"
        else:
            state="Bored"

    return state,float(engagement),coords
