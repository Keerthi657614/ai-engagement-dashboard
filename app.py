import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from collections import deque
import time
import pandas as pd

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Engagement Monitor", layout="wide")

DEVICE="cpu"

emotion_classes=[
    "surprise","fear","disgust",
    "happy","sad","angry","neutral"
]

EAR_THRESHOLD=0.22
GAZE_CENTER_TOL=0.10
HEAD_YAW_TOL=10
YAWN_THRESHOLD=0.45

BUFFER_SIZE=5

# =====================================================
# MODEL
# =====================================================
@st.cache_resource
def load_model():
    model=models.resnet50(weights=None)
    model.fc=torch.nn.Linear(model.fc.in_features,7)
    model.load_state_dict(
        torch.load("models/best_model.pth",map_location=DEVICE)
    )
    model.eval()
    return model

model=load_model()

emotion_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =====================================================
# MEDIAPIPE
# =====================================================
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]
MOUTH_VERT=(13,14)
MOUTH_HORZ=(78,308)
HEAD_POSE_POINTS=[1,33,61,199,263,291]

# =====================================================
# UTILS
# =====================================================
def dist(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def compute_EAR(landmarks,idx,w,h):
    pts=[(int(landmarks[i].x*w),int(landmarks[i].y*h)) for i in idx]
    return (dist(pts[1],pts[5])+dist(pts[2],pts[4]))/(2*dist(pts[0],pts[3])+1e-6)

def mouth_ratio(landmarks,w,h):
    top=(int(landmarks[MOUTH_VERT[0]].x*w),int(landmarks[MOUTH_VERT[0]].y*h))
    bot=(int(landmarks[MOUTH_VERT[1]].x*w),int(landmarks[MOUTH_VERT[1]].y*h))
    left=(int(landmarks[MOUTH_HORZ[0]].x*w),int(landmarks[MOUTH_HORZ[0]].y*h))
    right=(int(landmarks[MOUTH_HORZ[1]].x*w),int(landmarks[MOUTH_HORZ[1]].y*h))
    return dist(top,bot)/(dist(left,right)+1e-6)

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

# =====================================================
# BUFFERS
# =====================================================
ear_buffer=deque(maxlen=BUFFER_SIZE)
gaze_buffer=deque(maxlen=BUFFER_SIZE)
yaw_buffer=deque(maxlen=BUFFER_SIZE)
engagement_buffer=deque(maxlen=10)

analytics={"Focused":0,"Bored":0,"Looking Away":0,"Drowsy":0}

graph_data=deque(maxlen=200)

drowsy_counter=0

# =====================================================
# ANALYZE
# =====================================================
def analyze_frame(img):

    global drowsy_counter

    h,w=img.shape[:2]
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)

    state="No Face"
    engagement=0.0

    if results.multi_face_landmarks:

        landmarks=results.multi_face_landmarks[0].landmark

        EAR=(compute_EAR(landmarks,LEFT_EYE,w,h)+
             compute_EAR(landmarks,RIGHT_EYE,w,h))/2

        gaze=iris_gaze(landmarks,w,h)
        yaw=head_pose_yaw(landmarks,w,h)
        yawn=mouth_ratio(landmarks,w,h)

        ear_buffer.append(EAR)
        gaze_buffer.append(gaze)
        yaw_buffer.append(yaw)

        avg_ear=np.mean(ear_buffer)
        avg_gaze=np.mean(gaze_buffer)
        avg_yaw=np.mean(yaw_buffer)

        # ---- SCORES ----
        eye_score=1 if avg_ear>EAR_THRESHOLD else 0
        gaze_score=1 if abs(avg_gaze-0.5)<GAZE_CENTER_TOL else 0
        pose_score=1 if abs(avg_yaw)<HEAD_YAW_TOL else 0
        yawn_score=0 if yawn>YAWN_THRESHOLD else 1

        engagement=(0.3*pose_score+
                    0.25*gaze_score+
                    0.25*eye_score+
                    0.2*yawn_score)

        engagement_buffer.append(engagement)
        engagement=np.mean(engagement_buffer)

        # -------- DROWSY LOGIC --------
        if avg_ear < 0.18:
            drowsy_counter += 1
        else:
            drowsy_counter = 0

        # -------- PRIORITY STATES --------
        if drowsy_counter > 8:
            state="Drowsy"

        elif gaze_score==0 or pose_score==0:
            state="Looking Away"

        elif engagement > 0.75:
            state="Focused"

        else:
            state="Bored"

        analytics[state]+=1

        # DEBUG
        cv2.putText(img,f"EAR:{avg_ear:.2f}",(20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        cv2.putText(img,f"Gaze:{avg_gaze:.2f}",(20,135),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        cv2.putText(img,f"Yaw:{avg_yaw:.2f}",(20,160),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    graph_data.append(engagement)

    cv2.putText(img,f"State: {state}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),3)

    cv2.putText(img,f"Engagement:{engagement:.2f}",(20,75),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    return img,state

# =====================================================
# UI
# =====================================================
st.title("🎯 Engagement Monitor (Human-like + Analytics)")

run=st.checkbox("Start Camera")

frame_window=st.empty()
graph_placeholder=st.empty()

c1,c2,c3,c4=st.columns(4)
m1,m2,m3,m4=[c.empty() for c in [c1,c2,c3,c4]]

if run:

    cap=cv2.VideoCapture(0)

    while run:

        ret,frame=cap.read()
        if not ret:
            break

        output,state=analyze_frame(frame)

        total=sum(analytics.values())+1e-6

        m1.metric("Focused %",f"{analytics['Focused']/total*100:.1f}")
        m2.metric("Bored %",f"{analytics['Bored']/total*100:.1f}")
        m3.metric("Away %",f"{analytics['Looking Away']/total*100:.1f}")
        m4.metric("Drowsy %",f"{analytics['Drowsy']/total*100:.1f}")

        frame_window.image(
            cv2.cvtColor(output,cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        graph_placeholder.line_chart(
            pd.DataFrame({"Engagement":list(graph_data)})
        )

        time.sleep(0.02)

    cap.release()
