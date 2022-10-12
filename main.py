from email.mime import image
from detect import detect,craft_text_func,letterbox,Index_Direction_func,Crear_drection_list,Text_detect
import argparse
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
CLASSES = ['Phai', 'Phai-Dia diem', 'Phai-Khoang cach', 'Thang', 'Thang-Dia diem', 'Thang-Khoang cach', 'Trai', 'Trai-Dia diem', 'Trai-Khoang cach', 'Ra1', 'Ra1-Dia diem', 'Ra1-Khoang cach', 'Ra2', 'Ra2-Dia diem', 'Ra2-Khoang cach', 'Ra3', 'Ra3-Dia diem', 'Ra3-Khoang cach', 'Chech Phai', 'Chech Phai-Dia diem', 'Chech Phai-Khoang cach', 'Chech Trai', 'Chech Trai-Dia diem', 'Chech Trai-Khoang cach']
Index_Direction = []
for i in range(0,24,3):
        Index_Direction.append(i)

Direction = []
for i in Index_Direction:
        Direction.append(CLASSES[i])
Huong = ['Rẻ phải','Đi thẳng','Rẻ trái','Đi theo vòng xoay: Hướng ra thứ nhất','Đi theo vòng xoay: Hướng ra thứ hai','Đi theo vòng xoay: Hướng ra thứ ba','Đi chếch phải','Đi chếch trái']
if __name__ == '__main__':
    st.title('YOLOv7 Streamlit App')
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='last.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true',
                        help='don`t trace model')
    opt = parser.parse_args()
    
    opt.weights = "runs/train/exp8/weights/last.pt"
    source = ("Picture", "Video")
    source_index = st.sidebar.selectbox("Select File", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Picture", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            st.sidebar.image(uploaded_file)
            if st.button('Detect'):
                with st.spinner(text='Download resources...'):
                    st.sidebar.image(uploaded_file)
                    picture = Image.open(uploaded_file)
                    picture = picture.save(f'images/{uploaded_file.name}')
                    opt.source = f'images/{uploaded_file.name}'
                    result, image_detect = detect(opt)
                    image_detect = cv2.resize(image_detect, (640, 640))
                    st.subheader('This is a image detected')
                    st.image(image_detect)
                    #st.subheader('This is a text recognized')
                    st.write(result)
                    audio_file = open('audio/gtts.wav', 'rb')
                    audio_bytes = audio_file.read()
                    st.subheader('This is a audio recognized')
                    st.audio(audio_bytes)
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("Video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Download resources...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'


# Convert the file to an opencv image.


# detect_ocr()
