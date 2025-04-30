import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import pickle
import uuid
import time
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import torch
from torchvision import models, transforms
from groq import Groq
from openai import OpenAI
import base64
from io import BytesIO
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from imutils import build_montages
import shutil
from datetime import datetime
import hashlib
import random
import tempfile


# 设置页面标题
st.set_page_config(page_title="物体检测与相似搜索工具", layout="wide")

# 初始化session_state以跟踪图片处理状态
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'image' not in st.session_state:
    st.session_state.image = None
if 'last_upload_id' not in st.session_state:
    st.session_state.last_upload_id = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'detected_faces' not in st.session_state:
    st.session_state.detected_faces = []
if 'similar_images' not in st.session_state:
    st.session_state.similar_images = []
if 'similar_faces_results' not in st.session_state:
    st.session_state.similar_faces_results = []
if 'advanced_face_search_done' not in st.session_state:
    st.session_state.advanced_face_search_done = False
if 'face_best_matches' not in st.session_state:
    st.session_state.face_best_matches = []
if 'current_tolerance' not in st.session_state:
    st.session_state.current_tolerance = 0.6
if 'similar_texts' not in st.session_state:
    st.session_state.similar_texts = []
if 'batch_mode' not in st.session_state:
    st.session_state.batch_mode = False
if 'batch_images' not in st.session_state:
    st.session_state.batch_images = []  # 存储批量上传的图片
if 'batch_faces' not in st.session_state:
    st.session_state.batch_faces = []  # 存储批量上传的图片中检测到的人脸
if 'batch_processed_images' not in st.session_state:
    st.session_state.batch_processed_images = []  # 存储批量处理后的图片
if 'batch_image_vectors' not in st.session_state:
    st.session_state.batch_image_vectors = {}  # 存储批量图片的特征向量
if 'batch_face_vectors' not in st.session_state:
    st.session_state.batch_face_vectors = {}  # 存储批量人脸的特征向量
if 'batch_similarity_results' not in st.session_state:
    st.session_state.batch_similarity_results = []  # 存储批量图片间的相似度结果
if 'batch_face_similarity_results' not in st.session_state:
    st.session_state.batch_face_similarity_results = []  # 存储批量人脸间的相似度结果

# 标题
st.title("📷 高精度物体检测与相似搜索工具")
st.write("上传图片，检测人、酒杯等物体，并在历史图片中寻找相似图片和人脸")

# 创建存储目录
def create_directories():
    dirs = ["data", "data/images", "data/faces", "data/vectors"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return Path("data")

data_dir = create_directories()

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    
    # 模型选择
    model_option = st.selectbox(
        "选择模型",
        ["YOLOv11x (最高精度，较慢)", 
         "YOLOv11l (高精度)", 
         "YOLOv11m (中等精度)", 
         "YOLOv11s (较快)", 
         "YOLOv11n (最快，精度较低)",
         "YOLOv8x (最高精度，较慢)", 
         "YOLOv8l (高精度)", 
         "YOLOv8m (中等精度)", 
         "YOLOv8s (较快)", 
         "YOLOv8n (最快，精度较低)"],
        index=4  # 默认选择YOLOv11x
    )
    
    # 模型映射
    model_mapping = {
        "YOLOv11x (最高精度，较慢)": "yolo11x.pt",
        "YOLOv11l (高精度)": "yolo11l.pt",
        "YOLOv11m (中等精度)": "yolo11m.pt",
        "YOLOv11s (较快)": "yolo11s.pt",
        "YOLOv11n (最快，精度较低)": "yolo11n.pt",
        "YOLOv8x (最高精度，较慢)": "yolov8x.pt",
        "YOLOv8l (高精度)": "yolov8l.pt",
        "YOLOv8m (中等精度)": "yolov8m.pt",
        "YOLOv8s (较快)": "yolov8s.pt",
        "YOLOv8n (最快，精度较低)": "yolov8n.pt"
    }
    
    confidence = st.slider("置信度阈值", 0.1, 1.0, 0.5, 0.1)
    
    # 选择要检测的类别
    st.subheader("选择要检测的类别")
    detect_person = st.checkbox("人", value=True)
    detect_cup = st.checkbox("杯子/酒杯", value=True)
    detect_bottle = st.checkbox("瓶子", value=True)
    detect_all = st.checkbox("检测所有支持的物体", value=True)
    
    # 人脸检测选项
    st.subheader("人脸检测")
    detect_faces = st.checkbox("启用人脸检测", value=True)
    face_confidence = st.slider("人脸检测置信度", 0.1, 1.0, 0.8, 0.1)
    
    # 相似搜索选项
    st.subheader("相似搜索")
    enable_similarity_search = st.checkbox("启用相似图片搜索", value=True)
    enable_face_similarity = st.checkbox("启用人脸相似搜索", value=True)
    similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5, 0.05)
    top_k = st.slider("返回相似结果数量", 1, 10, 3)
    
    # 获取颜色设置
    st.subheader("标记颜色")
    box_color = st.color_picker("边框颜色", "#FF0000")
    face_box_color = st.color_picker("人脸边框颜色", "#00FF00")

    # 添加加强人脸检索选项
    st.subheader("加强人脸检索")
    enable_advanced_face_search = st.checkbox("启用加强人脸检索", value=True)
    if enable_advanced_face_search:
        face_cluster_tolerance = st.slider("人脸匹配严格度", 0.4, 0.8, 0.6, 0.05, help="数值越低匹配越严格")
        min_cluster_size = st.slider("最小聚类数量", 1, 10, 3, 1, help="形成聚类所需的最小图片数量")

    # 在侧边栏添加批量模式选项
    st.subheader("批量处理")
    batch_mode = st.checkbox("启用批量处理模式", value=False)
    if batch_mode:
        st.session_state.batch_mode = True
        st.info("批量模式已启用，您可以上传多张图片进行处理")
        if st.button("清除批量处理数据"):
            st.session_state.batch_images = []
            st.session_state.batch_faces = []
            st.session_state.batch_processed_images = []
            st.session_state.batch_image_vectors = {}
            st.session_state.batch_face_vectors = {}
            st.session_state.batch_similarity_results = []
            st.session_state.batch_face_similarity_results = []
            st.success("批量处理数据已清除")
    else:
        st.session_state.batch_mode = False

# 加载YOLO模型
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

# 加载人脸检测模型
@st.cache_resource
def load_face_detector():
    # 使用MTCNN人脸检测器
    detector = MTCNN()
    return detector


# 加载特征提取模型
@st.cache_resource
def load_feature_extractor():
    # 使用预训练的ResNet50
    model = models.resnet50(weights='DEFAULT')
    # 移除最后的全连接层
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

# 图像预处理转换
@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 提取图像特征
def extract_image_features(image, model, transform):
    # 预处理图像
    img_t = transform(image).unsqueeze(0)
    
    # 提取特征
    with torch.no_grad():
        features = model(img_t)
    
    # 返回特征向量
    return features.squeeze().cpu().numpy()

# 提取人脸特征
def extract_face_features(face_image, model, transform):
    # 如果人脸图像太小，可能需要调整大小
    if face_image.size[0] < 30 or face_image.size[1] < 30:
        return None
    
    # 预处理人脸图像
    face_t = transform(face_image).unsqueeze(0)
    
    # 提取特征
    with torch.no_grad():
        features = model(face_t)
    
    # 返回特征向量
    return features.squeeze().cpu().numpy()

# 初始化向量库
def initialize_vector_db():
    vector_file = data_dir / "vectors" / "image_vectors.pkl"
    face_vector_file = data_dir / "vectors" / "face_vectors.pkl"
    
    # 图像向量数据库
    if os.path.exists(vector_file):
        with open(vector_file, 'rb') as f:
            image_db = pickle.load(f)
    else:
        # 创建新的向量数据库
        # 使用字典存储: {image_id: {'vector': feature_vector, 'path': image_path}}
        image_db = {}
    
    # 人脸向量数据库
    if os.path.exists(face_vector_file):
        with open(face_vector_file, 'rb') as f:
            face_db = pickle.load(f)
    else:
        # 创建新的人脸向量数据库
        # 使用字典存储: {face_id: {'vector': feature_vector, 'path': face_path, 'image_id': parent_image_id}}
        face_db = {}
    
    return image_db, face_db

# 保存向量数据库
def save_vector_db(image_db, face_db):
    vector_file = data_dir / "vectors" / "image_vectors.pkl"
    face_vector_file = data_dir / "vectors" / "face_vectors.pkl"
    
    with open(vector_file, 'wb') as f:
        pickle.dump(image_db, f)
    
    with open(face_vector_file, 'wb') as f:
        pickle.dump(face_db, f)

# 更新向量数据库
def update_vector_db(image, faces, feature_extractor, transform):
    # 初始化向量数据库
    image_db, face_db = initialize_vector_db()
    
    # 生成唯一ID
    image_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # 保存图像
    image_filename = f"{image_id}.jpg"
    image_path = data_dir / "images" / image_filename
    # 确保图像为RGB模式
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(image_path)
    
    # 提取图像特征
    image_features = extract_image_features(image, feature_extractor, transform)
    
    # 更新图像向量数据库
    image_db[image_id] = {
        'vector': image_features,
        'path': str(image_path),
        'timestamp': timestamp
    }
    
    # 处理人脸
    face_ids = []
    for i, face in enumerate(faces):
        face_id = f"{image_id}_face_{i}"
        face_filename = f"{face_id}.jpg"
        face_path = data_dir / "faces" / face_filename
        
        # 将PIL Image转换为numpy数组并保存
        face_pil = Image.fromarray(face)
        # 确保人脸图像为RGB模式
        if face_pil.mode == 'RGBA':
            face_pil = face_pil.convert('RGB')
        face_pil.save(face_path)
        
        # 提取人脸特征
        face_features = extract_face_features(face_pil, feature_extractor, transform)
        
        if face_features is not None:
            # 更新人脸向量数据库
            face_db[face_id] = {
                'vector': face_features,
                'path': str(face_path),
                'image_id': image_id,
                'timestamp': timestamp
            }
            face_ids.append(face_id)
    
    # 保存更新后的向量数据库
    save_vector_db(image_db, face_db)
    
    return image_id, face_ids

# 执行相似搜索
def search_similar_images(query_vector, image_db, top_k=3, threshold=0.6):
    if not image_db:
        return []
    
    results = []
    
    # 计算查询向量与数据库中所有向量的相似度
    for image_id, data in image_db.items():
        db_vector = data['vector']
        similarity = cosine_similarity([query_vector], [db_vector])[0][0]
        
        if similarity >= threshold:
            results.append({
                'id': image_id,
                'path': data['path'],
                'similarity': similarity,
                'timestamp': data.get('timestamp', 0)
            })
    
    # 按相似度降序排序
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 返回前K个结果
    return results[:top_k]

# 执行人脸相似搜索
def search_similar_faces(query_vectors, face_db, top_k=3, threshold=0.6):
    if not face_db or not query_vectors:
        return []
    
    all_results = []
    
    # 对每个查询人脸进行相似度计算
    for i, query_vector in enumerate(query_vectors):
        results = []
        
        # 计算查询向量与数据库中所有人脸向量的相似度
        for face_id, data in face_db.items():
            db_vector = data['vector']
            similarity = cosine_similarity([query_vector], [db_vector])[0][0]
            
            if similarity >= threshold:
                results.append({
                    'id': face_id,
                    'path': data['path'],
                    'image_id': data['image_id'],
                    'similarity': similarity,
                    'timestamp': data.get('timestamp', 0)
                })
        
        # 按相似度降序排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 为当前查询人脸保存前K个结果
        all_results.append({
            'query_face_index': i,
            'matches': results[:top_k]
        })
    
    return all_results

# 处理预测结果并绘制边框
def process_prediction(image, results, selected_classes, conf_threshold):
    # 转换为OpenCV格式用于绘图
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # COCO数据集类别
    class_names = results[0].names
    
    # 如果选择了"检测所有支持的物体"，将显示所有类别
    if "检测所有支持的物体" in selected_classes:
        selected_class_ids = list(class_names.keys())
    else:
        # 创建类别名称到ID的映射
        class_mapping = {
            "人": 0,  # person
            "杯子/酒杯": 41,  # cup
            "瓶子": 39,  # bottle
        }
        # 获取选中类别的ID
        selected_class_ids = [class_mapping[cls] for cls in selected_classes if cls in class_mapping]
    
    # 解析十六进制颜色为BGR
    hex_color = box_color.lstrip('#')
    box_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # 转换为BGR
    
    # 如果有检测结果
    if results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            # 获取类别ID和置信度
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            
            # 如果类别在选中的类别中且置信度高于阈值
            if cls_id in selected_class_ids and conf >= conf_threshold:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), box_bgr, 2)
                
                # 添加标签
                label = f"{class_names[cls_id]} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), box_bgr, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # 转换回RGB用于Streamlit显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 人脸检测函数
def detect_face(image, face_detector, conf_threshold):
    # 转换为numpy数组并确保RGB格式
    img = np.array(image)
    if img.shape[-1] == 4:  # 检查是否为RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # 转换为RGB
    img_rgb = img.copy()
    
    # 使用MTCNN检测人脸
    faces = face_detector.detect_faces(img_rgb)
    
    # 筛选置信度高于阈值的人脸
    faces = [face for face in faces if face['confidence'] >= conf_threshold]
    
    # 解析人脸边框颜色
    hex_color = face_box_color.lstrip('#')
    face_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # 转换为BGR
    
    # 提取的人脸图像列表
    face_images = []
    
    # 在原图上标记人脸
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for face in faces:
        # 获取边界框坐标
        x, y, w, h = face['box']
        
        # 绘制人脸边框
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), face_bgr, 2)
        
        # 添加标签和置信度
        label = f"Face: {face['confidence']:.2f}"
        cv2.putText(img_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_bgr, 2)
        
        # 可选：绘制关键点
        keypoints = face['keypoints']
        for point in keypoints.values():
            cv2.circle(img_cv, point, 2, face_bgr, 2)
        
        # 提取人脸区域
        face_crop = img_rgb[y:y+h, x:x+w]
        face_images.append(face_crop)
    
    # 转换回RGB
    result_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    return result_img, face_images

# 显示相似图片结果
def display_similar_images(similar_images):
    if not similar_images:
        st.info("未找到相似图片")
        return
    
    st.subheader(f"相似图片检索结果 (Top {len(similar_images)})")
    
    # 显示结果
    cols = st.columns(min(len(similar_images), 3))
    for i, result in enumerate(similar_images):
        with cols[i % 3]:
            image_path = Path(result['path'])
            if image_path.exists():
                img = Image.open(image_path)
                st.image(img, caption=f"相似度: {result['similarity']:.2f}", use_container_width=True)
                
                # 显示时间戳（转换为可读格式）
                if 'timestamp' in result:
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))
                    st.caption(f"上传时间: {time_str}")

# 显示相似人脸结果
def display_similar_faces(similar_faces_results, detected_faces):
    if not similar_faces_results:
        st.info("未找到相似人脸")
        return
    
    st.subheader("人脸相似度检索结果")
    
    # 为每个查询人脸显示其匹配结果
    for result in similar_faces_results:
        query_face_idx = result['query_face_index']
        matches = result['matches']
        
        if matches:
            st.write(f"##### 查询人脸 #{query_face_idx + 1} 的匹配结果:")
            
            # 显示查询人脸
            query_face = detected_faces[query_face_idx]
            query_face_pil = Image.fromarray(query_face)
            
            # 创建列来显示查询人脸和匹配结果
            cols = st.columns(1 + min(len(matches), 3))
            
            # 显示查询人脸
            with cols[0]:
                st.image(query_face_pil, caption="查询人脸", use_container_width=True)
            
            # 显示匹配结果
            for i, match in enumerate(matches):
                if i < len(cols) - 1:  # 确保不超出列数
                    with cols[i + 1]:
                        face_path = Path(match['path'])
                        if face_path.exists():
                            match_face = Image.open(face_path)
                            st.image(match_face, caption=f"相似度: {match['similarity']:.2f}", use_container_width=True)
                            
                            # 显示时间戳（转换为可读格式）
                            if 'timestamp' in match:
                                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(match['timestamp']))
                                st.caption(f"上传时间: {time_str}")

# 修改图片转base64函数
def image_to_base64(image):
    # 确保图片是RGB模式
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# 修改analyze_with_groq函数
def analyze_with_groq(image, api_key):
    client_vision = Groq()
    client_vision.api_key = api_key
    
    try:
        # 确保图片是RGB模式
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
            
        base64_image = image_to_base64(image)
        
        response = client_vision.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "system",
                    "content": '''请按以下规则分析图片内容：
                    首先判断图片主要内容是否为文字内容（即图片中超过 50% 的区域为可识别的文字，如文档、海报、标语等）：
                    若是：直接提取图片中的所有可识别文字（无需额外描述）。
                    若否：详细描述图片内容，需包含以下信息（无相关信息则标注 "未提及" 或 "无法判断"）：
                    # 主体内容：清晰描述画面中的物体、场景、动作等细节（如 "室内会议场景，3 人围坐讨论，桌上摆放笔记本电脑和文件"）。
                    # 文字信息：尽可能提取图片里的文字信息
                    # 人物信息（若包含人）：
                    ・人头数：X 人（精确计数）。
                    ・性别比例：男性 X 人，女性 Y 人，性别不明 Z 人（无人物则填 0）。
                    ・种族比例：黑人，白人，亚洲人等
                    ・年龄段：按儿童（＜12 岁）、青年（12-35 岁）、中年（36-60 岁）、老年（＞60 岁）分类，标注各年龄段人数（如 "青年 2 人，中年 1 人"）。
                    # 所在场景：具体描述环境（如 "医院候诊区""户外公园""实验室" 等），需包含室内 / 室外、地点特征等细节。
                    # Sanofi 相关性：明确说明是否出现 Sanofi 标志（如 LOGO、品牌名称）、产品（如药品包装、宣传材料）、文字提及（如 "Sanofi""赛诺菲" 字样）或相关场景（如 Sanofi 活动、合作项目等），若无则标注 "无相关元素"。
                    使用说明：
                    若图片为纯文字（如合同、说明书），输出仅包含提取的文字内容。
                    若图片为非文字内容（如照片、插画），按上述格式分点详细描述，人物信息需严格按要求分类统计。
                    确保语言简洁准确，避免主观推断，仅基于图片可见内容输出。
                    新增验证项
                    针对以下场景，明确标注工具是否正确识别（需基于图片可见内容判断，而非主观推测）：
                    # 图片风格： 是否为真实照片，还是某种风格图像, 是否为视频截图或者翻拍，还是自拍
                    # 空场景识别：图片是否为无人、无明显物体的空场景（如空白墙面、纯色背景）？
                    # 识别结果：是 / 否 / 无法判断（若有物体或人物，标注 "非空场景"）。
                    # 会议无关场景识别：场景是否与会议无直接关联（如纯自然风景、单独静物、娱乐场所等）？
                    # 识别结果：是（会议无关）/ 否（可能与会议相关）/ 无法判断。
                    # 人数捕捉：工具是否准确统计图片中的人头数？（若有人物，需与实际计数一致）
                    # 实际人头数：X 人；工具识别结果：X 人（一致 / 不一致）。
                    # 会议设备识别：是否正确识别会议必要设备（如投影仪、电脑、麦克风、白板、会议桌等）？
                    # 识别设备：若有，列出具体设备名称（如 "投影仪、笔记本电脑"）；若无，标注 "未识别到会议设备"。
                    # 会议照片翻拍识别：图片是否为会议照片的翻拍（如对屏幕、纸质照片的二次拍摄，可能存在反光、变形等特征）？
                    # 识别结果：是（翻拍照片）/ 否（非翻拍照片）/ 无法判断。
                    # 非正常会议场所识别：场景是否为非正常会议场所（如娱乐场所、开放区域、家庭环境等）？
                    # 场所类型：正常会议场所（如会议室）/ 非正常会议场所（具体描述，如 "KTV 包厢""公园草坪"）/ 无法判断。
                    # 参会人露脸识别：是否正确识别人物露脸情况（全部露脸 / 部分未露脸 / 全部未露脸）？
                    # 露脸情况：若有人物，标注 "全部露脸" 或 "X 人未露脸"；无人物则标注 "无人物"。
                    # 非合规物件识别：是否出现与会议无关的非合规物件（如酒瓶、麻将桌、酒杯，游戏机等），需要着重检查？
                    # 识别结果：是（具体物件：XXX）/ 否（无非合规物件）/ 无法判断。
                    '''
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "请分析这张图片"
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq分析出错: {str(e)}"

# 添加OpenAI审查总结函数
def summarize_with_openai(cv_results, groq_analysis, api_key, image_dimensions=None):
    client = OpenAI(api_key=api_key,base_url="https://generativelanguage.googleapis.com/v1beta/")
    try:
        # 将图片尺寸信息添加到CV结果中
        if image_dimensions:
            cv_results["image_dimensions"] = image_dimensions
            
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": """你是一位严谨的药厂审查关员。请根据CV检测结果和AI分析结果，生成一份简洁的合规性分析报告。
                    重点关注：
                    0. 是否有sanofi的logo，或者明确和sanofi相关的因素，对于明确的sanofi logo，不建议直接判断为违规
                    1. 场景合规性（是否为正式会议场所),根据主体内容信息判断, 室外场景不被允许
                    2. 人员情况（人数、着装、行为是否得体），若人数为0，不被允许
                    3. 物品合规性（是否有违规物品如酒精饮品）
                    4. 图片风格是否为真实，仅真实照片被允许。若是截图或者翻拍，以及自拍，不被允许
                    5. 文字信息合规性:总结AI分析结果内的提取图片内容中的文字信息，如果有的话，判断是否合适
                    6. 图片尺寸: 是否存在拉伸或者压缩，是否为非标准照片尺寸，若非标准尺寸，仅提醒可能被裁剪，不建议直接判断为违规，常见的尺寸为：
                    1.0: "1:1 (正方形)",
                    1.33: "4:3 (常见屏幕)",
                    1.5: "3:2 (传统相机)",
                    1.78: "16:9 (宽屏)",
                    1.85: "1.85:1 (电影)",
                    2.35: "2.35:1 (电影宽银幕)"
                    7. 整体评估（是否建议通过审核）
                    请以表格形式输出，表格需要包含审核项目，审核结果，审核原因三列，对于违规情况，加粗字体，使用markdown格式。"""
                },
                {
                    "role": "user",
                    "content": f"CV检测结果：{cv_results}\n\nAI分析结果：{groq_analysis}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI总结出错: {str(e)}"

# 添加哈希计算函数
def get_image_hash(image_path):
    """
    计算图片的哈希值用于去重
    """
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"[ERROR] Error hashing image {image_path}: {e}")
        return None

# 添加人脸聚类函数
def face_clustering(input_folder, output_folder, detection_method="mtcnn", tolerance=0.6, min_cluster_size=3,
                  dim_reduction=True, n_components=50):
    """
    Cluster similar faces from a collection of images and organize them into folders
    
    Parameters:
    - input_folder: Path to folder containing images
    - output_folder: Path to output the clustered images
    - detection_method: Ignored (always uses MTCNN)
    - tolerance: How strict the face matching should be (lower is stricter)
    - min_cluster_size: Minimum number of images needed to form a cluster
    - dim_reduction: Whether to apply dimensionality reduction (default: True)
    - n_components: Number of components for dimensionality reduction (default: 50)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a directory for images where no faces were detected
    no_faces_dir = os.path.join(output_folder, "no_faces_detected")
    if not os.path.exists(no_faces_dir):
        os.makedirs(no_faces_dir)
    
    # Create a directory for unclustered faces (faces that don't belong to any cluster)
    unclustered_dir = os.path.join(output_folder, "unclustered_faces")
    if not os.path.exists(unclustered_dir):
        os.makedirs(unclustered_dir)
    
    print(f"[INFO] Processing images from {input_folder}")
    
    # Get all image files from the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    print(f"[INFO] Found {len(image_paths)} images")
    
    # Initialize MTCNN detector
    face_detector = MTCNN()
    
    # Initialize lists to store data
    known_embeddings = []
    known_image_paths = []
    
    # Process each image
    for (i, image_path) in enumerate(image_paths):
        print(f"[INFO] Processing image {i+1}/{len(image_paths)}")
        
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Could not read image {image_path}")
                continue
                
            # Convert BGR to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces using MTCNN
            faces = face_detector.detect_faces(rgb)
            
            # If no faces were found, move to the no_faces directory
            if len(faces) == 0:
                shutil.copy2(image_path, os.path.join(no_faces_dir, os.path.basename(image_path)))
                continue
            
            # Process each detected face
            for face in faces:
                if face['confidence'] < 0.8:  # Confidence threshold
                    continue
                
                # Extract face bounding box
                x, y, w, h = face['box']
                x = max(x, 0)
                y = max(y, 0)
                face_crop = rgb[y:y+h, x:x+w]
                
                # Ensure face crop is valid
                if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                    continue
                
                # Generate face embedding using DeepFace
                try:
                    embedding = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    known_embeddings.append(embedding)
                    known_image_paths.append(image_path)
                except Exception as e:
                    print(f"[ERROR] Error generating embedding for face in {image_path}: {e}")
                    continue
        
        except Exception as e:
            print(f"[ERROR] Error processing {image_path}: {e}")
    
    print(f"[INFO] Detected {len(known_embeddings)} faces in total")
    
    # Cluster the faces
    if len(known_embeddings) > 0:
        print("[INFO] Clustering faces...")
        
        # Convert embeddings to numpy array
        data = np.array(known_embeddings)
        
        # Apply dimensionality reduction if enabled
        if dim_reduction and len(data) > n_components:
            print(f"[INFO] Applying dimensionality reduction from {data.shape[1]} to {n_components} dimensions")
            from sklearn.decomposition import PCA
            
            # Initialize and fit PCA
            pca = PCA(n_components=n_components, random_state=42)
            data = pca.fit_transform(data)
            
            # Calculate and display explained variance
            explained_variance = sum(pca.explained_variance_ratio_) * 100
            print(f"[INFO] Explained variance with {n_components} components: {explained_variance:.2f}%")
        
        # Normalize the data
        print("[INFO] Normalizing feature vectors")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        
        # Perform DBSCAN clustering
        clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=tolerance, min_samples=min_cluster_size)
        clt.fit(data)
        
        # Get the number of unique clusters
        label_ids = np.unique(clt.labels_)
        num_unique_faces = len(np.where(label_ids > -1)[0])
        
        print(f"[INFO] Found {num_unique_faces} unique faces")
        
        # Create a folder for each cluster and copy the images there
        for label_id in label_ids:
            # If the cluster is -1, it's noise (unclustered)
            if label_id == -1:
                print("[INFO] Copying unclustered faces...")
                # Get indices of faces in this cluster
                indices = np.where(clt.labels_ == label_id)[0]
                
                # 用于存储已处理的图片哈希值，用于去重
                processed_hashes = set()
                
                # Copy each image to the unclustered directory
                for idx in indices:
                    image_path = known_image_paths[idx]
                    # 计算图片哈希值
                    image_hash = get_image_hash(image_path)
                    
                    # 如果图片已经处理过（重复），则跳过
                    if image_hash and image_hash in processed_hashes:
                        print(f"[INFO] Skipping duplicate image: {os.path.basename(image_path)}")
                        continue
                    
                    # 添加哈希值到已处理集合
                    if image_hash:
                        processed_hashes.add(image_hash)
                        
                    shutil.copy2(image_path, os.path.join(unclustered_dir, os.path.basename(image_path)))
            else:
                # Create a directory for this cluster
                cluster_dir = os.path.join(output_folder, f"person_{label_id}")
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                
                # Get indices of faces in this cluster
                indices = np.where(clt.labels_ == label_id)[0]
                
                # 用于存储已处理的图片哈希值，用于去重
                processed_hashes = set()
                # 存储已处理（去重后）的图片路径
                unique_image_paths = []
                
                # Copy each image to this cluster's directory
                for idx in indices:
                    image_path = known_image_paths[idx]
                    # 计算图片哈希值
                    image_hash = get_image_hash(image_path)
                    
                    # 如果图片已经处理过（重复），则跳过
                    if image_hash and image_hash in processed_hashes:
                        print(f"[INFO] Skipping duplicate image: {os.path.basename(image_path)}")
                        continue
                    
                    # 添加哈希值到已处理集合
                    if image_hash:
                        processed_hashes.add(image_hash)
                        unique_image_paths.append(image_path)
                        
                    shutil.copy2(image_path, os.path.join(cluster_dir, os.path.basename(image_path)))
                
                print(f"[INFO] Cluster {label_id}: {len(indices)} faces, {len(unique_image_paths)} unique images after deduplication")
    
    print("[INFO] Face clustering completed!")
    return output_folder

# 添加创建蒙太奇图的函数
def create_montages(output_folder, montage_size=(150, 150), images_per_row=5):
    """
    Create montages for each cluster to visualize the results
    """
    montages_info = []
    
    # Get all subdirectories (clusters)
    cluster_dirs = [d for d in os.listdir(output_folder) 
                    if os.path.isdir(os.path.join(output_folder, d)) and d.startswith("person_")]
    
    montages_dir = os.path.join(output_folder, "montages")
    if not os.path.exists(montages_dir):
        os.makedirs(montages_dir)
    
    for cluster_dir in cluster_dirs:
        # Get all images in this cluster
        image_paths = [os.path.join(output_folder, cluster_dir, f) 
                      for f in os.listdir(os.path.join(output_folder, cluster_dir))
                      if os.path.isfile(os.path.join(output_folder, cluster_dir, f))]
        
        # Skip if no images in the cluster
        if len(image_paths) == 0:
            print(f"[INFO] No images found in cluster {cluster_dir}, skipping montage creation")
            continue
        
        # 去重：检查图片内容是否相同
        unique_images = []
        image_hashes = set()
        
        for image_path in image_paths:
            image_hash = get_image_hash(image_path)
            
            # 如果这个哈希值没有见过，说明是不重复的图片
            if image_hash and image_hash not in image_hashes:
                image_hashes.add(image_hash)
                unique_images.append(image_path)
        
        print(f"[INFO] Cluster {cluster_dir}: {len(image_paths)} images, {len(unique_images)} unique for montage")
        
        # Load and resize images (只使用去重后的图片)
        images = []
        for image_path in unique_images:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, montage_size)
                images.append(image)
        
        # Create montage
        if images:
            montage = build_montages(images, montage_size, (images_per_row, (len(images) // images_per_row) + 1))[0]
            
            # Save the montage
            montage_path = os.path.join(montages_dir, f"{cluster_dir}_montage.jpg")
            cv2.imwrite(montage_path, montage)
            
            print(f"[INFO] Created montage for {cluster_dir} with {len(images)} unique images")
            
            # 添加蒙太奇路径和信息到返回列表
            montages_info.append({
                "path": montage_path,
                "cluster": cluster_dir,
                "images_count": len(images)
            })
        else:
            print(f"[INFO] No valid images for montage in {cluster_dir}")
    
    print("[INFO] Montage creation completed!")
    return montages_info

# 修改process_face_for_advanced_search函数，为每个检测到的人脸找到最相似的聚类
def process_face_for_advanced_search(detected_faces, tolerance=0.6):
    # 查找当前目录下的聚类结果目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 查找所有以clustered_faces_开头的目录
    cluster_dirs = [d for d in os.listdir(current_dir) 
                   if os.path.isdir(os.path.join(current_dir, d)) and d.startswith("clustered_faces_")]
    
    if not cluster_dirs:
        st.warning("未在当前目录下找到人脸聚类数据(clustered_faces_*)，请先运行离线聚类工具")
        return []
    
    # 按创建时间排序，使用最新的聚类结果
    cluster_dirs.sort(reverse=True)
    latest_cluster_dir = os.path.join(current_dir, cluster_dirs[0])
    
    # 蒙太奇图片路径
    montages_dir = os.path.join(latest_cluster_dir, "montages")
    if not os.path.exists(montages_dir):
        st.warning(f"未找到蒙太奇图片目录: {montages_dir}")
        return []
    
    # 获取所有人脸聚类目录
    person_dirs = [d for d in os.listdir(latest_cluster_dir) 
                  if os.path.isdir(os.path.join(latest_cluster_dir, d)) and d.startswith("person_")]
    
    if not person_dirs:
        st.warning("未找到有效的人脸聚类组")
        return []
    
    # Initialize MTCNN detector
    face_detector = MTCNN()
    
    # 结果列表，为每个检测到的人脸存储最相似的聚类
    face_best_matches = []
    
    # 处理每个检测到的人脸
    for face_idx, face in enumerate(detected_faces):
        try:
            # 将numpy数组转换为RGB格式
            if len(face.shape) == 3 and face.shape[2] == 3:
                rgb_face = face
            else:
                # 如果不是3通道彩色图像，跳过
                continue
                
            # Generate embedding for the detected face using DeepFace
            current_face_embedding = DeepFace.represent(rgb_face, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            
            # 为当前人脸寻找最相似的聚类
            best_match = None
            best_similarity = 0
            
            for person_dir in person_dirs:
                person_path = os.path.join(latest_cluster_dir, person_dir)
                # 获取该聚类中的所有图片
                face_images = [f for f in os.listdir(person_path) 
                              if os.path.isfile(os.path.join(person_path, f)) and 
                              f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                if not face_images:
                    continue
                
                # 从该聚类中随机选取几张图片进行相似度计算
                sample_size = min(5, len(face_images))
                sample_images = random.sample(face_images, sample_size)
                
                cluster_embeddings = []
                for img_file in sample_images:
                    img_path = os.path.join(person_path, img_file)
                    try:
                        # 读取图片
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        # 转换为RGB
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # 检测人脸
                        faces = face_detector.detect_faces(rgb_img)
                        if faces:
                            # Use the first detected face
                            x, y, w, h = faces[0]['box']
                            x = max(x, 0)
                            y = max(y, 0)
                            face_crop = rgb_img[y:y+h, x:x+w]
                            
                            # Generate embedding
                            if face_crop.size > 0 and face_crop.shape[0] >= 30 and face_crop.shape[1] >= 30:
                                embedding = DeepFace.represent(face_crop, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                                cluster_embeddings.append(embedding)
                    except Exception as e:
                        print(f"处理图片 {img_path} 时出错: {e}")
                
                if not cluster_embeddings:
                    continue
                
                # 计算当前人脸与聚类中所有人脸的相似度
                similarities = []
                for cluster_embedding in cluster_embeddings:
                    similarity = cosine_similarity([current_face_embedding], [cluster_embedding])[0][0]
                    similarities.append(similarity)
                
                # 取最高相似度
                if similarities:
                    max_similarity = max(similarities)
                    
                    # 如果相似度高于阈值且高于当前最高相似度
                    if max_similarity >= tolerance and max_similarity > best_similarity:
                        # 查找对应的蒙太奇图
                        montage_path = os.path.join(montages_dir, f"{person_dir}_montage.jpg")
                        if os.path.exists(montage_path):
                            best_similarity = max_similarity
                            best_match = {
                                "path": montage_path,
                                "cluster": person_dir,
                                "similarity": max_similarity,
                                "images_count": len(face_images),
                                "face_idx": face_idx
                            }
            
            # 如果找到最佳匹配，添加到结果列表
            if best_match:
                face_best_matches.append(best_match)
                
        except Exception as e:
            print(f"处理人脸 #{face_idx} 时出错: {e}")
    
    return face_best_matches

def check_image_dimensions(image):
    width, height = image.size

    # 标准的照片尺寸列表（去掉明显不标准的，比如 1920x1281）
    standard_sizes = [
        (640, 480), (800, 600), (1024, 768), (1280, 720),
        (1280, 800), (1280, 1024), (1366, 768), (1440, 900),
        (1600, 900), (1600, 1200), (1920, 1080), (2048, 1536),
        (2560, 1440), (3840, 2160), (4096, 2160), (7680, 4320)
    ]

    # 设置容差范围（单位：像素）
    tolerance = 10

    def within_tolerance(w1, h1, w2, h2, tol):
        return abs(w1 - w2) <= tol and abs(h1 - h2) <= tol

    # 检查是否为标准尺寸或翻转版本，加入容差
    is_standard = any(
        within_tolerance(width, height, std_w, std_h, tolerance) or
        within_tolerance(height, width, std_w, std_h, tolerance)
        for std_w, std_h in standard_sizes
    )

    # 获取宽高比
    aspect_ratio = width / height

    # 常见宽高比及名称
    common_ratios = {
        1.0: "1:1 (正方形)",
        4/3: "4:3 (常见屏幕)",
        3/2: "3:2 (传统相机)",
        16/9: "16:9 (宽屏)",
        1.85: "1.85:1 (电影)",
        2.35: "2.35:1 (电影宽银幕)"
    }

    # 判断最接近的宽高比
    closest_ratio = min(common_ratios, key=lambda x: abs(x - aspect_ratio))
    ratio_name = common_ratios[closest_ratio] if abs(closest_ratio - aspect_ratio) < 0.1 else "非标准"

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "ratio_name": ratio_name,
        "is_standard_size": is_standard
    }

# ... existing code ...
def extract_text_from_analysis(analysis):
    client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"),base_url="https://generativelanguage.googleapis.com/v1beta/")
    try: 
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": "你的任务是从AI分析结果中提取内容里的文字信息的部分，并明确排除不需要的信息，直接返回结果即可。 仅输出分析中提到的图片里包含的文字内容，不要添加任何解释或评论。如果没有找到文字信息，请回复'无文字内容'。"
                },
                {
                    "role": "user",
                    "content": f"请从以下AI分析中提取所有文字信息：\n\n{analysis}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"提取文字信息失败: {str(e)}"
    
def initialize_text_vectordb():
    text_vector_file = data_dir / "vectors" / "text_vectors.pkl"

    if os.path.exists(text_vector_file):
        with open(text_vector_file, "rb") as f:
            text_db = pickle.load(f)
    else:
   
   
        return {}
    
    return text_db
    
def save_text_vector_db(text_db):
    text_vector_file = data_dir / "vectors" / "text_vectors.pkl"

    with open(text_vector_file, "wb") as f:
        pickle.dump(text_db, f)

def update_text_vector_db(image_id, text_content):
    if not text_content or text_content == "无文字内容":
        return
    
    # 初始化文字向量数据库
    text_db = initialize_text_vectordb()

    try:

        from sentence_transformers import SentenceTransformer


        @st.cache_resource
        def load_sentence_transformer():
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        model = load_sentence_transformer()


        text_vector = model.encode(text_content)

        
        text_db[image_id] = {
            'vector': text_vector,
            'text': text_content
        }

        # 保存更新后的数据库
        save_text_vector_db(text_db)
    except Exception as e:
        st.error(f"生成文本向量数据库失败: {str(e)}")
        print(f"生成文本向量数据库失败: {str(e)}")
    
def search_similar_texts(image_id, top_k=3, threshold=0.5):
    text_db = initialize_text_vectordb()

    if not text_db or image_id not in text_db:
        return []
    
    query_vector = text_db[image_id]['vector']
    results = []


    for text_id, data in text_db.items():
        if text_id == image_id:
            continue
        
        db_vector = data['vector']

        try:


            similarity = np.dot(query_vector, db_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(db_vector))

            if similarity >= threshold:
                results.append({
                    'id': text_id,
                    'text': data['text'],
                    'similarity': float(similarity)
                })
        except Exception as e:
            print(f"计算文本向量相似度失败: {str(e)}")
            continue
    
    # 按相似度排序
    results.sort(key=lambda x: x['similarity'], reverse=True)


    return results[:top_k]

def display_similar_texts(similar_texts):
    if not similar_texts:
        st.info("没有找到相似的文本。")
        return
    
    st.subheader(f"相似文字内容检查结果 Top {len(similar_texts)}")

    #显示结果
    cols = st.columns(min(len(similar_texts), 3))
    for i,result in enumerate(similar_texts):
        with cols[i % 3]:
            text_id = result['id']


            image_db, _ = initialize_vector_db()

            if text_id in image_db:
                image_path = Path(image_db[text_id]['path'])
                if image_path.exists():
                    img = Image.open(image_path)
                    st.image(img, caption=f"相似度: {result['similarity']:.2f}", use_container_width=True)


                    if 'timestamp' in image_db[text_id]:
                        time_str = time.strftime("%Y-%m-%d %H:%M:%S", 
                                                 time.localtime(image_db[text_id]['timestamp']))
                        st.caption(f"上传时间: {time_str}")

                    
                    with st.expander("查看文字内容"):
                        st.write(result['text'])
                else:
                    st.error(f"图片文件不存在: {image_path}")
            else:
                st.warning(f"未找到ID为 {text_id} 的图片记录")
                st.write(result['text'])
                
    # 计算余弦相似度

# 添加一个清除处理状态的函数
def reset_processed_state():
    st.session_state.processed = False
    st.session_state.detection_results = None
    st.session_state.processed_img = None
    st.session_state.detected_faces = []
    st.session_state.similar_images = []
    st.session_state.similar_faces_results = []
    st.session_state.advanced_face_search_done = False
    st.session_state.face_best_matches = []
    st.session_state.current_tolerance = 0.6
    st.session_state.similar_texts = []

# 修改图片压缩函数
def compress_image(image, max_size_kb=500, quality=50, max_dimension=1500):
    """
    压缩图片使其小于指定大小
    
    参数:
    - image: PIL Image对象
    - max_size_kb: 最大文件大小(KB)
    - quality: 初始压缩质量
    - max_dimension: 最大尺寸限制
    
    返回:
    - 压缩后的PIL Image对象
    """
    # 首先限制图片尺寸，防止内存溢出
    width, height = image.size
    # 如果图片太大，先调整尺寸
    if max(width, height) > max_dimension:
        # 计算调整比例
        ratio = max_dimension / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        # 使用thumbnail节省内存
        image.thumbnail((new_width, new_height), Image.LANCZOS)
    
    # 确保图片是RGB模式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 用于存储压缩图像的字节流
    buffer = BytesIO()
    
    # 保存当前质量的图像
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    
    # 检查大小
    current_size = buffer.tell() / 1024  # KB
    
    # 如果大小已经小于限制，直接返回
    if current_size <= max_size_kb:
        buffer.seek(0)
        return Image.open(buffer)
    
    # 计算需要的压缩比
    compression_ratio = max_size_kb / current_size
    
    # 如果图像还是太大，进一步降低质量
    new_quality = int(quality * compression_ratio)
    new_quality = max(10, min(new_quality, 60))  # 限制质量范围
    
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=new_quality, optimize=True)
    current_size = buffer.tell() / 1024
    
    # 如果降低质量不够，需要进一步调整尺寸
    while current_size > max_size_kb and (width > 300 or height > 300):
        # 每次缩小到80%
        width = int(width * 0.8)
        height = int(height * 0.8)
        
        # 重新调整图像大小
        image = image.resize((width, height), Image.LANCZOS)
        
        # 保存并检查大小
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=new_quality, optimize=True)
        current_size = buffer.tell() / 1024
    
    # 将字节流转换回PIL图像
    buffer.seek(0)
    return Image.open(buffer)

# 添加批量处理函数
def process_batch_images(uploaded_files):
    """批量处理上传的图片"""
    with st.spinner(f"正在处理 {len(uploaded_files)} 张图片..."):
        # 清空之前的批量处理数据
        st.session_state.batch_images = []
        st.session_state.batch_faces = []
        st.session_state.batch_processed_images = []
        st.session_state.batch_image_vectors = {}
        st.session_state.batch_face_vectors = {}
        
        # 加载模型
        model = load_model(selected_model)
        if detect_faces:
            face_detector = load_face_detector()
        if enable_similarity_search or enable_face_similarity:
            feature_extractor = load_feature_extractor()
            transform = get_transform()
        
        # 处理每张图片
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # 读取图片
                image = Image.open(uploaded_file)
                
                # 获取文件大小并压缩大图片
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if file_size_mb > 2:
                    image = compress_image(image, max_size_kb=100)
                
                # 添加到批量图片列表
                st.session_state.batch_images.append(image)
                
                # 物体检测
                results = model.predict(image)
                
                # 获取选择的类别
                selected_classes = []
                if detect_person:
                    selected_classes.append("人")
                if detect_cup:
                    selected_classes.append("杯子/酒杯")
                if detect_bottle:
                    selected_classes.append("瓶子")
                if detect_all:
                    selected_classes.append("检测所有支持的物体")
                
                # 人脸检测
                batch_faces_for_image = []
                if detect_faces:
                    original_image_for_face = np.array(image)
                    _, detected_faces = detect_face(original_image_for_face, face_detector, face_confidence)
                    batch_faces_for_image = detected_faces
                    st.session_state.batch_faces.append(detected_faces)
                
                # 处理检测结果并标记
                processed_img = process_prediction(image, results, selected_classes, confidence)
                st.session_state.batch_processed_images.append(processed_img)
                
                # 提取图片特征向量
                if enable_similarity_search:
                    image_features = extract_image_features(image, feature_extractor, transform)
                    image_id = f"batch_image_{i}"
                    st.session_state.batch_image_vectors[image_id] = {
                        'vector': image_features,
                        'index': i
                    }
                
                # 提取人脸特征向量
                if enable_face_similarity and batch_faces_for_image:
                    for j, face in enumerate(batch_faces_for_image):
                        face_pil = Image.fromarray(face)
                        face_features = extract_face_features(face_pil, feature_extractor, transform)
                        if face_features is not None:
                            face_id = f"batch_face_{i}_{j}"
                            st.session_state.batch_face_vectors[face_id] = {
                                'vector': face_features,
                                'image_index': i,
                                'face_index': j
                            }
            
            except Exception as e:
                st.error(f"处理图片 #{i+1} 时出错: {str(e)}")
        
        # 计算批量图片间的相似度
        if enable_similarity_search and len(st.session_state.batch_image_vectors) > 1:
            compute_batch_image_similarity()
        
        # 计算批量人脸间的相似度
        if enable_face_similarity and len(st.session_state.batch_face_vectors) > 1:
            compute_batch_face_similarity()
        
        return len(uploaded_files)

# 添加计算批量图片相似度的函数
def compute_batch_image_similarity():
    """计算批量上传图片间的相似度"""
    similarity_results = []
    
    # 获取所有图片ID和特征向量
    image_ids = list(st.session_state.batch_image_vectors.keys())
    
    # 计算每对图片间的相似度
    for i in range(len(image_ids)):
        for j in range(i+1, len(image_ids)):
            id1, id2 = image_ids[i], image_ids[j]
            vector1 = st.session_state.batch_image_vectors[id1]['vector']
            vector2 = st.session_state.batch_image_vectors[id2]['vector']
            
            # 计算余弦相似度
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            
            # 添加到结果列表
            if similarity >= similarity_threshold:
                similarity_results.append({
                    'image1_id': id1,
                    'image2_id': id2,
                    'image1_index': st.session_state.batch_image_vectors[id1]['index'],
                    'image2_index': st.session_state.batch_image_vectors[id2]['index'],
                    'similarity': similarity
                })
    
    # 按相似度降序排序
    similarity_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 存储结果
    st.session_state.batch_similarity_results = similarity_results

# 添加计算批量人脸相似度的函数
def compute_batch_face_similarity():
    """计算批量上传图片中人脸间的相似度"""
    face_similarity_results = []
    
    # 获取所有人脸ID和特征向量
    face_ids = list(st.session_state.batch_face_vectors.keys())
    
    # 计算每对人脸间的相似度
    for i in range(len(face_ids)):
        for j in range(i+1, len(face_ids)):
            id1, id2 = face_ids[i], face_ids[j]
            vector1 = st.session_state.batch_face_vectors[id1]['vector']
            vector2 = st.session_state.batch_face_vectors[id2]['vector']
            
            # 计算余弦相似度
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            
            # 添加到结果列表
            if similarity >= similarity_threshold:
                face_similarity_results.append({
                    'face1_id': id1,
                    'face2_id': id2,
                    'image1_index': st.session_state.batch_face_vectors[id1]['image_index'],
                    'image2_index': st.session_state.batch_face_vectors[id2]['image_index'],
                    'face1_index': st.session_state.batch_face_vectors[id1]['face_index'],
                    'face2_index': st.session_state.batch_face_vectors[id2]['face_index'],
                    'similarity': similarity
                })
    
    # 按相似度降序排序
    face_similarity_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 存储结果
    st.session_state.batch_face_similarity_results = face_similarity_results

# 添加显示批量图片相似结果的函数
def display_batch_similarity_results():
    """显示批量图片之间的相似度结果"""
    if not st.session_state.batch_similarity_results:
        st.info("未找到相似图片")
        return
    
    st.subheader(f"批量图片相似度结果 (Top {min(5, len(st.session_state.batch_similarity_results))})")
    
    # 显示前5个相似结果
    for i, result in enumerate(st.session_state.batch_similarity_results[:5]):
        cols = st.columns(3)
        
        # 显示第一张图片
        with cols[0]:
            img1_index = result['image1_index']
            st.image(st.session_state.batch_images[img1_index], caption=f"图片 #{img1_index+1}", use_container_width=True)
        
        # 显示相似度
        with cols[1]:
            st.write("")
            st.write("")
            st.write("")
            st.markdown(f"### 相似度: {result['similarity']:.2f}")
            st.markdown("→")
        
        # 显示第二张图片
        with cols[2]:
            img2_index = result['image2_index']
            st.image(st.session_state.batch_images[img2_index], caption=f"图片 #{img2_index+1}", use_container_width=True)
        
        st.write("---")

# 添加显示批量人脸相似结果的函数
def display_batch_face_similarity_results():
    """显示批量图片中人脸之间的相似度结果"""
    if not st.session_state.batch_face_similarity_results:
        st.info("未找到相似人脸")
        return
    
    st.subheader(f"批量人脸相似度结果 (Top {min(5, len(st.session_state.batch_face_similarity_results))})")
    
    # 显示前5个相似结果
    for i, result in enumerate(st.session_state.batch_face_similarity_results[:5]):
        cols = st.columns(3)
        
        # 显示第一个人脸
        with cols[0]:
            img1_index = result['image1_index']
            face1_index = result['face1_index']
            face1 = st.session_state.batch_faces[img1_index][face1_index]
            st.image(face1, caption=f"图片 #{img1_index+1} 中的人脸 #{face1_index+1}", use_container_width=True)
        
        # 显示相似度
        with cols[1]:
            st.write("")
            st.write("")
            st.write("")
            st.markdown(f"### 相似度: {result['similarity']:.2f}")
            st.markdown("→")
        
        # 显示第二个人脸
        with cols[2]:
            img2_index = result['image2_index']
            face2_index = result['face2_index']
            face2 = st.session_state.batch_faces[img2_index][face2_index]
            st.image(face2, caption=f"图片 #{img2_index+1} 中的人脸 #{face2_index+1}", use_container_width=True)
        
        st.write("---")


# ... existing code ...
def summarize_batch_images(similarity_threshold=0.5):
    """
    根据批量上传图片的相似性总结场景数量
    """
    try:
        if 'batch_images' not in st.session_state or len(st.session_state.batch_images) == 0:
            return "未检测到上传的图片"
        
        # 检查batch_image_vectors是否包含特征向量
        if 'batch_image_vectors' not in st.session_state or not st.session_state.batch_image_vectors:
            return "未能提取图片特征"
        
        # 获取图片特征向量
        image_features = []
        for image_id, data in st.session_state.batch_image_vectors.items():
            if 'vector' in data and data['vector'] is not None:
                image_features.append(data['vector'])
        
        if not image_features:
            return "未能提取图片特征"
        
        # 使用层次聚类分析相似图片
        if len(image_features) > 1:
            features_array = np.vstack(image_features)
            #cats = np.concatenate([image_features], axis=1)
            
            # 计算余弦距离矩阵
            distance_matrix = 1 - cosine_similarity(features_array)
            
            # 使用层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=1 - similarity_threshold,  # 转换为距离阈值
                metric='precomputed',
                linkage='average'
            )
            clusters = clustering.fit_predict(distance_matrix)
            
            # 计算场景数量
            unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # 排除噪声点
            
            return f"分析结果：您上传的{len(image_features)}张图片中，大约包含{unique_clusters}个不同场景。"
        else:
            return "图片数量不足，无法进行场景聚类"
    except Exception as e:
        return f"场景总结出错: {str(e)}"

def summarize_batch_faces(similarity_threshold=0.5):
    """
    根据批量上传图片中的人脸总结不同人物数量
    """
    try:
        if 'batch_faces' not in st.session_state or len(st.session_state.batch_faces) == 0:
            return "未检测到人脸"
        
        # 创建临时目录存储人脸
        temp_input_dir = os.path.join(tempfile.gettempdir(), "face_clustering_input")
        temp_output_dir = os.path.join(tempfile.gettempdir(), "face_clustering_output")
        
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            
        os.makedirs(temp_input_dir)
        
        # 提取并保存所有人脸
        face_count = 0
        for i, faces in enumerate(st.session_state.batch_faces):
            if isinstance(faces, list):
                for j, face_np in enumerate(faces):
                    if isinstance(face_np, np.ndarray) and face_np.size > 0:
                        # 将numpy数组转换为PIL Image
                        face_img = Image.fromarray(face_np)
                        face_path = os.path.join(temp_input_dir, f"face_{i}_{j}.jpg")
                        face_img.save(face_path)
                        face_count += 1
        
        if face_count == 0:
            return "未在上传的图片中检测到人脸"
        
        # 使用face_clustering函数进行人脸聚类
        face_clustering(temp_input_dir, temp_output_dir, 
                        detection_method="hog",
                        tolerance=23,
                        min_cluster_size=1,
                        dim_reduction=False, n_components=3)  # 设置min_cluster_size为1以包含所有人脸
        
        # 计算聚类结果
        cluster_dirs = [d for d in os.listdir(temp_output_dir) 
                       if os.path.isdir(os.path.join(temp_output_dir, d)) and d.startswith("person_")]
        
        # 注意：此处不再删除临时目录，以便后续生成蒙太奇图
        # 返回临时目录路径和分析结果
        return f"分析结果：您上传的图片中，共检测到{face_count}个人脸，大约包含{len(cluster_dirs)}个不同人物。"
    except Exception as e:
        return f"人脸总结出错: {str(e)}"

def display_batch_analysis_summary():
    """
    显示批量图片分析的总结
    """
    if 'batch_images' in st.session_state and len(st.session_state.batch_images) > 0:
        st.subheader("批量分析总结")
        
        col1, col2 = st.columns(2)
        with col1:
            # similarity_threshold = st.slider("相似度阈值", 0.1, 0.9, 0.5, 0.1, 
            #                                 help="较低的值会将更多图片归为同一场景/同一人物")
            similarity_threshold = 0.5
        
        scene_summary = summarize_batch_images(similarity_threshold)
        face_summary = summarize_batch_faces(similarity_threshold)
        
        st.info(scene_summary)
        st.info(face_summary)

        # 添加人脸聚类蒙太奇图显示
        st.subheader("人脸聚类结果")
        try:
            # 使用与summarize_batch_faces相同的临时目录
            temp_output_dir = os.path.join(tempfile.gettempdir(), "face_clustering_output")
            
            if os.path.exists(temp_output_dir):
                # 生成蒙太奇图
                montages_info = create_montages(temp_output_dir, montage_size=(150, 150), images_per_row=5)
                
                if montages_info:
                    for montage in montages_info:
                        st.image(montage["path"], 
                                caption=f"聚类 {montage['cluster']} ({montage['images_count']} 张图片)",
                                use_container_width=True)
                else:
                    st.info("未生成人脸聚类蒙太奇图")
            else:
                st.info("未找到人脸聚类结果")
        except Exception as e:
            st.error(f"显示人脸聚类蒙太奇图出错: {str(e)}")
            
        # 在完成显示后才清理临时目录
        try:
            temp_input_dir = os.path.join(tempfile.gettempdir(), "face_clustering_input")
            temp_output_dir = os.path.join(tempfile.gettempdir(), "face_clustering_output")
            
            if os.path.exists(temp_input_dir):
                shutil.rmtree(temp_input_dir)
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
        except Exception as e:
            st.error(f"清理临时目录出错: {str(e)}")
# ... existing code ...

# 主程序
try:
    # 获取选择的模型名称
    selected_model = model_mapping[model_option]
    model = load_model(selected_model)
    
    # 加载人脸检测器
    if detect_faces:
        face_detector = load_face_detector()
    
    # 加载特征提取模型
    if enable_similarity_search or enable_face_similarity:
        feature_extractor = load_feature_extractor()
        transform = get_transform()
    
    # 显示模型信息
    st.info(f"当前使用: {model_option}")
    
    # 检查是否启用批量模式
    if st.session_state.batch_mode:
        # 批量上传图片
        uploaded_files = st.file_uploader("上传多张图片进行批量处理", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            # 显示上传的图片数量
            st.write(f"已上传 {len(uploaded_files)} 张图片")
            
            # 添加批量处理按钮
            if st.button(f"处理 {len(uploaded_files)} 张图片"):
                # 执行批量处理
                processed_count = process_batch_images(uploaded_files)
                st.success(f"成功处理 {processed_count} 张图片")
                
                # 显示处理结果概览
                st.subheader("批量处理结果概览")
                
                # 创建一个表格显示每张图片的基本信息
                data = []
                for i, (image, processed_img) in enumerate(zip(st.session_state.batch_images, st.session_state.batch_processed_images)):
                    # 检查是否有人脸
                    face_count = len(st.session_state.batch_faces[i]) if i < len(st.session_state.batch_faces) else 0
                    
                    # 添加行
                    data.append({
                        "图片序号": i+1,
                        "尺寸": f"{image.width}x{image.height}",
                        "检测到的人脸数": face_count
                    })
                
                # 显示表格
                st.dataframe(data)
                
                # 创建标签页显示详细结果
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["处理后的图片", "检测到的人脸", "图片相似度", "人脸相似度", "批量分析总结"])
                
                with tab1:
                    # 显示处理后的图片
                    st.subheader("处理后的图片")
                    cols = st.columns(3)
                    for i, processed_img in enumerate(st.session_state.batch_processed_images):
                        with cols[i % 3]:
                            st.image(processed_img, caption=f"图片 #{i+1}", use_container_width=True)
                
                with tab2:
                    # 显示检测到的人脸
                    st.subheader("检测到的人脸")
                    
                    # 遍历每张图片
                    for i, faces in enumerate(st.session_state.batch_faces):
                        if faces:
                            st.write(f"#### 图片 #{i+1} 中的人脸:")
                            cols = st.columns(min(len(faces), 4))
                            for j, face in enumerate(faces):
                                with cols[j % 4]:
                                    st.image(face, caption=f"人脸 #{j+1}", use_container_width=True)
                        else:
                            st.write(f"图片 #{i+1} 中未检测到人脸")
                
                with tab3:
                    # 显示图片相似度结果
                    display_batch_similarity_results()
                
                with tab4:
                    # 显示人脸相似度结果
                    display_batch_face_similarity_results()

                with tab5:
                    # 显示批量分析总结
                    display_batch_analysis_summary()

    else:
        # 原有的单张图片上传逻辑
        uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"], on_change=reset_processed_state)
        
        if uploaded_file is not None:
            # 为上传文件生成唯一ID
            current_upload_id = id(uploaded_file)
            
            # 显示原始图片
            col1, col2 = st.columns(2)
            
            # 如果是新上传的文件或者尚未处理
            if current_upload_id != st.session_state.last_upload_id or not st.session_state.processed:
                try:
                    # 获取文件大小
                    uploaded_file.seek(0)
                    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)  # 转换为MB
                    
                    # 读取上传的图片并立即压缩
                    image = Image.open(uploaded_file)
                    
                    # 始终压缩图片以防止内存问题，对于大图片显示提示
                    if file_size_mb > 2:
                        st.warning(f"上传的图片大小为 {file_size_mb:.2f}MB，超过2MB，将进行压缩。")
                    
                    # 压缩图片，限制在500KB以内
                    original_width, original_height = image.size
                    image = compress_image(image, max_size_kb=100)
                    
                    # 获取压缩后的大小和尺寸
                    buffer = BytesIO()
                    image.save(buffer, format='JPEG', optimize=True)
                    compressed_size_kb = buffer.tell() / 1024
                    current_width, current_height = image.size
                    
                    if file_size_mb > 2:
                        st.success(f"压缩完成！原始尺寸: {original_width}x{original_height}，现在: {current_width}x{current_height}，大小: {compressed_size_kb:.2f}KB")
                    
                    st.session_state.image = image
                    st.session_state.last_upload_id = current_upload_id
                    
                    with col1:
                        st.subheader("原始图片")
                        st.image(image, use_container_width=True)
                    
                    # 进行物体检测
                    with st.spinner(f"正在使用 {model_option} 进行物体检测..."):
                        results = model.predict(image)
                        st.session_state.detection_results = results
                        
                        # 获取选择的类别
                        selected_classes = []
                        if detect_person:
                            selected_classes.append("人")
                        if detect_cup:
                            selected_classes.append("杯子/酒杯")
                        if detect_bottle:
                            selected_classes.append("瓶子")
                        if detect_all:
                            selected_classes.append("检测所有支持的物体")
                        
                        # 先执行人脸检测（在原始图像上）
                        detected_faces = []
                        original_image_for_face = np.array(image)
                        if detect_faces:
                            with st.spinner("正在进行人脸检测..."):
                                _, detected_faces = detect_face(original_image_for_face, face_detector, face_confidence)
                                st.session_state.detected_faces = detected_faces
                        
                        # 处理结果并标记（先物体检测）
                        processed_img = process_prediction(image, results, selected_classes, confidence)
                        st.session_state.processed_img = processed_img
                        
                        # 如果启用了人脸检测，在物体检测结果上添加人脸标记
                        if detect_faces:
                            # 解析人脸边框颜色
                            hex_color = face_box_color.lstrip('#')
                            face_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # 转换为BGR
                            
                            # 将处理过的图像转换为OpenCV格式
                            img_cv = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2BGR)
                            
                            # 使用MTCNN检测人脸
                            faces = face_detector.detect_faces(original_image_for_face)
                            
                            # 筛选置信度高于阈值的人脸
                            faces = [face for face in faces if face['confidence'] >= face_confidence]
                            
                            # 在已处理的图像上添加人脸边框
                            for face in faces:
                                # 获取边界框坐标
                                x, y, w, h = face['box']
                                
                                # 绘制人脸边框
                                cv2.rectangle(img_cv, (x, y), (x+w, y+h), face_bgr, 2)
                                
                                # 添加标签和置信度
                                label = f"Face: {face['confidence']:.2f}"
                                cv2.putText(img_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_bgr, 2)
                                
                                # 可选：绘制关键点
                                keypoints = face['keypoints']
                                for point in keypoints.values():
                                    cv2.circle(img_cv, point, 2, face_bgr, 2)
                            
                            # 转换回RGB
                            processed_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) 
                        
                        # 将最终结果保存到session_state
                        st.session_state.processed_img = processed_img
                        
                        # 执行相似图片搜索
                        if enable_similarity_search or enable_face_similarity:
                            with st.spinner("正在添加图片到向量库并执行相似搜索..."):
                                # 提取特征并更新向量库
                                image_id, face_ids = update_vector_db(image, detected_faces, feature_extractor, transform)
                                
                                # 初始化向量数据库（获取最新的数据）
                                image_db, face_db = initialize_vector_db()
                                
                                # 获取当前图片的特征向量
                                current_image_vector = image_db[image_id]['vector']
                                
                                # 执行相似图片搜索
                                if enable_similarity_search:
                                    # 排除当前图片
                                    search_image_db = {k: v for k, v in image_db.items() if k != image_id}
                                    similar_images = search_similar_images(
                                        current_image_vector, 
                                        search_image_db, 
                                        top_k=top_k, 
                                        threshold=similarity_threshold
                                    )
                                    
                                    # 保存搜索结果到session_state
                                    st.session_state.similar_images = similar_images
                                
                                # 执行人脸相似搜索
                                if enable_face_similarity and detected_faces:
                                    # 获取当前图片检测到的人脸特征向量
                                    current_face_vectors = []
                                    for face_id in face_ids:
                                        if face_id in face_db:
                                            current_face_vectors.append(face_db[face_id]['vector'])
                                    
                                    # 排除当前图片的人脸
                                    search_face_db = {k: v for k, v in face_db.items() if k not in face_ids}
                                    
                                    # 对每个人脸执行相似搜索
                                    similar_faces_results = search_similar_faces(
                                        current_face_vectors, 
                                        search_face_db, 
                                        top_k=top_k, 
                                        threshold=similarity_threshold
                                    )
                                    
                                    # 保存搜索结果到session_state
                                    st.session_state.similar_faces_results = similar_faces_results
                        
                        # 标记为已处理
                        st.session_state.processed = True
                        # 重置加强人脸检索状态
                        st.session_state.advanced_face_search_done = False
                except Exception as e:
                    st.error(f"处理图片时出错: {str(e)}")
                    st.info("请尝试上传较小的图片或降低图片分辨率")
            else:
                # 使用之前处理过的结果
                image = st.session_state.image
                with col1:
                    st.subheader("原始图片")
                    st.image(image, use_container_width=True)
            
            # 显示处理后的图片
            with col2:
                st.subheader("检测结果")
                st.image(st.session_state.processed_img, use_container_width=True)
            
            # 显示相似图片搜索结果
            if enable_similarity_search and st.session_state.similar_images:
                display_similar_images(st.session_state.similar_images)
            
            # 显示相似人脸结果
            if enable_face_similarity and st.session_state.similar_faces_results:
                display_similar_faces(st.session_state.similar_faces_results, st.session_state.detected_faces)
            
            # 显示检测到的人脸
            if detect_faces and st.session_state.detected_faces:
                st.subheader(f"检测到的人脸与情绪分析 ({len(st.session_state.detected_faces)})")
                face_cols = st.columns(min(len(st.session_state.detected_faces), 4))
                
                # 情绪对应的表情符号
                emotion_emojis = {
                    "angry": "😠",
                    "disgust": "🤢",
                    "fear": "😨",
                    "happy": "😊",
                    "sad": "😢",
                    "surprise": "😲",
                    "neutral": "😐"
                }
                
                for i, face in enumerate(st.session_state.detected_faces):
                    with face_cols[i % 4]:
                        try:
                            # 分析人脸情绪
                            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                            emotion = result[0]['dominant_emotion']
                            emotion_emoji = emotion_emojis.get(emotion, "")
                            
                            # 显示人脸图像和情绪
                            st.image(face, caption=f"人脸 #{i+1}")
                            st.write(f"主要情绪: {emotion} {emotion_emoji}")
                            
                            # # 显示情绪分数
                            # emotions = result[0]['emotion']
                            # # 创建情绪分数条形图
                            # emotions_values = {k: v for k, v in emotions.items()}
                            # max_emotion = max(emotions_values, key=emotions_values.get)
                            # for emo, score in emotions_values.items():
                            #     emoji_icon = emotion_emojis.get(emo, "")
                            #     st.write(f"{emoji_icon} {emo}: {score:.2f}%")
                        except Exception as e:
                            st.error(f"分析人脸 #{i+1} 时出错: {str(e)}")
                            st.image(face, caption=f"人脸 #{i+1} (情绪分析失败)")
            elif detect_faces and not st.session_state.detected_faces:
                st.info("未检测到人脸")
            
            # 添加下载按钮
            buf = io.BytesIO()
            pil_img = Image.fromarray(st.session_state.processed_img)
            pil_img.save(buf, format="PNG")
            st.download_button(
                label="下载检测结果",
                data=buf.getvalue(),
                file_name="detected_image.png",
                mime="image/png",
            )
            
            # 显示检测统计信息
            if results[0].boxes is not None:
                # 获取所有检测结果的类别
                all_classes = results[0].boxes.cls.cpu().numpy()
                all_confidences = results[0].boxes.conf.cpu().numpy()
                
                # 统计检测到的物体数量
                st.subheader("检测统计")
                class_names = results[0].names
                
                # 创建一个字典，用于存储各类别的计数
                class_counts = {}
                
                for cls, conf in zip(all_classes, all_confidences):
                    cls_id = int(cls)
                    # 如果类别在选中的类别中且置信度高于阈值
                    if ((cls_id == 0 and "人" in selected_classes) or
                        (cls_id == 41 and "杯子/酒杯" in selected_classes) or
                        (cls_id == 39 and "瓶子" in selected_classes) or
                        detect_all) and conf >= confidence:
                        
                        cls_name = class_names[cls_id]
                        if cls_name in class_counts:
                            class_counts[cls_name] += 1
                        else:
                            class_counts[cls_name] = 1
                
                # 显示统计结果
                for cls_name, count in class_counts.items():
                    st.write(f"- 检测到 {count} 个 {cls_name}")
                
                # 添加人脸计数到统计结果
                if detect_faces and st.session_state.detected_faces:
                    st.write(f"- 检测到 {len(st.session_state.detected_faces)} 个人脸")
                
                if not class_counts and not (detect_faces and st.session_state.detected_faces):
                    st.write("未检测到任何物体")

                # 添加AI分析部分
                st.subheader("AI综合分析")
                
                with st.spinner("正在进行AI分析..."):
                    # 获取API keys
                    groq_api_key = os.environ.get("GROQ_API_KEY")
                    openai_api_key = os.environ.get("GEMINI_API_KEY")
                    
                    if groq_api_key and openai_api_key:
                        # 检查图片尺寸
                        image_dimensions = check_image_dimensions(image)
                        
                        # 执行Groq分析
                        groq_analysis = analyze_with_groq(image, groq_api_key)
                        print(groq_analysis)
                        print("--------------------------------")

                        # 准备CV检测结果摘要
                        cv_summary = {
                            "detected_objects": class_counts,
                            "face_count": len(st.session_state.detected_faces) if detect_faces else 0
                        }
                        print(cv_summary)
                        
                        # 执行OpenAI总结，加入图片尺寸信息
                        final_summary = summarize_with_openai(cv_summary, groq_analysis, openai_api_key, image_dimensions)
                        
                        # 显示分析结果
                        st.markdown(final_summary)
                    else:
                        st.warning("请设置GROQ_API_KEY和OPENAI_API_KEY环境变量以启用AI分析功能")

                    
                    if groq_api_key:
                        with st.spinner("正在提取文字内容并创建embedding..."):
                            text_content = extract_text_from_analysis(groq_analysis)


                            print(f"提取的文本内容：'{text_content}'")
                            print(f"文本内容长度：{len(text_content) if text_content else 0}")
                            print(f"是否等于'无文字内容': {text_content == '无文字内容'}")


                            if text_content and text_content.strip() != '无文字内容':

                                update_text_vector_db(image_id, text_content)
                                print("成功更新文本向量数据库")


                                print("开始搜索相似文本")
                                similar_texts = search_similar_texts(
                                    image_id,
                                    top_k=top_k,
                                    threshold=0.8
                                )

                                st.session_state.similar_texts = similar_texts
                                print(f"找到 {len(similar_texts)} 个相似文本")


                                if st.session_state.similar_texts and len(st.session_state.similar_texts) > 0:
                                    display_similar_texts(st.session_state.similar_texts)
                            else:

                                print("无文字内容， 跳过文本向量更新和相似文本搜索")
                                st.session_state.similar_texts = []


                            with st.expander("图片中提取的文字内容"):
                                st.write(text_content)

            # 加强人脸检索部分，独立于YOLO和Groq分析
            if enable_advanced_face_search and st.session_state.detected_faces:
                st.subheader("加强人脸检索结果")


                if len(st.session_state.detected_faces) > 5:
                    st.warning("检测到的人脸数量超过5个，为提高性能已跳过加强人脸搜索，请选择包含更少人脸的图片以获得更好的检索效果")
                else:
                    # 当tolerance参数变化或尚未执行过人脸检索时才执行
                    if not st.session_state.advanced_face_search_done or st.session_state.current_tolerance != face_cluster_tolerance:
                        with st.spinner("正在查找离线聚类中的相似人脸..."):
                            face_best_matches = process_face_for_advanced_search(
                                st.session_state.detected_faces,
                                tolerance=face_cluster_tolerance
                            )
                            
                            # 更新session_state
                            st.session_state.face_best_matches = face_best_matches
                            st.session_state.advanced_face_search_done = True
                            st.session_state.current_tolerance = face_cluster_tolerance
                    else:
                        # 使用已计算的结果
                        face_best_matches = st.session_state.face_best_matches
                    
                    # 显示人脸匹配结果
                    if face_best_matches:
                        st.write(f"为 {len(face_best_matches)} 个检测到的人脸找到了相似聚类")
                        
                        # 显示每个人脸及其对应的最佳匹配
                        for i, match in enumerate(face_best_matches):
                            st.write(f"##### 人脸 #{match['face_idx'] + 1} 的最佳匹配:")
                            
                            cols = st.columns(2)
                            with cols[0]:
                                # 显示原始检测到的人脸
                                face_img = Image.fromarray(st.session_state.detected_faces[match['face_idx']])
                                st.image(face_img, caption=f"检测到的人脸 #{match['face_idx'] + 1}", width=150)
                            
                            with cols[1]:
                                # 显示匹配的聚类蒙太奇图
                                st.image(match["path"], 
                                        caption=f"聚类 {match['cluster']} (相似度: {match['similarity']:.2f}, {match['images_count']} 张图片)",
                                        use_container_width=True)
                            
                            st.write("---")  # 添加分隔线
                    else:
                        st.info("未在已有聚类中找到相似人脸")

except Exception as e:
    st.error(f"发生错误: {e}")
    if "未能下载预训练权重" in str(e) or "Failed to download" in str(e):
        st.warning(f"首次运行时需要下载YOLOv8模型，请确保您有稳定的网络连接。较大的模型下载可能需要更长时间。")
    
# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 使用方法：
    1. 上传一张包含要检测物体的图片
    2. 在侧边栏调整检测设置：
       - 选择模型（精度与速度的平衡）
       - 设置置信度阈值（值越高，误检测越少，但也可能漏掉一些物体）
       - 选择要检测的物体类别
       - 启用/禁用人脸检测
       - 设置相似图片和人脸搜索选项
       - 自定义标记颜色
    3. 查看检测结果、相似图片搜索结果，并下载标记后的图片
    
    ### 模型比较：
    - YOLOv8x: 最高精度，但处理速度较慢，适合对精度要求高的场景
    - YOLOv8l: 高精度，速度适中
    - YOLOv8m: 平衡的精度和速度
    - YOLOv8s: 较快的速度，精度适中
    - YOLOv8n: 最快的速度，但精度较低
    
    ### 相似图片和人脸搜索：
    - 系统会将每次上传的图片添加到向量数据库中
    - 相似图片搜索：查找与上传图片视觉特征相似的历史图片
    - 人脸相似搜索：对检测到的每个人脸，查找数据库中相似的人脸
    - 可调整相似度阈值和返回结果数量
    
    ### 支持检测的物体：
    - 人物
    - 杯子/酒杯
    - 瓶子
    - 人脸（使用OpenCV的人脸检测器）
    """)

# 添加页脚
st.markdown("---")
st.markdown("📸 高精度物体检测工具 | 基于YOLOv8和Streamlit构建")

