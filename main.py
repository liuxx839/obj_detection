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
from mtcnn import MTCNN # Use MTCNN for detection
from deepface import DeepFace # Use DeepFace for face analysis and representation
# Removed: import face_recognition
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
    # Default tolerance for DeepFace (cosine distance). Lower is stricter.
    st.session_state.current_tolerance = 0.4 # Adjusted default for cosine distance
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
        index=4  # 默认选择YOLOv11n (changed from YOLOv11x for speed)
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
    face_confidence = st.slider("人脸检测置信度", 0.1, 1.0, 0.8, 0.1) # MTCNN confidence

    # 相似搜索选项
    st.subheader("相似搜索")
    enable_similarity_search = st.checkbox("启用相似图片搜索", value=True)
    enable_face_similarity = st.checkbox("启用人脸相似搜索", value=True)
    similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.5, 0.05) # Cosine Similarity threshold
    top_k = st.slider("返回相似结果数量", 1, 10, 3)

    # 获取颜色设置
    st.subheader("标记颜色")
    box_color = st.color_picker("边框颜色", "#FF0000")
    face_box_color = st.color_picker("人脸边框颜色", "#00FF00")

    # 添加加强人脸检索选项
    st.subheader("加强人脸检索")
    enable_advanced_face_search = st.checkbox("启用加强人脸检索", value=True)
    if enable_advanced_face_search:
        # Adjusted tolerance for DeepFace/Cosine Distance
        face_cluster_tolerance = st.slider("人脸匹配严格度 (距离)", 0.1, 1.0, 0.4, 0.05, help="数值越低匹配越严格 (使用Cosine距离)")
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
    # Check if model file exists, download if not (YOLO handles this internally)
    try:
        model = YOLO(model_name)
        # Perform a dummy prediction to ensure model is loaded/downloaded
        # model.predict(np.zeros((640, 640, 3)), verbose=False)
        return model
    except Exception as e:
        st.error(f"加载YOLO模型 {model_name} 失败: {e}")
        st.stop()


# 加载人脸检测模型 (MTCNN)
@st.cache_resource
def load_face_detector():
    try:
        detector = MTCNN()
        # Optional: Perform a dummy detection to ensure model is loaded
        # detector.detect_faces(np.zeros((100, 100, 3)))
        return detector
    except Exception as e:
        st.error(f"加载MTCNN人脸检测模型失败: {e}")
        st.stop()

# 加载特征提取模型 (ResNet50 for general images, DeepFace for faces)
@st.cache_resource
def load_feature_extractor():
    # General image feature extractor (ResNet50)
    model = models.resnet50(weights='DEFAULT')
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    # Face feature extractor is handled by DeepFace directly
    return model

# 图像预处理转换 (For ResNet50)
@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 提取图像特征 (Using ResNet50)
def extract_image_features(image, model, transform):
    try:
        img_t = transform(image.convert("RGB")).unsqueeze(0) # Ensure RGB
        with torch.no_grad():
            features = model(img_t)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        st.warning(f"提取图像特征失败: {e}")
        return None

# 提取人脸特征 (Using DeepFace)
# Input can be PIL Image or numpy array
def extract_face_features(face_image_or_array, model_name="ArcFace"):
    try:
        # DeepFace.represent returns a list of dictionaries
        # [{'embedding': [..], 'facial_area': {'x':.., 'y':.., 'w':.., 'h':..}, 'face_confidence': ..}]
        result = DeepFace.represent(
            img_path=face_image_or_array,
            model_name=model_name,
            enforce_detection=False # Assume input is already a face crop
        )
        if result and isinstance(result, list) and 'embedding' in result[0]:
            return np.array(result[0]['embedding'])
        else:
            st.warning("DeepFace未能提取人脸特征")
            return None
    except ValueError as ve: # Handle case where DeepFace might not find a face even if cropped
         st.warning(f"DeepFace无法处理此人脸图像: {ve}")
         return None
    except Exception as e:
        st.warning(f"提取人脸特征失败 (DeepFace): {e}")
        return None


# 初始化向量库
def initialize_vector_db():
    vector_file = data_dir / "vectors" / "image_vectors.pkl"
    face_vector_file = data_dir / "vectors" / "face_vectors.pkl"

    # 图像向量数据库
    image_db = {}
    if os.path.exists(vector_file):
        try:
            with open(vector_file, 'rb') as f:
                image_db = pickle.load(f)
            if not isinstance(image_db, dict): # Basic validation
                st.warning("图像向量数据库文件格式错误，将重新创建。")
                image_db = {}
        except (pickle.UnpicklingError, EOFError, TypeError) as e:
             st.warning(f"读取图像向量数据库失败 ({e})，将重新创建。")
             image_db = {}


    # 人脸向量数据库
    face_db = {}
    if os.path.exists(face_vector_file):
        try:
            with open(face_vector_file, 'rb') as f:
                face_db = pickle.load(f)
            if not isinstance(face_db, dict): # Basic validation
                st.warning("人脸向量数据库文件格式错误，将重新创建。")
                face_db = {}
        except (pickle.UnpicklingError, EOFError, TypeError) as e:
            st.warning(f"读取人脸向量数据库失败 ({e})，将重新创建。")
            face_db = {}

    return image_db, face_db

# 保存向量数据库
def save_vector_db(image_db, face_db):
    vector_file = data_dir / "vectors" / "image_vectors.pkl"
    face_vector_file = data_dir / "vectors" / "face_vectors.pkl"

    try:
        with open(vector_file, 'wb') as f:
            pickle.dump(image_db, f)
    except Exception as e:
        st.error(f"保存图像向量数据库失败: {e}")

    try:
        with open(face_vector_file, 'wb') as f:
            pickle.dump(face_db, f)
    except Exception as e:
        st.error(f"保存人脸向量数据库失败: {e}")


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
    try:
        img_to_save = image.convert('RGB') # Ensure RGB before saving
        img_to_save.save(image_path, format="JPEG", quality=85) # Specify format and quality
    except Exception as e:
        st.error(f"保存原始图片失败: {e}")
        return None, []

    # 提取图像特征 (ResNet50)
    image_features = extract_image_features(image, feature_extractor, transform)
    if image_features is None:
        st.warning(f"未能为图片 {image_id} 提取特征，跳过图像相似性。")
    else:
        # 更新图像向量数据库
        image_db[image_id] = {
            'vector': image_features,
            'path': str(image_path),
            'timestamp': timestamp
        }

    # 处理人脸 (Using DeepFace for features)
    face_ids = []
    for i, face_array in enumerate(faces): # faces are numpy arrays from MTCNN/detect_face
        face_id = f"{image_id}_face_{i}"
        face_filename = f"{face_id}.jpg"
        face_path = data_dir / "faces" / face_filename

        try:
            face_pil = Image.fromarray(face_array).convert('RGB') # Ensure RGB
            face_pil.save(face_path, format="JPEG", quality=85)
        except Exception as e:
            st.error(f"保存人脸图片 {face_id} 失败: {e}")
            continue # Skip this face

        # 提取人脸特征 (DeepFace)
        face_features = extract_face_features(face_array) # Pass numpy array

        if face_features is not None:
            # 更新人脸向量数据库
            face_db[face_id] = {
                'vector': face_features,
                'path': str(face_path),
                'image_id': image_id,
                'timestamp': timestamp
            }
            face_ids.append(face_id)
        else:
            st.warning(f"未能为人脸 {face_id} 提取特征。")

    # 保存更新后的向量数据库
    save_vector_db(image_db, face_db)

    return image_id, face_ids


# 执行相似搜索 (Cosine Similarity)
def search_similar_images(query_vector, image_db, top_k=3, threshold=0.6):
    if not image_db or query_vector is None:
        return []

    results = []
    query_vector_norm = np.linalg.norm(query_vector)
    if query_vector_norm == 0: return [] # Avoid division by zero

    # 计算查询向量与数据库中所有向量的相似度
    for image_id, data in image_db.items():
        db_vector = data.get('vector')
        if db_vector is None or not isinstance(db_vector, np.ndarray): continue

        db_vector_norm = np.linalg.norm(db_vector)
        if db_vector_norm == 0: continue # Avoid division by zero

        # Cosine Similarity
        similarity = np.dot(query_vector, db_vector) / (query_vector_norm * db_vector_norm)
        similarity = float(similarity) # Ensure float type

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

# 执行人脸相似搜索 (Cosine Similarity using DeepFace embeddings)
def search_similar_faces(query_vectors, face_db, top_k=3, threshold=0.6):
    if not face_db or not query_vectors:
        return []

    all_results = []

    # 对每个查询人脸进行相似度计算
    for i, query_vector in enumerate(query_vectors):
        if query_vector is None: continue # Skip if query vector extraction failed

        results = []
        query_vector_norm = np.linalg.norm(query_vector)
        if query_vector_norm == 0: continue

        # 计算查询向量与数据库中所有人脸向量的相似度
        for face_id, data in face_db.items():
            db_vector = data.get('vector')
            if db_vector is None or not isinstance(db_vector, np.ndarray): continue

            db_vector_norm = np.linalg.norm(db_vector)
            if db_vector_norm == 0: continue

            # Cosine Similarity
            similarity = np.dot(query_vector, db_vector) / (query_vector_norm * db_vector_norm)
            similarity = float(similarity)

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
    img = np.array(image.convert("RGB")) # Ensure RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # COCO数据集类别 (assuming results[0].names exists)
    class_names = results[0].names if hasattr(results[0], 'names') else {} # Handle if names not available

    # 如果选择了"检测所有支持的物体"，将显示所有类别
    if "检测所有支持的物体" in selected_classes:
        selected_class_ids = list(class_names.keys()) if class_names else []
    else:
        # 创建类别名称到ID的映射 (COCO IDs)
        class_mapping = {
            "人": 0,  # person
            "杯子/酒杯": 41,  # cup (also covers wine glass often)
            "瓶子": 39,  # bottle
        }
        # 获取选中类别的ID
        selected_class_ids = [class_mapping[cls] for cls in selected_classes if cls in class_mapping]

    # 解析十六进制颜色为BGR
    try:
        hex_color = box_color.lstrip('#')
        box_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # 转换为BGR
    except ValueError:
        box_bgr = (0, 0, 255) # Default to red if color code is invalid

    # 如果有检测结果
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
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
                label = f"{class_names.get(cls_id, f'ID:{cls_id}')} {conf:.2f}" # Use get for safety
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                # Ensure label background doesn't go out of bounds
                label_bg_y1 = max(y1 - text_size[1] - 5, 0)
                label_bg_y2 = y1
                cv2.rectangle(img, (x1, label_bg_y1), (x1 + text_size[0], label_bg_y2), box_bgr, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 转换回RGB用于Streamlit显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 人脸检测函数 (Using MTCNN)
def detect_face(image, face_detector, conf_threshold):
    # 转换为numpy数组并确保RGB格式
    img_array = np.array(image.convert("RGB")) # Ensure RGB
    img_rgb = img_array.copy()

    # 使用MTCNN检测人脸
    # MTCNN returns list of dicts: {'box': [x, y, width, height], 'confidence': score, 'keypoints': {...}}
    try:
        faces_detected = face_detector.detect_faces(img_rgb)
    except Exception as e:
        st.error(f"MTCNN人脸检测失败: {e}")
        faces_detected = []

    # 筛选置信度高于阈值的人脸
    filtered_faces = [face for face in faces_detected if face['confidence'] >= conf_threshold]

    # 解析人脸边框颜色
    try:
        hex_color = face_box_color.lstrip('#')
        face_bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # 转换为BGR
    except ValueError:
        face_bgr = (0, 255, 0) # Default to green if color code is invalid

    # 提取的人脸图像列表 (as numpy arrays)
    face_image_arrays = []

    # 在原图上标记人脸
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # Work with BGR for drawing
    for face in filtered_faces:
        # 获取边界框坐标
        x, y, w, h = face['box']
        # Ensure coordinates are valid
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_cv.shape[1], x + w), min(img_cv.shape[0], y + h)

        # 绘制人脸边框
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), face_bgr, 2)

        # 添加标签和置信度
        label = f"Face: {face['confidence']:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_bg_y1 = max(y1 - text_size[1] - 5, 0)
        cv2.rectangle(img_cv, (x1, label_bg_y1), (x1 + text_size[0], y1), face_bgr, -1)
        cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 可选：绘制关键点 (if needed)
        # keypoints = face['keypoints']
        # for point in keypoints.values():
        #     cv2.circle(img_cv, tuple(point), 2, face_bgr, 2)

        # 提取人脸区域 (numpy array)
        if y2 > y1 and x2 > x1: # Ensure valid crop dimensions
            face_crop = img_rgb[y1:y2, x1:x2]
            face_image_arrays.append(face_crop)

    # 转换回RGB for display
    result_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    return result_img_rgb, face_image_arrays # Return marked image and list of face arrays

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
            try:
                image_path = Path(result['path'])
                if image_path.exists():
                    img = Image.open(image_path)
                    st.image(img, caption=f"相似度: {result['similarity']:.2f}", use_container_width=True)

                    # 显示时间戳（转换为可读格式）
                    if 'timestamp' in result:
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))
                        st.caption(f"上传时间: {time_str}")
                else:
                    st.warning(f"相似图片文件丢失: {image_path.name}")
            except Exception as e:
                st.error(f"显示相似图片时出错: {e}")

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

            # Check if query face index is valid
            if query_face_idx < len(detected_faces):
                query_face_array = detected_faces[query_face_idx]
                query_face_pil = Image.fromarray(query_face_array)

                # 创建列来显示查询人脸和匹配结果
                cols = st.columns(1 + min(len(matches), 3))

                # 显示查询人脸
                with cols[0]:
                    st.image(query_face_pil, caption="查询人脸", use_container_width=True)

                # 显示匹配结果
                for i, match in enumerate(matches):
                    if i < len(cols) - 1:  # 确保不超出列数
                        with cols[i + 1]:
                            try:
                                face_path = Path(match['path'])
                                if face_path.exists():
                                    match_face = Image.open(face_path)
                                    st.image(match_face, caption=f"相似度: {match['similarity']:.2f}", use_container_width=True)

                                    # 显示时间戳（转换为可读格式）
                                    if 'timestamp' in match:
                                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(match['timestamp']))
                                        st.caption(f"上传时间: {time_str}")
                                else:
                                     st.warning(f"相似人脸文件丢失: {face_path.name}")
                            except Exception as e:
                                st.error(f"显示相似人脸时出错: {e}")
            else:
                st.warning(f"查询人脸索引 #{query_face_idx + 1} 无效。")
        # Optionally add a message if a query face had no matches above threshold
        # else:
        #     st.write(f"查询人脸 #{query_face_idx + 1}: 未找到相似度高于阈值的匹配结果。")


# 修改图片转base64函数
def image_to_base64(image):
    # 确保图片是RGB模式
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')

    buffered = BytesIO()
    image.save(buffered, format="JPEG") # Use JPEG for potentially smaller size
    return base64.b64encode(buffered.getvalue()).decode()

# 修改analyze_with_groq函数
def analyze_with_groq(image, api_key):
    # Use Groq only if API key is provided
    if not api_key:
        return "Groq API Key 未设置"

    try:
        client_vision = Groq(api_key=api_key) # Pass key directly

        # Ensure image is PIL and RGB
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image) # Try converting if it's an array
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        base64_image = image_to_base64(image)

        response = client_vision.chat.completions.create(
            model="llama3-70b-8192", # Updated model
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
                        # LLaMA3 Vision input format differs from some others
                        {"type": "text", "text": "请分析这张图片"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ]
                }
            ],
            max_tokens=4096 # Increase max tokens if needed
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq分析出错: {str(e)}"


# 添加OpenAI/Gemini审查总结函数
def summarize_with_openai(cv_results, groq_analysis, api_key, image_dimensions=None):
    # Use Gemini only if API key is provided
    if not api_key:
        return "Gemini API Key 未设置"

    try:
        # Use google.generativeai endpoint for Gemini
        client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta") # Corrected base URL

        # 将图片尺寸信息添加到CV结果中
        cv_payload = cv_results.copy() # Avoid modifying original dict
        if image_dimensions:
            cv_payload["image_dimensions"] = image_dimensions

        # Ensure payload is JSON serializable (convert numpy types if any)
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        cv_payload_serializable = convert_types(cv_payload)


        response = client.chat.completions.create(
            model="models/gemini-1.5-flash-latest", # Use appropriate Gemini model
            messages=[
                {
                    "role": "user", # System role might not be directly supported, combine into user prompt
                    "parts": [
                        {
                            "text": f"""你是一位严谨的药厂审查关员。请根据以下CV检测结果和AI分析结果，生成一份简洁的合规性分析报告。
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
                            请以表格形式输出，表格需要包含审核项目，审核结果，审核原因三列，对于违规情况，请加粗字体，使用markdown格式。

                            CV检测结果：{cv_payload_serializable}

                            AI分析结果：{groq_analysis}
                            """
                        }
                    ]
                }
            ]
             # Add generation_config if needed, e.g., temperature=0.5
            # generation_config={"temperature": 0.5}
        )
        # Access Gemini response correctly
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             return response.choices[0].message.content
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            # Handle potential structured response from Gemini
            return "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
            return "OpenAI/Gemini返回了空的总结"
    except Exception as e:
        # Log the full error for debugging
        st.error(f"OpenAI/Gemini总结出错: {type(e).__name__} - {str(e)}")
        # You might want to print the full traceback in your server logs
        # import traceback
        # print(traceback.format_exc())
        return f"OpenAI/Gemini总结出错: {str(e)}"

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

# 添加人脸聚类函数 (Using MTCNN for detection, DeepFace for encoding)
def face_clustering(input_folder, output_folder, tolerance=0.4, min_cluster_size=3):
    """
    Cluster similar faces from a collection of images using MTCNN and DeepFace.

    Parameters:
    - input_folder: Path to folder containing images
    - output_folder: Path to output the clustered images
    - tolerance: Cosine distance threshold for clustering (lower is stricter, e.g., 0.4 corresponds to ~0.6 similarity)
    - min_cluster_size: Minimum number of images needed to form a cluster
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a directory for images where no faces were detected
    no_faces_dir = os.path.join(output_folder, "no_faces_detected")
    os.makedirs(no_faces_dir, exist_ok=True)

    # Create a directory for unclustered faces
    unclustered_dir = os.path.join(output_folder, "unclustered_faces")
    os.makedirs(unclustered_dir, exist_ok=True)

    print(f"[INFO] Processing images from {input_folder}")

    # Get all image files from the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'] # Added more extensions
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"[INFO] Found {len(image_paths)} images")

    # Initialize MTCNN detector
    try:
        detector = MTCNN()
    except Exception as e:
        st.error(f"Failed to initialize MTCNN in clustering: {e}")
        return output_folder # Return early

    # Initialize lists to store data
    known_encodings = [] # Stores DeepFace embeddings
    known_image_paths = [] # Stores the path of the original image for each embedding

    # Process each image
    for (i, image_path) in enumerate(image_paths):
        print(f"[INFO] Processing image {i+1}/{len(image_paths)}")

        try:
            # Load the image using OpenCV (handles various formats)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] Could not read image {image_path}")
                continue

            # Convert BGR to RGB (MTCNN expects RGB)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            faces = detector.detect_faces(rgb)

            # If no faces were found, copy to the no_faces directory
            if not faces:
                shutil.copy2(image_path, os.path.join(no_faces_dir, os.path.basename(image_path)))
                continue

            # Compute facial embeddings for each detected face
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
                # Add a small buffer/check confidence if needed
                # Example: if confidence < 0.9: continue

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(rgb.shape[1], x + w), min(rgb.shape[0], y + h)

                if x2 > x1 and y2 > y1: # Ensure valid crop
                    face_crop = rgb[y1:y2, x1:x2]
                    try:
                        # Get embedding using DeepFace (e.g., ArcFace model)
                        embedding_obj = DeepFace.represent(
                            img_path=face_crop,
                            model_name="ArcFace", # Consistent model choice
                            enforce_detection=False # Already detected
                        )
                        if embedding_obj and isinstance(embedding_obj, list):
                             embedding = embedding_obj[0]['embedding']
                             known_encodings.append(embedding)
                             known_image_paths.append(image_path) # Store path associated with this face
                        else:
                            print(f"[WARNING] DeepFace failed to get embedding for a face in {image_path}")

                    except ValueError as ve: # Handle cases where represent fails on crop
                         print(f"[WARNING] DeepFace ValueError for face in {image_path}: {ve}")
                    except Exception as e:
                         print(f"[ERROR] DeepFace represent error for face in {image_path}: {e}")

        except Exception as e:
            print(f"[ERROR] Error processing {image_path}: {e}")

    print(f"[INFO] Detected and encoded {len(known_encodings)} faces in total")

    # Cluster the faces using DBSCAN with Cosine Distance
    if len(known_encodings) >= min_cluster_size: # Need enough samples to cluster
        print("[INFO] Clustering faces...")

        # Convert encodings to numpy array
        data = np.array(known_encodings)

        # Perform DBSCAN clustering using cosine distance
        # eps is the maximum distance between samples for one to be considered as in the neighborhood of the other
        clt = DBSCAN(metric="cosine", eps=tolerance, min_samples=min_cluster_size, n_jobs=-1)
        clt.fit(data)

        # Get the unique cluster labels (-1 is for noise/outliers)
        label_ids = np.unique(clt.labels_)
        num_unique_faces = len(np.where(label_ids > -1)[0])

        print(f"[INFO] Found {num_unique_faces} unique face clusters (excluding noise)")

        # Process each cluster label
        for label_id in label_ids:
            # Get indices of faces belonging to this label
            indices = np.where(clt.labels_ == label_id)[0]

            # Determine output directory based on label
            if label_id == -1:
                current_cluster_dir = unclustered_dir
                print(f"[INFO] Copying {len(indices)} unclustered faces...")
            else:
                current_cluster_dir = os.path.join(output_folder, f"person_{label_id}")
                os.makedirs(current_cluster_dir, exist_ok=True)
                print(f"[INFO] Copying {len(indices)} faces to Cluster {label_id}...")

            # Copy images containing these faces to the corresponding directory, avoiding duplicates
            processed_hashes = set()
            unique_image_count = 0
            for idx in indices:
                image_path = known_image_paths[idx]
                image_hash = get_image_hash(image_path)

                # If image hash is valid and not already copied to this cluster dir
                if image_hash and image_hash not in processed_hashes:
                    target_path = os.path.join(current_cluster_dir, os.path.basename(image_path))
                    # Check if file already exists (might happen with multiple faces from same image in same cluster)
                    if not os.path.exists(target_path):
                       shutil.copy2(image_path, target_path)
                    processed_hashes.add(image_hash)
                    unique_image_count +=1
                # elif image_hash:
                    # print(f"[DEBUG] Skipping duplicate image hash {image_hash} for cluster {label_id}")
                # else:
                    # print(f"[WARNING] Could not get hash for {image_path}, copying anyway if target doesn't exist.")
                    # target_path = os.path.join(current_cluster_dir, os.path.basename(image_path))
                    # if not os.path.exists(target_path):
                    #    shutil.copy2(image_path, target_path)

            if label_id != -1:
                 print(f"[INFO] Cluster {label_id}: Copied {unique_image_count} unique images.")
            else:
                 print(f"[INFO] Unclustered: Copied {unique_image_count} unique images.")
    else:
         print("[INFO] Not enough faces detected ({len(known_encodings)}) to perform clustering (min required: {min_cluster_size}). Check 'unclustered_faces' and 'no_faces_detected' folders.")


    print("[INFO] Face clustering completed!")
    return output_folder


# 添加创建蒙太奇图的函数
def create_montages(output_folder, montage_size=(150, 150), images_per_row=5):
    """
    Create montages for each cluster to visualize the results
    """
    montages_info = []

    # Get all actual cluster subdirectories
    cluster_dirs = [d for d in os.listdir(output_folder)
                    if os.path.isdir(os.path.join(output_folder, d)) and d.startswith("person_")]

    montages_dir = os.path.join(output_folder, "montages")
    os.makedirs(montages_dir, exist_ok=True)

    for cluster_dir_name in cluster_dirs:
        cluster_full_path = os.path.join(output_folder, cluster_dir_name)
        # Get all image files in this cluster
        image_paths = [os.path.join(cluster_full_path, f)
                      for f in os.listdir(cluster_full_path)
                      if os.path.isfile(os.path.join(cluster_full_path, f))]

        # Skip if no images in the cluster
        if not image_paths:
            continue

        # Deduplicate based on hash (already done during clustering copy, but good practice here too)
        unique_images_paths = []
        image_hashes = set()

        for image_path in image_paths:
            image_hash = get_image_hash(image_path)
            if image_hash and image_hash not in image_hashes:
                image_hashes.add(image_hash)
                unique_images_paths.append(image_path)
            elif not image_hash: # Handle hash failure case
                unique_images_paths.append(image_path) # Include if hash failed

        print(f"[INFO] Cluster {cluster_dir_name}: {len(image_paths)} files, {len(unique_images_paths)} unique for montage")

        # Load and resize images (using unique paths)
        images_for_montage = []
        max_images_for_montage = images_per_row * 10 # Limit images per montage for performance/size
        random.shuffle(unique_images_paths) # Show a variety if too many images

        for image_path in unique_images_paths[:max_images_for_montage]:
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, montage_size)
                    images_for_montage.append(image)
                else:
                    print(f"[WARNING] Failed to read image for montage: {image_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load/resize image {image_path} for montage: {e}")


        # Create montage
        if images_for_montage:
            # Calculate grid size dynamically
            num_images = len(images_for_montage)
            rows = (num_images + images_per_row - 1) // images_per_row
            montage = build_montages(images_for_montage, montage_size, (images_per_row, rows))[0]

            # Save the montage
            montage_filename = f"{cluster_dir_name}_montage.jpg"
            montage_path = os.path.join(montages_dir, montage_filename)
            try:
                cv2.imwrite(montage_path, montage)
                print(f"[INFO] Created montage: {montage_path} with {len(images_for_montage)} images")
                montages_info.append({
                    "path": montage_path,
                    "cluster": cluster_dir_name,
                    "images_count": len(images_for_montage) # Count actually used in montage
                })
            except Exception as e:
                 print(f"[ERROR] Failed to save montage {montage_path}: {e}")
        else:
            print(f"[INFO] No valid images found for montage in cluster {cluster_dir_name}")


    print("[INFO] Montage creation completed!")
    return montages_info

# 修改process_face_for_advanced_search函数，使用DeepFace比较
def process_face_for_advanced_search(detected_faces_arrays, tolerance=0.4):
    # Find the latest cluster results directory
    current_dir = Path(".") # Use current working directory where Streamlit runs
    try:
        cluster_parent_dirs = [d for d in current_dir.iterdir()
                               if d.is_dir() and d.name.startswith("clustered_faces_")]
        if not cluster_parent_dirs:
            st.warning("未在当前目录下找到人脸聚类数据(clustered_faces_*)，请先运行离线聚类工具。")
            return []

        # Sort by name (assuming timestamp is in name like clustered_faces_YYYYMMDD_HHMMSS)
        cluster_parent_dirs.sort(key=lambda x: x.name, reverse=True)
        latest_cluster_dir = cluster_parent_dirs[0]
    except Exception as e:
         st.error(f"查找聚类目录时出错: {e}")
         return []


    montages_dir = latest_cluster_dir / "montages"
    if not montages_dir.exists():
        st.warning(f"未找到蒙太奇图片目录: {montages_dir}")
        # We can still proceed by comparing against individual cluster images if montages are missing
        # return []

    # Get all person cluster directories within the latest results
    try:
        person_dirs = [d for d in latest_cluster_dir.iterdir()
                       if d.is_dir() and d.name.startswith("person_")]
        if not person_dirs:
            st.warning(f"在 {latest_cluster_dir} 中未找到有效的人脸聚类组 (person_*)")
            return []
    except Exception as e:
         st.error(f"查找person目录时出错: {e}")
         return []

    # Results list for each detected face
    face_best_matches = []
    model_name = "ArcFace" # Use consistent model

    # Process each detected face (which are numpy arrays)
    for face_idx, face_array in enumerate(detected_faces_arrays):
        if face_array is None or face_array.size == 0:
            continue

        try:
            # Get embedding for the current detected face
            current_face_embedding_obj = DeepFace.represent(
                img_path=face_array, # Pass numpy array directly
                model_name=model_name,
                enforce_detection=False
            )
            if not current_face_embedding_obj or not isinstance(current_face_embedding_obj, list):
                 print(f"无法为检测到的人脸 #{face_idx} 获取embedding")
                 continue
            current_face_embedding = current_face_embedding_obj[0]['embedding']
            current_face_embedding = np.array(current_face_embedding) # Ensure numpy array

        except Exception as e:
            print(f"处理检测到的人脸 #{face_idx} 时获取embedding出错: {e}")
            continue # Skip this face if embedding fails

        # Find the best matching cluster for this face
        best_match_info = None
        best_similarity_score = -1 # Initialize with -1 (cosine similarity is between -1 and 1)

        for person_dir_path in person_dirs:
            cluster_name = person_dir_path.name
            # Get image files within this person cluster directory
            try:
                 face_image_files = [f for f in person_dir_path.iterdir()
                                    if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            except Exception as e:
                 print(f"读取聚类目录 {cluster_name} 时出错: {e}")
                 continue

            if not face_image_files:
                continue

            # Sample some images from the cluster for comparison
            sample_size = min(5, len(face_image_files))
            sampled_image_paths = random.sample(face_image_files, sample_size)

            cluster_embeddings = []
            # Get embeddings for the sampled images
            for img_path in sampled_image_paths:
                try:
                    # Use DeepFace.represent on the image path
                    # Enforce detection to find the main face in the sample image
                    embedding_obj = DeepFace.represent(
                        img_path=str(img_path),
                        model_name=model_name,
                        enforce_detection=True # Find face in sample image
                    )
                    if embedding_obj and isinstance(embedding_obj, list):
                        cluster_embeddings.append(embedding_obj[0]['embedding'])
                except ValueError: # Handles case where DeepFace finds no face
                    # print(f"No face found by DeepFace in cluster sample: {img_path.name}")
                    pass
                except Exception as e:
                    print(f"获取聚类样本 {img_path.name} 的embedding时出错: {e}")

            if not cluster_embeddings:
                continue # No valid embeddings from this cluster's sample

            # Calculate cosine similarity between the detected face and cluster samples
            cluster_embeddings_array = np.array(cluster_embeddings)
            similarities = cosine_similarity(current_face_embedding.reshape(1, -1), cluster_embeddings_array)[0]

            # Find the max similarity within this cluster
            if similarities.size > 0:
                max_similarity_in_cluster = np.max(similarities)

                # Compare with overall best similarity found so far
                # Similarity threshold check: higher similarity is better
                # Convert distance tolerance to similarity threshold: similarity >= 1 - tolerance
                similarity_threshold = 1.0 - tolerance
                if max_similarity_in_cluster >= similarity_threshold and max_similarity_in_cluster > best_similarity_score:
                    best_similarity_score = max_similarity_in_cluster

                    # Find the corresponding montage path (if it exists)
                    montage_path = montages_dir / f"{cluster_name}_montage.jpg"

                    best_match_info = {
                        "path": str(montage_path) if montage_path.exists() else None, # Store path or None
                        "cluster": cluster_name,
                        "similarity": float(best_similarity_score),
                        "images_count": len(face_image_files), # Total images in this cluster
                        "face_idx": face_idx
                    }

        # Append the best match found for this face_idx (if any)
        if best_match_info:
            face_best_matches.append(best_match_info)

    return face_best_matches


def check_image_dimensions(image):
    try:
        width, height = image.size
        if height == 0: return None # Avoid division by zero

        # Standard Sizes (Approximate aspect ratios)
        standard_sizes_info = {
            (640, 480): "4:3", (800, 600): "4:3", (1024, 768): "4:3",
            (1280, 720): "16:9", (1920, 1080): "16:9", (2560, 1440): "16:9", (3840, 2160): "16:9",
            (1280, 960): "4:3", (2048, 1536): "4:3", # More 4:3
            (1280, 800): "16:10", (1440, 900): "16:10", (1680, 1050): "16:10", (1920, 1200): "16:10", # 16:10
            (1080, 1920): "9:16 (Vertical)", # Vertical
            (750, 1334): "~9:16 (iPhone)", # Common Mobile
            (1125, 2436): "~9:19.5 (iPhone X)",
            (1080, 1080): "1:1 (Square)", (2048, 2048): "1:1 (Square)" # Square
        }

        tolerance = 15 # Pixel tolerance for matching standard sizes

        is_standard = False
        standard_ratio_name = "Non-standard"
        for (std_w, std_h), ratio_name in standard_sizes_info.items():
            if (abs(width - std_w) <= tolerance and abs(height - std_h) <= tolerance) or \
               (abs(width - std_h) <= tolerance and abs(height - std_w) <= tolerance): # Check rotated
                is_standard = True
                standard_ratio_name = ratio_name
                break

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Determine common ratio name based on value
        ratio_thresholds = {
            (0.95, 1.05): "1:1 (Square)",
            (1.30, 1.40): "4:3",
            (1.45, 1.55): "3:2",
            (1.58, 1.68): "16:10", # Between 3:2 and 16:9
            (1.72, 1.82): "16:9",
            (0.70, 0.80): "3:4 (Vertical)", # Approx vertical 4:3
            (0.52, 0.62): "9:16 (Vertical)", # Approx vertical 16:9
        }

        calculated_ratio_name = "Other Ratio"
        for (low, high), name in ratio_thresholds.items():
            if low <= aspect_ratio <= high:
                calculated_ratio_name = name
                break

        # Refine name if it matched a standard size directly
        final_ratio_name = standard_ratio_name if is_standard else calculated_ratio_name

        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 4),
            "ratio_name": final_ratio_name,
            "is_standard_size": is_standard
        }
    except Exception as e:
        print(f"Error checking dimensions: {e}")
        return None

# ... existing code ...
def extract_text_from_analysis(analysis):
    # Use Gemini/OpenAI for extraction if key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Fallback or simple extraction if no API key
        try:
            # Basic keyword search in the analysis text
            lines = analysis.split('\n')
            text_info_line = next((line for line in lines if "# 文字信息：" in line), None)
            if text_info_line:
                extracted = text_info_line.split("# 文字信息：", 1)[1].strip()
                if extracted and extracted not in ["未提及", "无法判断", "无"]:
                    return extracted
            return "无文字内容" # Default if not found
        except Exception:
            return "提取文字信息失败 (Fallback)" # Fallback error

    try:
        client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta")
        response = client.chat.completions.create(
            model="models/gemini-1.5-flash-latest",
            messages=[
                {
                     "role": "user",
                     "parts": [
                         {
                              "text": f"""从以下AI分析结果中，仅提取 '# 文字信息：' 部分后面的具体内容。
                              如果该部分内容是 '未提及'、'无法判断' 或 '无'，则输出 '无文字内容'。
                              不要包含 '# 文字信息：' 本身，不要添加任何解释、注释或标题。

                              AI分析结果:
                              ---
                              {analysis}
                              ---

                              提取的文字内容:"""
                         }
                    ]
                }
            ],
             # Add generation_config if needed
            # generation_config={"temperature": 0.1, "stop_sequences": ["\n"]} # Example config
        )
        # Access Gemini response
        if response.choices and response.choices[0].message and response.choices[0].message.content:
             result = response.choices[0].message.content.strip()
             # Final check if the model still included boilerplate
             if result.startswith("提取的文字内容:"):
                 result = result.replace("提取的文字内容:", "").strip()
             if not result or result in ["未提及", "无法判断", "无"]:
                 return "无文字内容"
             return result
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            result = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
            if result.startswith("提取的文字内容:"):
                 result = result.replace("提取的文字内容:", "").strip()
            if not result or result in ["未提及", "无法判断", "无"]:
                 return "无文字内容"
            return result
        else:
            return "提取文字信息失败 (Gemini无响应)"
    except Exception as e:
        return f"提取文字信息失败 (Gemini): {str(e)}"

def initialize_text_vectordb():
    text_vector_file = data_dir / "vectors" / "text_vectors.pkl"
    text_db = {}
    if os.path.exists(text_vector_file):
        try:
            with open(text_vector_file, "rb") as f:
                text_db = pickle.load(f)
            if not isinstance(text_db, dict):
                 st.warning("文本向量数据库格式错误，将重新创建。")
                 text_db = {}
        except (pickle.UnpicklingError, EOFError, TypeError) as e:
            st.warning(f"读取文本向量数据库失败 ({e})，将重新创建。")
            text_db = {}
    return text_db

def save_text_vector_db(text_db):
    text_vector_file = data_dir / "vectors" / "text_vectors.pkl"
    try:
        with open(text_vector_file, "wb") as f:
            pickle.dump(text_db, f)
    except Exception as e:
        st.error(f"保存文本向量数据库失败: {e}")


def update_text_vector_db(image_id, text_content):
    if not text_content or text_content == "无文字内容":
        print("跳过文本向量更新：无有效文本内容。")
        return

    # 初始化文字向量数据库
    text_db = initialize_text_vectordb()

    try:
        # Lazy load sentence transformer
        from sentence_transformers import SentenceTransformer

        @st.cache_resource
        def load_sentence_transformer():
            try:
                # Specify cache folder within the data directory if possible
                cache_folder = str(data_dir / "st_cache")
                os.makedirs(cache_folder, exist_ok=True)
                print(f"Loading Sentence Transformer model (cache: {cache_folder})...")
                # You might need to pre-download the model if running in restricted envs
                model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder=cache_folder)
                print("Sentence Transformer model loaded.")
                return model
            except Exception as e:
                st.error(f"加载Sentence Transformer模型失败: {e}")
                print(f"Error loading Sentence Transformer: {e}")
                return None

        model = load_sentence_transformer()
        if model is None:
             st.error("无法加载文本编码模型，跳过文本向量更新。")
             return

        # Check if text is too long - adjust if needed, or chunk
        # max_seq_length = model.max_seq_length # Get model's max sequence length
        # if len(text_content) > max_seq_length * 4: # Heuristic: truncate very long text
        #     print(f"文本内容过长 ({len(text_content)} chars)，将截断。")
        #     text_content = text_content[:max_seq_length*4]

        print(f"为图片 {image_id} 生成文本向量...")
        text_vector = model.encode(text_content, show_progress_bar=False)
        print(f"文本向量生成完毕，维度: {text_vector.shape}")

        text_db[image_id] = {
            'vector': text_vector.astype(np.float32), # Store as float32 to save space
            'text': text_content
        }

        # 保存更新后的数据库
        save_text_vector_db(text_db)
        print(f"文本向量数据库已为 {image_id} 更新。")

    except ImportError:
        st.error("请安装 'sentence-transformers' 库以启用文本相似性搜索: pip install sentence-transformers")
        print("sentence-transformers not installed.")
    except Exception as e:
        st.error(f"更新文本向量数据库时出错: {str(e)}")
        print(f"Error updating text vector DB: {str(e)}")


def search_similar_texts(image_id, top_k=3, threshold=0.5):
    text_db = initialize_text_vectordb()

    if not text_db or image_id not in text_db:
        print(f"文本数据库为空或查询ID {image_id} 不存在。")
        return []

    query_entry = text_db.get(image_id)
    if not query_entry or 'vector' not in query_entry:
        print(f"查询ID {image_id} 的向量不存在。")
        return []

    query_vector = query_entry['vector']
    if query_vector is None: return []

    results = []
    query_vector_norm = np.linalg.norm(query_vector)
    if query_vector_norm == 0: return []

    print(f"搜索与图片 {image_id} 文本相似的内容...")
    for text_id, data in text_db.items():
        # Skip comparing the item with itself
        if text_id == image_id:
            continue

        db_vector = data.get('vector')
        if db_vector is None or not isinstance(db_vector, np.ndarray):
            continue

        db_vector_norm = np.linalg.norm(db_vector)
        if db_vector_norm == 0: continue

        try:
            # Calculate Cosine Similarity
            similarity = np.dot(query_vector, db_vector) / (query_vector_norm * db_vector_norm)
            similarity = float(similarity) # Ensure float

            if similarity >= threshold:
                results.append({
                    'id': text_id, # This ID links back to the image
                    'text': data['text'],
                    'similarity': similarity
                })
        except Exception as e:
            print(f"计算文本向量相似度时出错 (ID: {text_id}): {str(e)}")
            continue

    # Sort by similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)
    print(f"找到 {len(results)} 个相似文本 (阈值 > {threshold})。")

    return results[:top_k]


def display_similar_texts(similar_texts):
    if not similar_texts:
        st.info("没有找到相似的文本内容。")
        return

    st.subheader(f"相似文字内容检查结果 (Top {len(similar_texts)})")

    # Display results
    cols = st.columns(min(len(similar_texts), 3))
    image_db, _ = initialize_vector_db() # Load image DB to find associated images

    for i, result in enumerate(similar_texts):
        with cols[i % 3]:
            text_id = result['id'] # This is the image_id associated with the similar text

            # Try to find the corresponding image
            image_info = image_db.get(text_id)
            if image_info and 'path' in image_info:
                image_path = Path(image_info['path'])
                try:
                    if image_path.exists():
                        img = Image.open(image_path)
                        st.image(img, caption=f"相似度: {result['similarity']:.2f}", use_container_width=True)

                        if 'timestamp' in image_info:
                            time_str = time.strftime("%Y-%m-%d %H:%M:%S",
                                                     time.localtime(image_info['timestamp']))
                            st.caption(f"上传时间: {time_str}")

                        # Show the similar text in an expander
                        with st.expander("查看相似文字内容"):
                            st.write(result['text'])
                    else:
                        st.error(f"图片文件不存在: {image_path.name}")
                        # Still show the text even if image is missing
                        with st.expander("查看相似文字内容 (图片丢失)"):
                             st.write(result['text'])
                             st.caption(f"(来自图片 ID: {text_id})")

                except Exception as e:
                     st.error(f"显示相似文本图片时出错: {e}")
                     # Still show text on error
                     with st.expander("查看相似文字内容 (显示错误)"):
                         st.write(result['text'])
                         st.caption(f"(来自图片 ID: {text_id})")
            else:
                 # If image record not found, just show the text
                 st.warning(f"未找到ID为 {text_id} 的图片记录")
                 with st.expander("查看相似文字内容"):
                     st.write(result['text'])
                     st.caption(f"(来自图片 ID: {text_id}, 相似度: {result['similarity']:.2f})")


# 添加一个清除处理状态的函数
def reset_processed_state():
    keys_to_reset = [
        'processed', 'detection_results', 'processed_img', 'detected_faces',
        'similar_images', 'similar_faces_results', 'advanced_face_search_done',
        'face_best_matches', 'similar_texts'
        # Keep 'last_upload_id', 'image', 'current_tolerance' maybe?
        # Resetting 'image' means the preview disappears on re-run after upload.
        # Let's keep 'image' and 'last_upload_id' if an upload happened.
    ]
    for key in keys_to_reset:
        if key in st.session_state:
             # Reset to default values
             if key in ['detected_faces', 'similar_images', 'similar_faces_results', 'face_best_matches', 'similar_texts']:
                 st.session_state[key] = []
             elif key == 'processed' or key == 'advanced_face_search_done':
                 st.session_state[key] = False
             else:
                 st.session_state[key] = None
    print("Processed state reset.")


# 修改图片压缩函数
def compress_image(image, max_size_kb=500, quality=65, max_dimension=1500): # Increased default quality slightly
    """
    Compresses a PIL Image object to be below a target size, also resizing if needed.

    Args:
        image (PIL.Image.Image): Input image.
        max_size_kb (int): Target maximum size in kilobytes.
        quality (int): Initial JPEG quality (1-95).
        max_dimension (int): Maximum width or height allowed.

    Returns:
        PIL.Image.Image: Compressed (and potentially resized) image.
    """
    try:
        img = image.copy() # Work on a copy

        # 1. Resize if dimensions exceed max_dimension
        width, height = img.size
        if max(width, height) > max_dimension:
            ratio = max_dimension / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS) # Use LANCZOS

        # 2. Ensure RGB format (JPEG doesn't support alpha)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 3. Iteratively reduce quality to meet size target
        buffer = BytesIO()
        current_quality = quality
        img.save(buffer, format='JPEG', quality=current_quality, optimize=True)
        current_size_kb = buffer.tell() / 1024

        while current_size_kb > max_size_kb and current_quality > 10:
            # Reduce quality more aggressively if far from target
            reduction = 10 if current_size_kb > max_size_kb * 1.5 else 5
            current_quality -= reduction
            print(f"Reducing quality to {current_quality}...")
            buffer = BytesIO() # Reset buffer
            img.save(buffer, format='JPEG', quality=current_quality, optimize=True)
            current_size_kb = buffer.tell() / 1024

        # 4. If still too large after quality reduction, resize further (optional, maybe warn instead)
        # This part might degrade quality significantly. Consider just returning the lowest quality version.
        if current_size_kb > max_size_kb and current_quality <= 10:
             print(f"Warning: Image still exceeds size limit ({current_size_kb:.1f}KB) even at lowest quality (10). Further resizing might occur.")
             # Optional: Add further resizing loop here if absolutely necessary
             # while current_size_kb > max_size_kb and max(img.size) > 300: # Example condition
             #     width, height = img.size
             #     img = img.resize((int(width * 0.9), int(height * 0.9)), Image.Resampling.LANCZOS)
             #     buffer = BytesIO()
             #     img.save(buffer, format='JPEG', quality=10, optimize=True)
             #     current_size_kb = buffer.tell() / 1024


        buffer.seek(0)
        print(f"Final compressed size: {current_size_kb:.1f} KB, Quality: {current_quality}, Dimensions: {img.size}")
        return Image.open(buffer)

    except Exception as e:
        st.error(f"图片压缩失败: {e}")
        return image # Return original on failure


# 添加批量处理函数
def process_batch_images(uploaded_files):
    """批量处理上传的图片"""
    processed_count = 0
    failed_files = []

    with st.spinner(f"准备处理 {len(uploaded_files)} 张图片..."):
        # --- Pre-load models ---
        try:
            model = load_model(selected_model)
        except Exception as e:
            st.error(f"无法加载YOLO模型 {selected_model}，批量处理中止。错误：{e}")
            return 0, []

        face_detector = None
        if detect_faces:
            try:
                face_detector = load_face_detector()
            except Exception as e:
                st.error(f"无法加载人脸检测模型，将跳过人脸检测。错误：{e}")
                # Continue without face detection? Or stop? Let's continue for now.
                # return 0, [] # Uncomment to stop if face detector fails

        feature_extractor = None
        transform = None
        if enable_similarity_search or enable_face_similarity:
            try:
                feature_extractor = load_feature_extractor()
                transform = get_transform()
            except Exception as e:
                st.error(f"无法加载特征提取模型，将跳过相似性搜索。错误：{e}")
                # Continue without similarity? Let's continue.

        # --- Clear previous batch state ---
        st.session_state.batch_images = []
        st.session_state.batch_faces = [] # List of lists (one list of faces per image)
        st.session_state.batch_processed_images = []
        st.session_state.batch_image_vectors = {} # {image_id: {'vector': vec, 'index': i}}
        st.session_state.batch_face_vectors = {} # {face_id: {'vector': vec, 'image_index': i, 'face_index': j}}

        # Get selected object classes for detection
        selected_classes = []
        if detect_person: selected_classes.append("人")
        if detect_cup: selected_classes.append("杯子/酒杯")
        if detect_bottle: selected_classes.append("瓶子")
        if detect_all: selected_classes.append("检测所有支持的物体")

    # --- Process each image ---
    prog_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"处理图片 {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        try:
            # Read image
            image_pil = Image.open(uploaded_file)

            # Compress if large (adjust max size as needed)
            uploaded_file.seek(0) # Reset file pointer
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > 1.5: # Compress images > 1.5 MB
                print(f"Compressing large image: {uploaded_file.name} ({file_size_mb:.2f} MB)")
                image_pil = compress_image(image_pil, max_size_kb=800, max_dimension=1920) # Target 800KB, max 1920px

            # Store original (potentially compressed) PIL image
            st.session_state.batch_images.append(image_pil)

            # --- Object Detection (YOLO) ---
            try:
                results = model.predict(image_pil, verbose=False) # verbose=False for cleaner logs
            except Exception as e:
                 print(f"YOLO预测失败 for {uploaded_file.name}: {e}")
                 results = [None] # Handle prediction failure gracefully

            # --- Face Detection (MTCNN) ---
            detected_faces_list = [] # Faces for this specific image
            image_for_face_detection = image_pil # Use the (potentially compressed) PIL image
            if detect_faces and face_detector:
                try:
                    _, detected_faces_list = detect_face(image_for_face_detection, face_detector, face_confidence)
                except Exception as e:
                     print(f"人脸检测失败 for {uploaded_file.name}: {e}")
                     # detected_faces_list remains empty

            st.session_state.batch_faces.append(detected_faces_list) # Append list of faces for this image

            # --- Process and Draw Detections ---
            processed_img_array = np.array(image_pil.convert("RGB")) # Start with current image
            if results[0] is not None: # Check if YOLO prediction succeeded
                processed_img_array = process_prediction(image_pil, results, selected_classes, confidence)
            # Add face boxes onto the image that already has object boxes
            if detect_faces and face_detector and detected_faces_list: # Draw if faces were detected
                # Re-run detection on the original array to get boxes again (or store boxes from detect_face)
                img_rgb_for_drawing = np.array(image_for_face_detection.convert("RGB"))
                faces_data = face_detector.detect_faces(img_rgb_for_drawing)
                filtered_faces_data = [f for f in faces_data if f['confidence'] >= face_confidence]

                # Draw boxes on the 'processed_img_array' which has object detections
                img_bgr_for_drawing = cv2.cvtColor(processed_img_array, cv2.COLOR_RGB2BGR)
                hex_color = face_box_color.lstrip('#')
                face_bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                for face in filtered_faces_data:
                     x, y, w, h = face['box']
                     x1, y1, x2, y2 = max(0, x), max(0, y), min(img_bgr_for_drawing.shape[1], x+w), min(img_bgr_for_drawing.shape[0], y+h)
                     cv2.rectangle(img_bgr_for_drawing, (x1, y1), (x2, y2), face_bgr_color, 2)
                     # Add label if needed
                     # label = f"Face: {face['confidence']:.2f}"
                     # cv2.putText(img_bgr_for_drawing, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_bgr_color, 2)
                processed_img_array = cv2.cvtColor(img_bgr_for_drawing, cv2.COLOR_BGR2RGB)


            st.session_state.batch_processed_images.append(processed_img_array) # Store final marked image array

            # --- Feature Extraction ---
            image_id = f"batch_image_{i}" # Unique ID for this batch image

            # Image Features (ResNet50)
            if enable_similarity_search and feature_extractor and transform:
                img_features = extract_image_features(image_pil, feature_extractor, transform)
                if img_features is not None:
                    st.session_state.batch_image_vectors[image_id] = {
                        'vector': img_features,
                        'index': i
                    }

            # Face Features (DeepFace)
            if enable_face_similarity and detected_faces_list:
                for j, face_array in enumerate(detected_faces_list):
                    face_features = extract_face_features(face_array) # Pass numpy array
                    if face_features is not None:
                        face_id = f"batch_face_{i}_{j}"
                        st.session_state.batch_face_vectors[face_id] = {
                            'vector': face_features,
                            'image_index': i,
                            'face_index': j
                        }

            processed_count += 1

        except Exception as e:
            st.error(f"处理图片 {uploaded_file.name} 时发生意外错误: {str(e)}")
            failed_files.append(uploaded_file.name)
            # Add placeholders for failed images to keep lists aligned?
            st.session_state.batch_images.append(None) # Or a placeholder image
            st.session_state.batch_faces.append([])
            st.session_state.batch_processed_images.append(None) # Or a placeholder

        # Update progress bar
        prog_bar.progress((i + 1) / len(uploaded_files))

    status_text.text(f"批量处理完成。成功: {processed_count}, 失败: {len(failed_files)}.")
    if failed_files:
        st.warning(f"以下文件处理失败: {', '.join(failed_files)}")

    # --- Compute Similarities (after processing all images) ---
    with st.spinner("计算相似度..."):
        if enable_similarity_search and len(st.session_state.batch_image_vectors) > 1:
            compute_batch_image_similarity()

        if enable_face_similarity and len(st.session_state.batch_face_vectors) > 1:
            compute_batch_face_similarity()

    return processed_count, failed_files


# 添加计算批量图片相似度的函数
def compute_batch_image_similarity():
    """计算批量上传图片间的相似度"""
    similarity_results = []
    image_vectors_dict = st.session_state.batch_image_vectors

    image_ids = list(image_vectors_dict.keys())
    vectors = [image_vectors_dict[id]['vector'] for id in image_ids]
    indices = [image_vectors_dict[id]['index'] for id in image_ids]

    if not vectors: return # No vectors to compare

    # Calculate cosine similarity matrix
    try:
        # Ensure vectors are valid numpy arrays
        valid_vectors = [v for v in vectors if isinstance(v, np.ndarray)]
        if len(valid_vectors) < 2: return

        vectors_array = np.array(valid_vectors)
        # Normalize vectors to avoid issues with zero vectors if any slipped through
        norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        # Prevent division by zero
        norms[norms == 0] = 1e-6
        normalized_vectors = vectors_array / norms

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_vectors)

        # Extract pairs above threshold
        num_images = len(image_ids)
        for i in range(num_images):
            for j in range(i + 1, num_images):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    similarity_results.append({
                        'image1_id': image_ids[i],
                        'image2_id': image_ids[j],
                        'image1_index': indices[i],
                        'image2_index': indices[j],
                        'similarity': float(similarity)
                    })

    except Exception as e:
         st.error(f"计算批量图片相似度时出错: {e}")
         st.session_state.batch_similarity_results = []
         return


    # Sort by similarity
    similarity_results.sort(key=lambda x: x['similarity'], reverse=True)
    st.session_state.batch_similarity_results = similarity_results


# 添加计算批量人脸相似度的函数
def compute_batch_face_similarity():
    """计算批量上传图片中人脸间的相似度"""
    face_similarity_results = []
    face_vectors_dict = st.session_state.batch_face_vectors

    face_ids = list(face_vectors_dict.keys())
    vectors = [face_vectors_dict[id]['vector'] for id in face_ids]
    image_indices = [face_vectors_dict[id]['image_index'] for id in face_ids]
    face_indices = [face_vectors_dict[id]['face_index'] for id in face_ids]

    if len(vectors) < 2: return # Need at least two faces

    # Calculate cosine similarity matrix for faces
    try:
         # Ensure vectors are valid numpy arrays
        valid_vectors = [v for v in vectors if isinstance(v, np.ndarray)]
        if len(valid_vectors) < 2: return

        vectors_array = np.array(valid_vectors)
        # Normalize vectors
        norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        normalized_vectors = vectors_array / norms

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(normalized_vectors)

        # Extract pairs above threshold
        num_faces = len(face_ids)
        for i in range(num_faces):
            for j in range(i + 1, num_faces):
                 # Avoid comparing faces from the exact same original image? Optional.
                 # if image_indices[i] == image_indices[j]: continue

                 similarity = similarity_matrix[i, j]
                 if similarity >= similarity_threshold: # Use general similarity threshold for now
                     face_similarity_results.append({
                         'face1_id': face_ids[i],
                         'face2_id': face_ids[j],
                         'image1_index': image_indices[i],
                         'image2_index': image_indices[j],
                         'face1_index': face_indices[i],
                         'face2_index': face_indices[j],
                         'similarity': float(similarity)
                     })

    except Exception as e:
         st.error(f"计算批量人脸相似度时出错: {e}")
         st.session_state.batch_face_similarity_results = []
         return


    # Sort by similarity
    face_similarity_results.sort(key=lambda x: x['similarity'], reverse=True)
    st.session_state.batch_face_similarity_results = face_similarity_results

# 添加显示批量图片相似结果的函数
def display_batch_similarity_results():
    """显示批量图片之间的相似度结果"""
    results = st.session_state.get('batch_similarity_results', [])
    images = st.session_state.get('batch_images', [])

    if not results:
        st.info("在批量上传的图片中未找到相似度足够高的图片对。")
        return

    st.subheader(f"批量内图片相似度结果 (Top {min(10, len(results))}, 相似度 > {similarity_threshold:.2f})")

    # Display top N results
    for i, result in enumerate(results[:10]):
        cols = st.columns([1, 0.5, 1]) # Image | Similarity | Image

        img1_index = result.get('image1_index')
        img2_index = result.get('image2_index')
        similarity = result.get('similarity', 0)

        # Display first image
        with cols[0]:
            if img1_index is not None and img1_index < len(images) and images[img1_index] is not None:
                st.image(images[img1_index], caption=f"图片 #{img1_index+1}", use_container_width=True)
            else:
                st.warning(f"图片 #{img1_index+1} 无法显示")

        # Display similarity
        with cols[1]:
            st.markdown(f"<div style='text-align: center; margin-top: 50px;'><h3>相似度<br>{similarity:.3f}</h3></div>", unsafe_allow_html=True)


        # Display second image
        with cols[2]:
            if img2_index is not None and img2_index < len(images) and images[img2_index] is not None:
                st.image(images[img2_index], caption=f"图片 #{img2_index+1}", use_container_width=True)
            else:
                 st.warning(f"图片 #{img2_index+1} 无法显示")

        st.markdown("---") # Separator


# 添加显示批量人脸相似结果的函数
def display_batch_face_similarity_results():
    """显示批量图片中人脸之间的相似度结果"""
    results = st.session_state.get('batch_face_similarity_results', [])
    all_batch_faces = st.session_state.get('batch_faces', []) # List of lists

    if not results:
        st.info("在批量上传的图片中未找到相似度足够高的人脸对。")
        return

    st.subheader(f"批量内人脸相似度结果 (Top {min(10, len(results))}, 相似度 > {similarity_threshold:.2f})")

    # Display top N results
    for i, result in enumerate(results[:10]):
        cols = st.columns([1, 0.5, 1]) # Face | Similarity | Face

        img1_idx = result.get('image1_index')
        face1_idx = result.get('face1_index')
        img2_idx = result.get('image2_index')
        face2_idx = result.get('face2_index')
        similarity = result.get('similarity', 0)

        # Display first face
        with cols[0]:
            face1_array = None
            caption1 = f"图片 #{img1_idx+1}, 人脸 #{face1_idx+1}"
            try:
                if img1_idx < len(all_batch_faces) and face1_idx < len(all_batch_faces[img1_idx]):
                    face1_array = all_batch_faces[img1_idx][face1_idx]
                    if isinstance(face1_array, np.ndarray) and face1_array.size > 0:
                        st.image(face1_array, caption=caption1, use_container_width=True)
                    else: raise ValueError("无效的人脸数据")
                else: raise IndexError("索引越界")
            except Exception as e:
                st.warning(f"人脸无法显示 ({caption1}): {e}")


        # Display similarity
        with cols[1]:
             st.markdown(f"<div style='text-align: center; margin-top: 50px;'><h3>相似度<br>{similarity:.3f}</h3></div>", unsafe_allow_html=True)


        # Display second face
        with cols[2]:
            face2_array = None
            caption2 = f"图片 #{img2_idx+1}, 人脸 #{face2_idx+1}"
            try:
                if img2_idx < len(all_batch_faces) and face2_idx < len(all_batch_faces[img2_idx]):
                     face2_array = all_batch_faces[img2_idx][face2_idx]
                     if isinstance(face2_array, np.ndarray) and face2_array.size > 0:
                         st.image(face2_array, caption=caption2, use_container_width=True)
                     else: raise ValueError("无效的人脸数据")
                else: raise IndexError("索引越界")
            except Exception as e:
                st.warning(f"人脸无法显示 ({caption2}): {e}")

        st.markdown("---") # Separator


# ... existing code ...
def summarize_batch_images(similarity_threshold=0.5):
    """
    根据批量上传图片的相似性总结场景数量 (using image embeddings)
    """
    try:
        if 'batch_images' not in st.session_state or not st.session_state.batch_images:
            return "未检测到上传的图片"

        # Get image feature vectors from batch processing results
        image_vectors_dict = st.session_state.get('batch_image_vectors', {})
        if not image_vectors_dict:
            return "未能提取图片特征，无法进行场景总结。"

        image_features = [data['vector'] for data in image_vectors_dict.values() if data.get('vector') is not None]

        if len(image_features) < 2:
            return f"仅有 {len(image_features)} 张图片具有有效特征，无法进行场景聚类。"

        # Use Agglomerative Clustering based on Cosine Distance
        features_array = np.array(image_features)

        # Calculate Cosine Distance matrix (1 - similarity)
        # Handle potential normalization issues
        norms = np.linalg.norm(features_array, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6 # Avoid division by zero
        normalized_features = features_array / norms
        similarity_matrix = cosine_similarity(normalized_features)
        # Ensure distance is non-negative and handle precision issues
        distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, 2.0)


        # Perform Agglomerative Clustering
        # distance_threshold is based on the distance metric (cosine distance here)
        # A threshold of (1 - similarity_threshold) groups items with similarity >= similarity_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold= (1.0 - similarity_threshold),
            metric='precomputed', # Use the precomputed distance matrix
            linkage='average' # Or 'complete', 'single'
        )
        clusters = clustering.fit_predict(distance_matrix)

        # Calculate the number of distinct scenes (clusters)
        unique_clusters = len(set(clusters))
        # DBSCAN might produce -1 for noise, Agglomerative usually doesn't unless distance > threshold for all
        # unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0) # More relevant for DBSCAN

        return f"分析结果：基于图片内容相似度（阈值={similarity_threshold:.2f}），上传的 {len(image_features)} 张有效图片中，大约包含 **{unique_clusters}** 个不同场景。"

    except Exception as e:
        st.error(f"场景总结出错: {str(e)}")
        # import traceback
        # print(traceback.format_exc()) # Log traceback for debugging
        return f"场景总结出错: {str(e)}"

def summarize_batch_faces(face_dist_tolerance=0.4):
    """
    根据批量上传图片中的人脸总结不同人物数量 (using face clustering results)
    """
    try:
        # Check if face processing happened and if there are faces
        if 'batch_faces' not in st.session_state:
            return "人脸数据未处理。"

        all_faces_list = [face for sublist in st.session_state.batch_faces if sublist for face in sublist]
        total_detected_faces = len(all_faces_list)

        if total_detected_faces == 0:
            return "未在上传的图片中检测到人脸。"

        # Create temporary directories for clustering input and output
        temp_base = Path(tempfile.gettempdir()) / f"streamlit_face_cluster_{uuid.uuid4().hex[:8]}"
        temp_input_dir = temp_base / "input"
        temp_output_dir = temp_base / "output"

        # Cleanup previous temp dirs if they exist (robustness)
        if temp_base.exists(): shutil.rmtree(temp_base)
        temp_input_dir.mkdir(parents=True, exist_ok=True)

        # Save all detected faces (numpy arrays) to the temp input directory
        face_count_saved = 0
        for i, faces_in_image in enumerate(st.session_state.batch_faces):
            if isinstance(faces_in_image, list):
                for j, face_np in enumerate(faces_in_image):
                    if isinstance(face_np, np.ndarray) and face_np.size > 0:
                        try:
                            face_img = Image.fromarray(face_np).convert("RGB") # Ensure RGB
                            face_path = temp_input_dir / f"face_{i}_{j}.jpg"
                            face_img.save(face_path, format="JPEG")
                            face_count_saved += 1
                        except Exception as e:
                            print(f"Error saving temp face image face_{i}_{j}: {e}")


        if face_count_saved == 0:
            # Cleanup if no faces were saved
            if temp_base.exists(): shutil.rmtree(temp_base)
            return "无法保存检测到的人脸图像进行聚类。"

        print(f"准备对 {face_count_saved} 个保存的人脸进行聚类...")

        # Run the face clustering function (which now uses MTCNN+DeepFace)
        # Use the tolerance provided (cosine distance)
        # Set min_cluster_size=2 to identify pairs as clusters, or 1 to count every face initially
        effective_min_cluster_size = 2
        face_clustering(str(temp_input_dir), str(temp_output_dir),
                        tolerance=face_dist_tolerance,
                        min_cluster_size=effective_min_cluster_size)

        # Count the number of resulting clusters (people)
        cluster_dirs = [d for d in temp_output_dir.iterdir()
                       if d.is_dir() and d.name.startswith("person_")]
        num_clusters = len(cluster_dirs)

        # Count unclustered faces (optional info)
        unclustered_dir = temp_output_dir / "unclustered_faces"
        num_unclustered = 0
        if unclustered_dir.exists():
            num_unclustered = len([f for f in unclustered_dir.iterdir() if f.is_file()])


        # Prepare summary message
        summary = f"分析结果：基于人脸相似度（距离阈值={face_dist_tolerance:.2f}），上传图片中共检测到 **{total_detected_faces}** 个有效人脸，识别出大约 **{num_clusters}** 个不同的人物聚类。"
        if num_unclustered > 0:
            summary += f" 另有 {num_unclustered} 个人脸未归入任何聚类（噪声点）。"

        # Return summary and the path to the output directory for montage display
        return summary, str(temp_output_dir)

    except Exception as e:
        st.error(f"人脸总结及聚类出错: {str(e)}")
        # import traceback
        # print(traceback.format_exc()) # Log traceback for debugging
        # Cleanup temp dir on error
        if 'temp_base' in locals() and temp_base.exists(): shutil.rmtree(temp_base)
        return f"人脸总结出错: {str(e)}", None


def display_batch_analysis_summary():
    """
    Displays the summary of batch image analysis (scenes and faces).
    """
    if 'batch_images' not in st.session_state or not st.session_state.batch_images:
        st.info("请先上传并处理批量图片。")
        return

    st.subheader("批量分析总结")

    # Sliders for tuning summarization thresholds
    col1, col2 = st.columns(2)
    with col1:
        scene_similarity_threshold = st.slider("场景聚类相似度阈值", 0.1, 0.9, 0.6, 0.05, # Higher threshold = more scenes
                                            help="值越高，需要图片越相似才会被归为同一场景，从而可能识别出更多场景。")
    with col2:
        face_distance_threshold = st.slider("人脸聚类距离阈值", 0.1, 1.0, 0.4, 0.05, # Lower threshold = more people
                                            help="值越低，匹配越严格（需要人脸特征距离越小），从而可能识别出更多不同的人。使用Cosine距离。")

    # --- Scene Summary ---
    with st.spinner("总结场景数量..."):
        scene_summary = summarize_batch_images(similarity_threshold=scene_similarity_threshold)
        st.info(scene_summary)

    # --- Face Summary and Montage Display ---
    temp_output_dir_path = None # To store path for montage creation
    with st.spinner("总结人物数量并进行聚类..."):
        face_summary, temp_output_dir_path = summarize_batch_faces(face_dist_tolerance=face_distance_threshold)
        st.info(face_summary)

    # Display face cluster montages if clustering was successful
    if temp_output_dir_path and os.path.exists(temp_output_dir_path):
        st.subheader("人脸聚类结果可视化")
        with st.spinner("生成并加载人脸聚类蒙太奇图..."):
            try:
                # Generate montages from the temp output directory
                montages_info = create_montages(temp_output_dir_path, montage_size=(100, 100), images_per_row=6) # Smaller size for batch

                if montages_info:
                    montage_cols = st.columns(min(len(montages_info), 3)) # Display up to 3 montages side-by-side
                    for i, montage in enumerate(montages_info):
                        with montage_cols[i % 3]:
                            try:
                                st.image(montage["path"],
                                        caption=f"{montage['cluster']} ({montage['images_count']} faces)",
                                        use_container_width=True)
                            except Exception as img_e:
                                st.error(f"无法显示蒙太奇图 {montage['cluster']}: {img_e}")
                elif os.path.exists(temp_output_dir_path): # Check if clustering ran but produced no montage-able clusters
                     # Check for unclustered or no-face images
                     unclustered_dir = Path(temp_output_dir_path) / "unclustered_faces"
                     no_faces_dir = Path(temp_output_dir_path) / "no_faces_detected"
                     msg = "未生成人脸聚类蒙太奇图。"
                     if unclustered_dir.exists() and any(unclustered_dir.iterdir()):
                         msg += " (可能所有聚类的人数都少于阈值，或人脸未形成聚类)"
                     elif no_faces_dir.exists() and any(no_faces_dir.iterdir()):
                         msg += " (部分图片未检测到人脸)"
                     st.info(msg)


            except Exception as e:
                st.error(f"显示人脸聚类蒙太奇图时出错: {str(e)}")
                # import traceback
                # print(traceback.format_exc())

        # --- Cleanup Temporary Directory ---
        # Perform cleanup AFTER potentially displaying montages
        try:
            temp_base_path = Path(temp_output_dir_path).parent # Get the base temp folder path
            if temp_base_path.exists() and temp_base_path.name.startswith("streamlit_face_cluster_"):
                print(f"Cleaning up temporary clustering directory: {temp_base_path}")
                shutil.rmtree(temp_base_path)
        except Exception as e:
            st.warning(f"清理临时聚类目录时出错: {e}")
            # Don't let cleanup errors stop the app
    elif face_summary and "出错" not in face_summary: # Check if face summary indicated an error earlier
         st.info("人脸聚类未产生有效输出目录，无法显示蒙太奇图。")


# ... existing code ...

# Main program logic wrapped in try-except
try:
    # Get selected model name and load it
    selected_model = model_mapping[model_option]
    model = load_model(selected_model) # Handles model loading/downloading

    # Load face detector (MTCNN) if enabled
    face_detector = None
    if detect_faces:
        face_detector = load_face_detector() # Handles MTCNN loading

    # Load feature extractor models if needed
    feature_extractor = None
    transform = None
    if enable_similarity_search or enable_face_similarity:
        feature_extractor = load_feature_extractor() # Loads ResNet50
        transform = get_transform()
        # DeepFace models are loaded on demand by DeepFace functions

    st.info(f"当前使用模型: {model_option}")

    # --- Batch Processing Mode ---
    if st.session_state.batch_mode:
        st.header("批量处理模式")
        uploaded_files = st.file_uploader("上传多张图片进行批量处理", type=["jpg", "jpeg", "png", "bmp", "tiff"], accept_multiple_files=True)

        if uploaded_files:
            st.write(f"已选择 {len(uploaded_files)} 张图片")

            if st.button(f"开始处理 {len(uploaded_files)} 张图片"):
                # Execute batch processing
                processed_count, failed_files = process_batch_images(uploaded_files)
                st.success(f"批量处理完成！成功处理 {processed_count} 张图片。")
                if failed_files:
                     st.warning(f"处理失败的文件: {', '.join(failed_files)}")

                # --- Display Batch Results ---
                if processed_count > 0:
                    st.subheader("批量处理结果概览")

                    # Create a summary DataFrame
                    batch_summary_data = []
                    valid_images = [img for img in st.session_state.batch_images if img is not None]
                    valid_processed = [img for img in st.session_state.batch_processed_images if img is not None]
                    valid_faces = [f for f in st.session_state.batch_faces if f is not None] # List of lists or Nones

                    for i in range(len(valid_images)): # Iterate based on successful image loads
                         image = valid_images[i]
                         face_list = valid_faces[i] if i < len(valid_faces) else []
                         face_count = len(face_list) if isinstance(face_list, list) else 0

                         batch_summary_data.append({
                             "图片序号": i + 1,
                             "原始尺寸": f"{image.width}x{image.height}",
                             "检测到的人脸数": face_count
                         })
                    if batch_summary_data:
                        st.dataframe(batch_summary_data)
                    else:
                        st.info("没有成功处理的图片可供概览。")


                    # Tabs for detailed results
                    tab_titles = ["处理后图片", "检测到人脸"]
                    if enable_similarity_search: tab_titles.append("图片相似度")
                    if enable_face_similarity: tab_titles.append("人脸相似度")
                    tab_titles.append("批量分析总结")

                    tabs = st.tabs(tab_titles)
                    tab_idx = 0

                    # Tab 1: Processed Images
                    with tabs[tab_idx]:
                        st.subheader("处理后的图片")
                        cols = st.columns(3)
                        img_display_count = 0
                        for i, processed_img_array in enumerate(valid_processed):
                             if processed_img_array is not None:
                                 with cols[img_display_count % 3]:
                                     st.image(processed_img_array, caption=f"图片 #{i+1}", use_container_width=True)
                                     img_display_count += 1
                        if img_display_count == 0: st.info("无处理后的图片可显示。")
                    tab_idx += 1

                    # Tab 2: Detected Faces
                    with tabs[tab_idx]:
                        st.subheader("检测到的人脸")
                        face_display_count = 0
                        for i, faces_list in enumerate(valid_faces):
                             if isinstance(faces_list, list) and faces_list:
                                 st.write(f"--- 图片 #{i+1} ({len(faces_list)} 人脸) ---")
                                 cols = st.columns(min(len(faces_list), 5)) # Show up to 5 faces per row
                                 for j, face_array in enumerate(faces_list):
                                     if isinstance(face_array, np.ndarray) and face_array.size > 0:
                                          with cols[j % 5]:
                                             st.image(face_array, caption=f"人脸 #{j+1}", use_container_width=True)
                                             face_display_count +=1
                        if face_display_count == 0: st.info("未检测到任何人脸。")
                    tab_idx += 1


                    # Tab 3: Image Similarity (Conditional)
                    if enable_similarity_search:
                         with tabs[tab_idx]:
                             display_batch_similarity_results()
                         tab_idx += 1

                    # Tab 4: Face Similarity (Conditional)
                    if enable_face_similarity:
                         with tabs[tab_idx]:
                             display_batch_face_similarity_results()
                         tab_idx += 1

                    # Tab 5: Batch Analysis Summary
                    with tabs[tab_idx]:
                        display_batch_analysis_summary()
                    tab_idx += 1

                else: # If processed_count is 0
                    st.warning("未能成功处理任何图片，无法显示结果。")

    # --- Single Image Processing Mode ---
    else:
        st.header("单张图片处理模式")
        # Use on_change to clear state BEFORE processing the new file
        uploaded_file = st.file_uploader("上传单张图片", type=["jpg", "jpeg", "png", "bmp", "tiff"], on_change=reset_processed_state)

        if uploaded_file is not None:
            current_upload_id = f"{uploaded_file.name}-{uploaded_file.size}" # More robust ID

            # Check if it's a new file or if processing hasn't happened yet
            if current_upload_id != st.session_state.get('last_upload_id') or not st.session_state.get('processed'):
                st.session_state.processed = False # Ensure reset before processing starts
                st.session_state.image = None
                st.session_state.processed_img = None
                st.session_state.detected_faces = []
                st.session_state.similar_images = []
                st.session_state.similar_faces_results = []
                st.session_state.face_best_matches = []
                st.session_state.similar_texts = []
                st.session_state.advanced_face_search_done = False


                processing_placeholder = st.empty() # Placeholder for status messages/spinners
                col1, col2 = st.columns(2) # Setup columns for display later

                with processing_placeholder:
                    with st.spinner("读取和预处理图片..."):
                        try:
                            # Read image
                            image = Image.open(uploaded_file)

                            # Get original size and file size
                            original_width, original_height = image.size
                            uploaded_file.seek(0)
                            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

                            # Compress if large
                            if file_size_mb > 1.5:
                                st.warning(f"图片较大 ({file_size_mb:.2f}MB)，将进行压缩...")
                                image = compress_image(image, max_size_kb=800, max_dimension=1920)
                                compressed_size_kb = len(image.tobytes()) / 1024 # Approximation after compression
                                st.success(f"压缩完成！ 新尺寸: {image.width}x{image.height} (~{compressed_size_kb:.1f}KB)")

                            st.session_state.image = image # Store potentially compressed image
                            st.session_state.last_upload_id = current_upload_id

                            with col1:
                                st.subheader("原始图片 (处理后)")
                                st.image(image, use_container_width=True)

                        except Exception as e:
                            st.error(f"读取或压缩图片时出错: {str(e)}")
                            st.stop() # Stop execution if image loading fails

                # --- Object Detection ---
                with processing_placeholder, st.spinner(f"物体检测 ({model_option})..."):
                    try:
                        results = model.predict(st.session_state.image, verbose=False)
                        st.session_state.detection_results = results
                        # Get selected classes
                        selected_classes = []
                        if detect_person: selected_classes.append("人")
                        if detect_cup: selected_classes.append("杯子/酒杯")
                        if detect_bottle: selected_classes.append("瓶子")
                        if detect_all: selected_classes.append("检测所有支持的物体")

                    except Exception as e:
                         st.error(f"物体检测失败: {e}")
                         results = [None] # Set results to None on failure
                         st.session_state.detection_results = results


                # --- Face Detection ---
                detected_faces = [] # List of numpy arrays for faces
                if detect_faces and face_detector:
                     with processing_placeholder, st.spinner("人脸检测 (MTCNN)..."):
                         try:
                             # Use the potentially compressed image for detection
                             _, detected_faces = detect_face(st.session_state.image, face_detector, face_confidence)
                             st.session_state.detected_faces = detected_faces # Store numpy arrays
                             print(f"检测到 {len(detected_faces)} 个人脸 (MTCNN, conf>{face_confidence})")
                         except Exception as e:
                             st.error(f"人脸检测失败: {e}")


                # --- Draw Detections and Faces ---
                with processing_placeholder, st.spinner("绘制检测结果..."):
                    # Start with the base image
                    processed_img_base = st.session_state.image.copy().convert("RGB")
                    processed_img_array = np.array(processed_img_base)

                    # Draw object boxes if detection was successful
                    if st.session_state.detection_results and st.session_state.detection_results[0] is not None:
                         processed_img_array = process_prediction(processed_img_base, st.session_state.detection_results, selected_classes, confidence)

                    # Draw face boxes on top of the image (which might already have object boxes)
                    if detected_faces: # Draw if faces were found
                        # Re-detect face bounding boxes on the original image array for drawing accuracy
                        img_rgb_for_drawing = np.array(st.session_state.image.convert("RGB"))
                        faces_data_for_drawing = face_detector.detect_faces(img_rgb_for_drawing)
                        filtered_faces_data = [f for f in faces_data_for_drawing if f['confidence'] >= face_confidence]

                        # Convert current processed array (with object boxes) to BGR for cv2 drawing
                        img_bgr_for_drawing = cv2.cvtColor(processed_img_array, cv2.COLOR_RGB2BGR)
                        try:
                            hex_color = face_box_color.lstrip('#')
                            face_bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                        except ValueError: face_bgr_color = (0, 255, 0)

                        for face in filtered_faces_data:
                            x, y, w, h = face['box']
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(img_bgr_for_drawing.shape[1], x + w), min(img_bgr_for_drawing.shape[0], y + h)
                            cv2.rectangle(img_bgr_for_drawing, (x1, y1), (x2, y2), face_bgr_color, 2)
                            # Add label if needed
                            # label = f"Face: {face['confidence']:.2f}"
                            # cv2.putText(img_bgr_for_drawing, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                        # Convert back to RGB
                        processed_img_array = cv2.cvtColor(img_bgr_for_drawing, cv2.COLOR_BGR2RGB)

                    # Store the final image with all markings
                    st.session_state.processed_img = processed_img_array


                # --- Update DB and Similarity Search ---
                image_id = None # Initialize image_id
                if enable_similarity_search or enable_face_similarity:
                    with processing_placeholder, st.spinner("更新向量库并执行相似搜索..."):
                        try:
                            # Update DB returns the new image_id and list of face_ids
                            image_id, face_ids = update_vector_db(
                                st.session_state.image, # The potentially compressed PIL image
                                st.session_state.detected_faces, # List of face numpy arrays
                                feature_extractor,
                                transform
                            )

                            if image_id:
                                # Reload DBs to get the latest data including the current item
                                image_db, face_db = initialize_vector_db()

                                # Image Similarity Search
                                if enable_similarity_search and image_id in image_db:
                                    current_image_vector = image_db[image_id].get('vector')
                                    if current_image_vector is not None:
                                        # Exclude current image from search results
                                        search_image_db = {k: v for k, v in image_db.items() if k != image_id}
                                        similar_images = search_similar_images(
                                            current_image_vector, search_image_db, top_k=top_k, threshold=similarity_threshold
                                        )
                                        st.session_state.similar_images = similar_images
                                    else: st.session_state.similar_images = []

                                # Face Similarity Search
                                if enable_face_similarity and face_ids:
                                    current_face_vectors = [face_db[fid]['vector'] for fid in face_ids if fid in face_db and face_db[fid].get('vector') is not None]
                                    if current_face_vectors:
                                        # Exclude current image's faces from search
                                        search_face_db = {k: v for k, v in face_db.items() if k not in face_ids}
                                        similar_faces_results = search_similar_faces(
                                            current_face_vectors, search_face_db, top_k=top_k, threshold=similarity_threshold # Use general threshold
                                        )
                                        st.session_state.similar_faces_results = similar_faces_results
                                    else: st.session_state.similar_faces_results = []
                            else:
                                 st.warning("无法更新数据库，跳过相似性搜索。")
                                 st.session_state.similar_images = []
                                 st.session_state.similar_faces_results = []

                        except Exception as e:
                            st.error(f"相似性搜索或数据库更新失败: {e}")
                            st.session_state.similar_images = []
                            st.session_state.similar_faces_results = []


                # --- Mark processing as done ---
                st.session_state.processed = True
                processing_placeholder.empty() # Clear the spinner/status message


            # --- Display Results (if processed) ---
            if st.session_state.get('processed'):
                 # Display potentially compressed original image
                 if 'image' in st.session_state and st.session_state.image:
                     with col1:
                         st.subheader("原始图片 (处理后)")
                         st.image(st.session_state.image, use_container_width=True)

                 # Display processed image with detections
                 if 'processed_img' in st.session_state and st.session_state.processed_img is not None:
                    with col2:
                        st.subheader("检测结果")
                        st.image(st.session_state.processed_img, use_container_width=True)
                        # Add download button for the processed image
                        try:
                            buf = io.BytesIO()
                            pil_img = Image.fromarray(st.session_state.processed_img)
                            pil_img.save(buf, format="PNG")
                            st.download_button(
                                label="下载检测结果图",
                                data=buf.getvalue(),
                                file_name=f"detected_{Path(uploaded_file.name).stem}.png",
                                mime="image/png",
                            )
                        except Exception as e:
                            st.error(f"创建下载文件失败: {e}")
                 else:
                     with col2:
                         st.warning("无法显示处理后的图片。")


                 # --- Display Sections: Detections, Faces, Similarity, AI Analysis ---
                 st.markdown("---") # Separator

                 # Column layout for results
                 res_col1, res_col2 = st.columns(2)

                 with res_col1:
                    # Display Detection Statistics
                    st.subheader("物体检测统计")
                    if st.session_state.get('detection_results') and st.session_state.detection_results[0] is not None and hasattr(st.session_state.detection_results[0], 'boxes') and st.session_state.detection_results[0].boxes is not None:
                        results = st.session_state.detection_results
                        boxes = results[0].boxes
                        class_names = results[0].names if hasattr(results[0], 'names') else {}
                        class_counts = {}

                        # Recalculate selected classes based on sidebar state *now*
                        selected_classes_now = []
                        if detect_person: selected_classes_now.append("人")
                        if detect_cup: selected_classes_now.append("杯子/酒杯")
                        if detect_bottle: selected_classes_now.append("瓶子")
                        if detect_all: selected_classes_now.append("检测所有支持的物体")
                        # Map names to IDs
                        class_mapping_now = {"人": 0, "杯子/酒杯": 41, "瓶子": 39}
                        selected_class_ids_now = list(class_names.keys()) if detect_all else [class_mapping_now[cls] for cls in selected_classes_now if cls in class_mapping_now]

                        for box in boxes:
                            cls_id = int(box.cls.item())
                            conf = box.conf.item()
                            if cls_id in selected_class_ids_now and conf >= confidence:
                                cls_name = class_names.get(cls_id, f'ID:{cls_id}')
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                        if class_counts:
                             for cls_name, count in class_counts.items():
                                 st.write(f"- **{cls_name}**: {count} 个")
                        else:
                            st.write("未检测到选定类别的物体。")
                    else:
                        st.write("无物体检测结果。")

                    # Display Detected Faces (if any)
                    if detect_faces:
                        st.subheader(f"人脸检测与情绪分析 ({len(st.session_state.get('detected_faces', []))})")
                        detected_faces_list = st.session_state.get('detected_faces', [])
                        if detected_faces_list:
                            face_cols = st.columns(min(len(detected_faces_list), 4))
                            emotion_emojis = {"angry": "😠","disgust": "🤢","fear": "😨","happy": "😊","sad": "😢","surprise": "😲","neutral": "😐"}

                            for i, face_array in enumerate(detected_faces_list):
                                with face_cols[i % 4]:
                                    st.image(face_array, caption=f"人脸 #{i+1}", use_container_width=True)
                                    try:
                                        with st.spinner("分析情绪..."):
                                             # DeepFace emotion analysis
                                             # Important: DeepFace expects BGR sometimes, but MTCNN gives RGB crop.
                                             # Let's try passing the RGB array. If it fails, try converting.
                                             try:
                                                  result = DeepFace.analyze(face_array, actions=['emotion'], enforce_detection=False, detector_backend='skip')
                                             except ValueError as ve: # If it fails, try BGR
                                                  if "BGR" in str(ve):
                                                       print("DeepFace analyze failed with RGB, trying BGR...")
                                                       face_bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
                                                       result = DeepFace.analyze(face_bgr, actions=['emotion'], enforce_detection=False, detector_backend='skip')
                                                  else: raise ve # Re-raise other ValueErrors

                                             # Check result format
                                             if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                                                  emotion = result[0].get('dominant_emotion', '未知')
                                                  emotion_emoji = emotion_emojis.get(emotion, "")
                                                  st.write(f"情绪: **{emotion}** {emotion_emoji}")
                                             else:
                                                  st.write("情绪: 分析失败")

                                    except Exception as e:
                                        st.write(f"情绪: 分析出错 ({type(e).__name__})")
                                        # print(f"DeepFace emotion error face {i}: {e}") # Log error
                        else:
                            st.info("未检测到人脸。")

                 with res_col2:
                    # Display AI Analysis (Groq + Gemini/OpenAI)
                    st.subheader("AI 综合分析")
                    with st.spinner("执行 AI 分析和合规总结..."):
                        groq_api_key = os.environ.get("GROQ_API_KEY")
                        openai_api_key = os.environ.get("GEMINI_API_KEY") # Using Gemini Key

                        if groq_api_key and openai_api_key:
                            image_to_analyze = st.session_state.get('image')
                            if image_to_analyze:
                                # 1. Get image dimensions
                                image_dimensions = check_image_dimensions(image_to_analyze)

                                # 2. Groq Analysis
                                groq_analysis = analyze_with_groq(image_to_analyze, groq_api_key)
                                # st.text_area("Groq 详细分析:", groq_analysis, height=200) # Optional: show raw Groq output

                                # 3. Prepare CV Summary for OpenAI/Gemini
                                cv_summary_dict = {
                                    "detected_objects": class_counts if 'class_counts' in locals() else {},
                                    "face_count": len(st.session_state.get('detected_faces', []))
                                }

                                # 4. OpenAI/Gemini Summary
                                final_summary = summarize_with_openai(cv_summary_dict, groq_analysis, openai_api_key, image_dimensions)
                                st.markdown("##### 合规性分析报告:")
                                st.markdown(final_summary) # Display the summary table


                                # 5. Text Extraction and Similarity (after Groq)
                                if image_id: # Ensure image_id was set from DB update
                                    text_content = extract_text_from_analysis(groq_analysis)
                                    print(f"提取的文本内容: '{text_content}'") # Debugging
                                    if text_content and text_content.strip().lower() != '无文字内容':
                                        update_text_vector_db(image_id, text_content)
                                        print(f"Searching similar text for {image_id}...")
                                        similar_texts = search_similar_texts(image_id, top_k=top_k, threshold=0.85) # Higher threshold for text
                                        st.session_state.similar_texts = similar_texts
                                        if similar_texts:
                                             display_similar_texts(similar_texts)
                                        else:
                                             print("No similar text found above threshold.")
                                    else:
                                        print("无有效文字内容，跳过文本相似性搜索。")
                                        st.session_state.similar_texts = []

                                    with st.expander("图片中提取的文字内容 (来自 AI 分析)"):
                                        st.write(text_content if text_content else "无文字内容")
                                else:
                                     st.warning("无法提取或比较文本，因为图片未能添加到数据库。")


                            else:
                                st.warning("无法执行 AI 分析，因为处理后的图片不可用。")
                        else:
                            st.warning("请设置 GROQ_API_KEY 和 GEMINI_API_KEY 环境变量以启用完整的 AI 分析功能。")


                 st.markdown("---") # Separator

                 # --- Similarity Search Results ---
                 results_exist = st.session_state.get('similar_images') or st.session_state.get('similar_faces_results')
                 if results_exist:
                     st.subheader("相似性搜索结果")
                     sim_col1, sim_col2 = st.columns(2)
                     with sim_col1:
                         if enable_similarity_search and st.session_state.get('similar_images'):
                             display_similar_images(st.session_state.similar_images)
                         elif enable_similarity_search:
                             st.info("未找到相似图片。")

                     with sim_col2:
                         if enable_face_similarity and st.session_state.get('similar_faces_results'):
                             display_similar_faces(st.session_state.similar_faces_results, st.session_state.get('detected_faces', []))
                         elif enable_face_similarity:
                             st.info("未找到相似人脸。")


                 st.markdown("---") # Separator

                 # --- Advanced Face Search (Offline Clustering Comparison) ---
                 if enable_advanced_face_search and st.session_state.get('detected_faces'):
                     st.subheader("加强人脸检索 (与离线聚类比较)")
                     detected_faces_list = st.session_state.get('detected_faces', [])

                     if len(detected_faces_list) > 10: # Limit for performance
                          st.warning(f"检测到的人脸数量 ({len(detected_faces_list)}) 较多，为提高性能已跳过加强人脸检索。")
                     else:
                         # Check if tolerance changed or search hasn't been done
                         tolerance_changed = st.session_state.get('current_tolerance') != face_cluster_tolerance
                         search_needed = not st.session_state.get('advanced_face_search_done') or tolerance_changed

                         if search_needed:
                             with st.spinner("与离线聚类比较人脸..."):
                                 st.session_state.current_tolerance = face_cluster_tolerance # Update stored tolerance
                                 # Pass the list of numpy arrays
                                 face_best_matches = process_face_for_advanced_search(
                                     detected_faces_list,
                                     tolerance=face_cluster_tolerance # Pass current tolerance (distance)
                                 )
                                 st.session_state.face_best_matches = face_best_matches
                                 st.session_state.advanced_face_search_done = True
                         else:
                             face_best_matches = st.session_state.get('face_best_matches', [])

                         # Display results
                         if face_best_matches:
                             st.write(f"为 {len(face_best_matches)} 个检测到的人脸在离线聚类中找到了最佳匹配：")
                             match_cols = st.columns(min(len(face_best_matches), 3))
                             for i, match in enumerate(face_best_matches):
                                 with match_cols[i % 3]:
                                     face_idx = match['face_idx']
                                     st.write(f"**检测人脸 #{face_idx + 1}**")
                                     # Display the detected face
                                     if face_idx < len(detected_faces_list):
                                          st.image(detected_faces_list[face_idx], width=100)
                                     # Display the matched cluster montage (if path exists)
                                     if match.get('path') and os.path.exists(match['path']):
                                         st.image(match["path"],
                                                  caption=f"匹配聚类: {match['cluster']} (相似度: {match['similarity']:.3f})",
                                                  use_container_width=True)
                                     elif match.get('cluster'): # Show cluster name even if montage missing
                                          st.info(f"匹配聚类: {match['cluster']} (相似度: {match['similarity']:.3f}, 蒙太奇图丢失)")
                                     else: # Should not happen if match exists
                                          st.warning("匹配信息不完整。")
                         else:
                              st.info("未在离线聚类中找到与当前图片人脸足够相似的聚类。")

        # End of single image processing block
        # Display usage instructions if no file is uploaded yet
        elif not uploaded_file:
             st.info("请在上方上传单张图片开始处理，或在侧边栏切换到批量处理模式。")


except Exception as e:
    st.error(f"应用发生严重错误: {type(e).__name__} - {str(e)}")
    st.error("请检查控制台日志获取详细信息。如果问题持续，请尝试刷新页面或重启应用。")
    # Optional: Log the full traceback for debugging
    import traceback
    st.code(traceback.format_exc())
    # Attempt to reset state partially on major error
    reset_processed_state()


# Add usage instructions at the end
st.markdown("---")
with st.expander("使用说明和模型信息", expanded=False):
    st.markdown("""
    ### 使用方法：
    1.  **模式选择**: 在侧边栏底部选择“单张图片处理”或启用“批量处理模式”。
    2.  **上传**:
        *   单张模式：使用主界面的上传框上传一张图片。
        *   批量模式：使用主界面的上传框上传多张图片。
    3.  **设置 (侧边栏)**:
        *   **模型选择**: 选择合适的YOLO模型（精度 vs. 速度）。
        *   **置信度阈值**: 调整物体检测的置信度（越高越严格）。
        *   **检测类别**: 选择希望检测的物体。
        *   **人脸检测**: 启用/禁用MTCNN人脸检测及置信度。
        *   **相似搜索**: 启用/禁用图片及人脸相似性搜索，设置阈值和返回数量。
        *   **标记颜色**: 自定义物体和人脸边界框颜色。
        *   **加强人脸检索**: 启用后，会将检测到的人脸与预先生成的离线人脸聚类进行比较。调整“严格度”（距离阈值，越低越严）和“最小聚类数量”。**注意**: 此功能需要预先运行 `face_clustering` 脚本生成 `clustered_faces_*` 目录。
    4.  **处理**:
        *   单张模式：图片上传后自动处理。
        *   批量模式：上传图片后，点击“开始处理”按钮。
    5.  **查看结果**:
        *   单张模式：结果直接显示在主界面。
        *   批量模式：结果显示在不同的标签页中。
        *   **AI综合分析**: （需要API Key）提供Groq的详细分析和Gemini/OpenAI的合规性总结。
        *   **相似文本检查**: （需要API Key和`sentence-transformers`）如果AI分析提取到文字，会与历史图片中的文字进行相似度比较。

    ### 模型和技术:
    *   **物体检测**: YOLOv8 / YOLOv11 (Ultralytics) - 基于COCO数据集预训练。
    *   **人脸检测**: MTCNN (Multi-task Cascaded Convolutional Networks)。
    *   **人脸特征提取/比较**: DeepFace (支持VGG-Face, Facenet, ArcFace等模型，默认使用ArcFace进行特征提取和相似度计算)。
    *   **图片特征提取**: ResNet50 (PyTorch).
    *   **相似度计算**: Cosine Similarity.
    *   **AI分析**: Groq (Llama 3 Vision) / OpenAI (Gemini Pro/Flash)。
    *   **文本特征提取**: Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`).
    *   **人脸聚类**: DBSCAN / Agglomerative Clustering (Scikit-learn) 结合 DeepFace 特征。
    """)

# Footer
st.markdown("---")
st.markdown("工具版本 v1.2 | 技术栈: Streamlit, YOLO, MTCNN, DeepFace, Groq, Gemini, Scikit-learn")
