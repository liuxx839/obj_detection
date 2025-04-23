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
import faiss
import torch
from torchvision import models, transforms
from groq import Groq
from openai import OpenAI
import base64
from io import BytesIO
from mtcnn import MTCNN
from deepface import DeepFace



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
    similarity_threshold = st.slider("相似度阈值", 0.0, 1.0, 0.8, 0.05)
    top_k = st.slider("返回相似结果数量", 1, 10, 3)
    
    # 获取颜色设置
    st.subheader("标记颜色")
    box_color = st.color_picker("边框颜色", "#FF0000")
    face_box_color = st.color_picker("人脸边框颜色", "#00FF00")

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
    # 转换为numpy数组
    img = np.array(image)
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

# 添加一个清除处理状态的函数
def reset_processed_state():
    st.session_state.processed = False
    st.session_state.detection_results = None
    st.session_state.processed_img = None
    st.session_state.detected_faces = []
    st.session_state.similar_images = []
    st.session_state.similar_faces_results = []

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
                    ▪️ 主体内容：清晰描述画面中的物体、场景、动作等细节（如 "室内会议场景，3 人围坐讨论，桌上摆放笔记本电脑和文件"）。
                    ▪️ 人物信息（若包含人）：
                    ・人头数：X 人（精确计数）。
                    ・性别比例：男性 X 人，女性 Y 人，性别不明 Z 人（无人物则填 0）。
                    ・种族比例：黑人，白人，亚洲人等
                    ・年龄段：按儿童（＜12 岁）、青年（12-35 岁）、中年（36-60 岁）、老年（＞60 岁）分类，标注各年龄段人数（如 "青年 2 人，中年 1 人"）。
                    ▪️ 所在场景：具体描述环境（如 "医院候诊区""户外公园""实验室" 等），需包含室内 / 室外、地点特征等细节。
                    ▪️ Sanofi 相关性：明确说明是否出现 Sanofi 标志（如 LOGO、品牌名称）、产品（如药品包装、宣传材料）、文字提及（如 "Sanofi""赛诺菲" 字样）或相关场景（如 Sanofi 活动、合作项目等），若无则标注 "无相关元素"。
                    使用说明：
                    若图片为纯文字（如合同、说明书），输出仅包含提取的文字内容。
                    若图片为非文字内容（如照片、插画），按上述格式分点详细描述，人物信息需严格按要求分类统计。
                    确保语言简洁准确，避免主观推断，仅基于图片可见内容输出。
                    新增验证项
                    针对以下场景，明确标注工具是否正确识别（需基于图片可见内容判断，而非主观推测）：
                    图片风格： 是否为真实照片，还是某种风格图像
                    空场景识别：图片是否为无人、无明显物体的空场景（如空白墙面、纯色背景）？
                    ▶ 识别结果：是 / 否 / 无法判断（若有物体或人物，标注 "非空场景"）。
                    会议无关场景识别：场景是否与会议无直接关联（如纯自然风景、单独静物、娱乐场所等）？
                    ▶ 识别结果：是（会议无关）/ 否（可能与会议相关）/ 无法判断。
                    人数捕捉：工具是否准确统计图片中的人头数？（若有人物，需与实际计数一致）
                    ▶ 实际人头数：X 人；工具识别结果：X 人（一致 / 不一致）。
                    会议设备识别：是否正确识别会议必要设备（如投影仪、电脑、麦克风、白板、会议桌等）？
                    ▶ 识别设备：若有，列出具体设备名称（如 "投影仪、笔记本电脑"）；若无，标注 "未识别到会议设备"。
                    会议照片翻拍识别：图片是否为会议照片的翻拍（如对屏幕、纸质照片的二次拍摄，可能存在反光、变形等特征）？
                    ▶ 识别结果：是（翻拍照片）/ 否（非翻拍照片）/ 无法判断。
                    非正常会议场所识别：场景是否为非正常会议场所（如娱乐场所、开放区域、家庭环境等）？
                    ▶ 场所类型：正常会议场所（如会议室）/ 非正常会议场所（具体描述，如 "KTV 包厢""公园草坪"）/ 无法判断。
                    参会人露脸识别：是否正确识别人物露脸情况（全部露脸 / 部分未露脸 / 全部未露脸）？
                    ▶ 露脸情况：若有人物，标注 "全部露脸" 或 "X 人未露脸"；无人物则标注 "无人物"。
                    非合规物件识别：是否出现与会议无关的非合规物件（如酒瓶、麻将桌、酒杯，游戏机等），需要着重检查？
                    ▶ 识别结果：是（具体物件：XXX）/ 否（无非合规物件）/ 无法判断。
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
def summarize_with_openai(cv_results, groq_analysis, api_key):
    client = OpenAI(api_key=api_key,base_url="https://generativelanguage.googleapis.com/v1beta/")
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": """你是一位严谨的药厂审查关员。请根据CV检测结果和AI分析结果，生成一份简洁的合规性分析报告。
                    重点关注：
                    0. 是否有sanofi的logo，或者明确和sanofi相关的因素
                    1. 场景合规性（是否为正式会议场所）
                    2. 人员情况（人数、着装、行为是否得体）
                    3. 物品合规性（是否有违规物品如酒精饮品）
                    4. 图片风格是否为真实
                    5. 整体评估（是否建议通过审核）
                    请以表格形式输出，对于违规情况，加粗字体，使用markdown格式。"""
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
    
    # 修改上传图片处理部分
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"], on_change=reset_processed_state)

    # 检查是否有文件上传且尚未处理
    if uploaded_file is not None:
        # 为上传文件生成唯一ID
        current_upload_id = id(uploaded_file)
        
        # 显示原始图片
        col1, col2 = st.columns(2)
        
        # 如果是新上传的文件或者尚未处理
        if current_upload_id != st.session_state.last_upload_id or not st.session_state.processed:
            # 读取上传的图片
            image = Image.open(uploaded_file)
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
                    # 执行Groq分析
                    groq_analysis = analyze_with_groq(image, groq_api_key)
                    print(groq_analysis)
                    # 准备CV检测结果摘要
                    cv_summary = {
                        "detected_objects": class_counts,
                        "face_count": len(st.session_state.detected_faces) if detect_faces else 0
                    }
                    
                    # 执行OpenAI总结
                    final_summary = summarize_with_openai(cv_summary, groq_analysis, openai_api_key)
                    
                    # 显示分析结果
                    st.markdown(final_summary)
                else:
                    st.warning("请设置GROQ_API_KEY和OPENAI_API_KEY环境变量以启用AI分析功能")

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


###
