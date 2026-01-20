import cv2
import numpy as np
from ultralytics import YOLO
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import streamlit as st

st.set_page_config(page_title="E-commerce Image Auditor")
st.title('🛒 E-commerce Image Auditor')
st.write('AI-powered professional quality evaluation using YOLOv12 and DeepSeek-R1.')

@st.cache_resource
def load_models():
    """Loading the YOLO model to track the object in the image"""
    try:
        model = YOLO('./models/yolo12n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_models()

def get_composition_metrics(img, yolo_boxes):
    """
    This function block will extract the following signals from the images
    1. object_centered :- Indicates whether the object is in the centre of the image or not
    2. object_coverage :- Also known as aspect ratio tells us how much area does the object/product occupies in the image
    3. lightning_quality :- Tells us nature of the lightning of the image
    """

    h, w, _ = img.shape
    img_center = (w / 2, h / 2)
    
    metrics = {
        "is_centered": False,
        "object_coverage": 0.0,
        "brightness_score": 0.0,
        "lighting_quality": "Unknown"
    }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    metrics["brightness_score"] = round(avg_brightness / 255, 2)
    
    if avg_brightness < 50: metrics["lighting_quality"] = "Under-exposed"
    elif avg_brightness > 220: metrics["lighting_quality"] = "Over-exposed"
    else: metrics["lighting_quality"] = "Good"

    if len(yolo_boxes) > 0:
        x1, y1, x2, y2 = yolo_boxes[0]
        obj_w, obj_h = x2 - x1, y2 - y1
        obj_center = (x1 + obj_w/2, y1 + obj_h/2)
        
        dist = np.sqrt((obj_center[0] - img_center[0])**2 + (obj_center[1] - img_center[1])**2)
        max_dist = np.sqrt(w**2 + h**2) / 2
        metrics["is_centered"] = (dist / max_dist) < 0.15 
        metrics["object_coverage"] = round((obj_w * obj_h) / (w * h), 2)

    return metrics

def get_normalized_sharpness(image_array, target_size=640):
    """To detect the blur of the images"""

    if image_array is None or image_array.size == 0:
        return 0.0

    resized = cv2.resize(image_array, (target_size, target_size), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    raw_score = cv2.Laplacian(resized, cv2.CV_64F).var()
    return round(min(raw_score / 1000.0, 1.0), 2) # Normalized to 0-1

def get_signal_metrics(orig_img):
    """
    this function will 
    1. Detects the object in the image
    2. Gives us the confidence score of that detected objects
    3. Helps to call the other function to get sharpness of the image
    """

    output = model.predict(orig_img, device='cpu', save=False, conf=0.5)
    boxes = output[0].boxes.xyxy.cpu().numpy()
    
    conf = [round(float(c), 2) for c in output[0].boxes.conf.cpu().numpy()]
    detected_labels = [model.names[int(cls)] for cls in output[0].boxes.cls]
    
    crop_score = 0.0
    composition = {"is_centered": False, "object_coverage": 0.0, "lighting_quality": "N/A"}
    
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        crops = orig_img[y1:y2, x1:x2]
        crop_score = get_normalized_sharpness(crops)
        composition = get_composition_metrics(orig_img, [[x1, y1, x2, y2]])
    else:
        st.write("**No valid image is found..... Retry with another image ⚠️⚠️⚠️**")
        st.stop()

    global_score = get_normalized_sharpness(orig_img)

    return {
        "detected_objects": detected_labels,
        "confidence_score": conf, 
        "metrics": composition,
        "global_sharpness": global_score, 
        "object_sharpness": crop_score if len(boxes) > 0 else 0.0,
        "is_subject_focused": crop_score > (global_score * 1.1) if len(boxes) > 0 else False
    }

class AuditVerdict(BaseModel):
    """Tuning the output of the LLM"""

    is_professional: bool = Field(description="Final decision")
    reasoning: str = Field(description="Technical explanation")
    confidence_score: float = Field(description="Confidence in audit")
    suggested_improvements: List[str] = Field(description="Fixes")

llm = ChatOllama(model="deepseek-r1:8b", temperature=0, format="json").with_structured_output(AuditVerdict)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Professional E-commerce Image Auditor. Audit images based ONLY on technical signals provided."),
    ("human", "Signals: {signals}")
])

uploaded_file = st.file_uploader('Upload Product Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) # For correct display
    
    st.image(opencv_image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner('Analyzing visual signals...'):
        
        bgr_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        signals = get_signal_metrics(bgr_image)
        
        st.subheader("Extracted Signals")
        st.json(signals)
        
    with st.spinner('LLM Reasoning in progress...'):
        chain = prompt | llm
        verdict = chain.invoke({"signals": str(signals)})
        
        st.success("Audit Complete!")
        st.write('Raw verdict - ', verdict)
        
        st.write(f"**Professional Quality:** {'✅ Yes' if verdict.is_professional else '❌ No'}")
        st.write(f"**Reasoning:** {verdict.reasoning}")
        st.write(f"**Suggestions:** {', '.join(verdict.suggested_improvements)}")