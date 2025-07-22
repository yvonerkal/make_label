import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import os
import io
import uuid
from PIL import Image
import zipfile
from io import BytesIO


# 切换时自动音频播放功能
# ======== 工具函数 =========
@st.cache_data
def load_audio(file):
    return librosa.load(file, sr=None)


@st.cache_data
def generate_spectrogram_image(y, sr):
    fig, ax = plt.subplots(figsize=(5, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="Spectrogram (dB)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


@st.cache_data
def generate_waveform_image(y, sr):
    fig, ax = plt.subplots(figsize=(5, 3))
    librosa.display.waveshow(y, sr=sr)
    ax.set(title="Waveform")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def is_fully_annotated(file):
    info = st.session_state.segment_info.get(file.name)
    if info is None:
        return False
    return info["current_seg"] >= info["total_seg"]


# ======== Session 状态初始化 =========
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "segment_info" not in st.session_state:
    st.session_state.segment_info = {}
if "last_audio_file" not in st.session_state:
    st.session_state.last_audio_file = None
if "last_seg_idx" not in st.session_state:
    st.session_state.last_seg_idx = -1
if "auto_play" not in st.session_state:
    st.session_state.auto_play = True  # 自动播放开关
if "audio_muted" not in st.session_state:
    st.session_state.audio_muted = True  # 音频默认静音


st.set_page_config(layout="wide")


st.title("🐸 青蛙音频标注工具")


# ======== 侧边栏 =========
with st.sidebar:
    uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True)
    output_dir = st.text_input("保存目录", "E:/Frog audio classification/uploaded_audios")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, encoding="utf-8")
    else:
        df_old = pd.DataFrame(columns=["filename", "segment_index", "start_time", "end_time", "labels"])


    # 下载区域
    st.markdown("### 📥 下载标注结果")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                label="📄 下载标注CSV文件",
                data=f,
                file_name="annotations.csv",
                mime="text/csv"
            )


    # 音频片段下载
    annotated_paths = []
    if os.path.exists(csv_path):
        df_tmp = pd.read_csv(csv_path)
        for fname in df_tmp["segment_index"]:
            full_path = os.path.join(output_dir, fname)
            if os.path.exists(full_path):
                annotated_paths.append(full_path)


    if annotated_paths:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for path in annotated_paths:
                arcname = os.path.basename(path)
                zip_file.write(path, arcname=arcname)
        zip_buffer.seek(0)
        st.download_button(
            label="🎵 下载标注音频 (ZIP)",
            data=zip_buffer,
            file_name="annotated_audio_segments.zip",
            mime="application/zip"
        )


    # 自动播放开关
    st.session_state.auto_play = st.checkbox("自动播放音频", value=st.session_state.auto_play)
    
    # 音频静音/取消静音
    if st.button("🔇 取消静音" if st.session_state.audio_muted else "🔊 静音"):
        st.session_state.audio_muted = not st.session_state.audio_muted


    # 标注状态显示
    if uploaded_files:
        with st.expander("✅ 已标注音频", expanded=True):
            for f in uploaded_files:
                if f.name in st.session_state.processed_files:
                    st.write(f.name)
        with st.expander("🕓 未标注音频", expanded=True):
            st.write([f.name for f in uploaded_files if f.name not in st.session_state.processed_files])


# ======== 主处理区域 =========
SEGMENT_DURATION = 5.0  # 每段时长（秒）


if uploaded_files:
    unprocessed = [f for f in uploaded_files if not is_fully_annotated(f)]


    if st.session_state.current_index < len(unprocessed):
        audio_file = unprocessed[st.session_state.current_index]
        y, sr = load_audio(audio_file)
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / SEGMENT_DURATION))


        if audio_file.name not in st.session_state.segment_info:
            st.session_state.segment_info[audio_file.name] = {"current_seg": 0, "total_seg": total_segments}


        seg_info = st.session_state.segment_info[audio_file.name]
        seg_idx = seg_info["current_seg"]


        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")


        # 计算当前段落的时间范围
        start_sec = seg_idx * SEGMENT_DURATION
        end_sec = min((seg_idx + 1) * SEGMENT_DURATION, total_duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]


        # 布局调整：左侧音频信息，右侧标签和操作
        col_main, col_labels = st.columns([3, 1])


        with col_main:
            # 播放音频段
            st.subheader("🎧 播放当前音频片段")
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            
            # 生成随机ID，确保每次加载新音频时ID不同
            audio_id = f"audio_{uuid.uuid4()}"
            
            # 使用HTML音频元素，支持自动播放和静音属性
            if st.session_state.auto_play:
