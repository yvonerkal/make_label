import streamlit as st
from streamlit_drawable_canvas import st_canvas
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


# ======== 工具函数 =========
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
if "label_reset_key" not in st.session_state:
    st.session_state.label_reset_key = str(uuid.uuid4())
if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = set()
# 添加重置checkbox的标志（关键修改）
if "reset_checkboxes" not in st.session_state:
    st.session_state.reset_checkboxes = False
if "segment_info" not in st.session_state:
    st.session_state.segment_info = {}

def toggle_label(label):
    if label in st.session_state.selected_labels:
        st.session_state.selected_labels.remove(label)
    else:
        st.session_state.selected_labels.add(label)

st.set_page_config(layout="wide")

st.title("🐸 青蛙音频标注工具")

# ======== 侧边栏 =========
uploaded_files = st.sidebar.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True)
output_dir = st.sidebar.text_input("保存目录", "E:/Frog audio classification/uploaded_audios")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "annotations.csv")
if os.path.exists(csv_path):
    df_old = pd.read_csv(csv_path, encoding="utf-8")
else:
    df_old = pd.DataFrame(columns=["filename", "segment_index", "start_time", "end_time", "labels"])
# ======== 下载标注结果和音频片段（添加到左侧边栏）=========
st.sidebar.markdown("### 📥 下载标注结果")

# 下载标注CSV文件
if os.path.exists(csv_path):
    with open(csv_path, "rb") as f:
        st.sidebar.download_button(
            label="📄 下载标注CSV文件",
            data=f,
            file_name="annotations.csv",
            mime="text/csv"
        )

# 下载所有标注后的音频片段（压缩成ZIP）
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

    st.sidebar.download_button(
        label="🎵 下载标注音频 (ZIP)",
        data=zip_buffer,
        file_name="annotated_audio_segments.zip",
        mime="application/zip"
    )


# 已标注 / 未标注 显示
if uploaded_files:
    with st.sidebar.expander("✅ 已标注音频", expanded=True):
        for f in uploaded_files:
            if f.name in st.session_state.processed_files:
                st.write(f.name)

    with st.sidebar.expander("🕓 未标注音频", expanded=True):
        for f in uploaded_files:
            if f.name not in st.session_state.processed_files:
                st.write(f.name)

# ======== 主处理区域 =========
SEGMENT_DURATION = 5.0  # 每段时长（秒）

if uploaded_files:
    current_key_prefix = f"{audio_file.name}_seg{seg_idx}"

    unprocessed = [f for f in uploaded_files if not is_fully_annotated(f)]

    if "segment_info" not in st.session_state:
        st.session_state.segment_info = {}


    if st.session_state.current_index < len(unprocessed):
        audio_file = unprocessed[st.session_state.current_index]
        y, sr = librosa.load(audio_file, sr=None)
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / SEGMENT_DURATION))

        if audio_file.name not in st.session_state.segment_info:
            st.session_state.segment_info[audio_file.name] = {"current_seg": 0, "total_seg": total_segments}

        seg_info = st.session_state.segment_info[audio_file.name]
        seg_idx = seg_info["current_seg"]

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx+1}/{total_segments} 段")

        # 检查是否切换片段，清空选中标签
        last_audio = st.session_state.get("last_audio_file", None)
        last_seg = st.session_state.get("last_seg_idx", None)
        if last_audio != audio_file.name or last_seg != seg_idx:
            st.session_state.selected_labels.clear()
            st.session_state.last_audio_file = audio_file.name
            st.session_state.last_seg_idx = seg_idx

        # 计算当前段落的时间范围
        start_sec = seg_idx * SEGMENT_DURATION
        end_sec = min((seg_idx + 1) * SEGMENT_DURATION, total_duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]

        # 播放音频段
        st.subheader("🎧 播放当前音频片段")
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, segment_y, sr, format='WAV')
        st.audio(audio_bytes, format="audio/wav", start_time=0)

        # 波形图 + 频谱图

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 波形图")
            fig, ax = plt.subplots(figsize=(5, 3))
            librosa.display.waveshow(segment_y, sr=sr)
            ax.set(title="Waveform")
            st.pyplot(fig)

        with col2:
            st.markdown("#### 🎞️ 频谱图")
            img = generate_spectrogram_image(segment_y, sr)
            st.image(img, caption="Spectrogram (dB)", use_container_width=True)

        # 标签选择区域
        st.markdown("### 🐸 请选择该段音频中出现的物种标签（可多选）")
        species_list = ["Rana", "Hyla", "Bufo", "Fejervarya", "Microhyla", "Other"]
        cols = st.columns(len(species_list))

        # 关键修改：在实例化checkbox前重置状态
        current_key_prefix = f"{audio_file.name}_{seg_idx}"
        if st.session_state.reset_checkboxes:
            for label in species_list:
                key = f"label_checkbox_{label}_{current_key_prefix}"
                st.session_state[key] = False  # 在checkbox实例化前重置
            st.session_state.reset_checkboxes = False  # 重置标志

        # 创建checkbox
        for i, label in enumerate(species_list):
            key = f"label_checkbox_{label}_{current_key_prefix}"
            if key not in st.session_state:  # 初始化状态（首次创建时）
                st.session_state[key] = False
            # 实例化checkbox（使用当前session_state值）
            checked = cols[i].checkbox(label, key=key)

            # 美化按钮样式
            button_color = "#4CAF50" if checked else "#E0E0E0"
            text_color = "white" if checked else "black"
            cols[i].markdown(
                f"""
                <style>
                div[data-testid="stButton"] > button:nth-child(1) {{
                    background-color: {button_color};
                    color: {text_color};
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

        # 保存按钮逻辑
        # ======== 保存按钮逻辑（优化版）========
save_clicked = st.button("保存本段标注", key=f"save_btn_{current_key_prefix}")

if save_clicked:
    selected_labels = [
        label for label in species_list
        if st.session_state.get(f"label_checkbox_{label}_{current_key_prefix}", False)
    ]

    if not selected_labels:
        st.warning("❗请先选择至少一个物种标签！")
    else:
        # 保存分片音频
        segment_filename = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        sf.write(segment_path, segment_y, sr)

        # 保存到CSV
        entry = {
            "filename": audio_file.name,
            "segment_index": segment_filename,
            "start_time": round(start_sec, 3),
            "end_time": round(end_sec, 3),
            "labels": ",".join(selected_labels)
        }

        st.session_state.annotations.append(entry)
        df_combined = pd.concat([df_old, pd.DataFrame([entry])], ignore_index=True)
        df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 清除checkbox（刷新逻辑交由下次加载处理）
        for label in species_list:
            cb_key = f"label_checkbox_{label}_{current_key_prefix}"
            if cb_key in st.session_state:
                del st.session_state[cb_key]

        # 切换分片或下一个文件
        if seg_idx + 1 < total_segments:
            st.session_state.segment_info[audio_file.name]["current_seg"] += 1
        else:
            st.session_state.processed_files.add(audio_file.name)
            st.session_state.current_index += 1

        # 稳定刷新页面
        st.rerun()



    # 检查是否所有音频的所有片段都已标注完成

    all_done = True

    for f in uploaded_files:

        info = st.session_state.segment_info.get(f.name)

        if info is None or info["current_seg"] < info["total_seg"]:
            all_done = False

            break

    if all_done:
        st.success("🎉 所有上传的音频都已标注完成！")

else:
    st.info("请先在左侧上传至少一个音频文件")
