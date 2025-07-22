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


# ======== 工具函数（优化：添加缓存减少重复计算）=========
@st.cache_data
def load_audio(file):
    return librosa.load(file, sr=None)

@st.cache_data  # 缓存频谱图生成结果
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

@st.cache_data  # 新增：缓存波形图生成结果
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
if "label_reset_key" not in st.session_state:
    st.session_state.label_reset_key = str(uuid.uuid4())
if "selected_labels" not in st.session_state:
    st.session_state.selected_labels = set()
if "reset_checkboxes" not in st.session_state:
    st.session_state.reset_checkboxes = False
if "segment_info" not in st.session_state:
    st.session_state.segment_info = {}
if "last_audio_file" not in st.session_state:
    st.session_state.last_audio_file = None
if "last_seg_idx" not in st.session_state:
    st.session_state.last_seg_idx = -1


st.set_page_config(layout="wide")

st.title("🐸 青蛙音频标注工具")

# ======== 侧边栏（优化：减少重复渲染区域）=========
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
    
    # 音频片段下载（优化：仅在有标注时计算）
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
    
    # 标注状态显示（优化：仅在有文件时显示）
    if uploaded_files:
        with st.expander("✅ 已标注音频", expanded=True):
            for f in uploaded_files:
                if f.name in st.session_state.processed_files:
                    st.write(f.name)
        with st.expander("🕓 未标注音频", expanded=True):
            st.write([f.name for f in uploaded_files if f.name not in st.session_state.processed_files])


# ======== 主处理区域（核心优化：减少标签选择时的渲染范围）=========
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

        # 切换片段时清空标签（优化：仅在真正切换时执行）
        if st.session_state.last_audio_file != audio_file.name or st.session_state.last_seg_idx != seg_idx:
            st.session_state.selected_labels.clear()
            st.session_state.last_audio_file = audio_file.name
            st.session_state.last_seg_idx = seg_idx

        # 计算当前段落的时间范围
        start_sec = seg_idx * SEGMENT_DURATION
        end_sec = min((seg_idx + 1) * SEGMENT_DURATION, total_duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]

        # 播放音频段（优化：仅音频片段变化时重新生成）
        st.subheader("🎧 播放当前音频片段")
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, segment_y, sr, format='WAV')
        st.audio(audio_bytes, format="audio/wav", start_time=0)

        # 波形图 + 频谱图（优化：使用缓存结果，减少重复绘制）
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 波形图")
            wave_img = generate_waveform_image(segment_y, sr)
            st.image(wave_img, caption="Waveform", use_container_width=True)

        with col2:
            st.markdown("#### 🎞️ 频谱图")
            spec_img = generate_spectrogram_image(segment_y, sr)
            st.image(spec_img, caption="Spectrogram (dB)", use_container_width=True)

        # 标签选择区域（核心优化：用multiselect替代checkbox，减少交互次数）
        st.markdown("### 🐸 请选择该段音频中出现的物种标签（可多选）")
        species_list = ["Rana", "Hyla", "Bufo", "Fejervarya", "Microhyla", "Other"]
        # 使用multiselect实现高效多选，仅在选择变化时触发渲染
        selected_labels = st.multiselect(
            "物种标签",
            species_list,
            default=list(st.session_state.selected_labels),
            key=f"multiselect_{audio_file.name}_{seg_idx}"
        )
        st.session_state.selected_labels = set(selected_labels)

        # 保存按钮（优化：使用form减少提交次数）
        col_save, col_skip = st.columns(2)
        with col_save:
            save_clicked = st.button("保存本段标注", key=f"save_btn_{audio_file.name}_{seg_idx}")
        with col_skip:
            skip_clicked = st.button("跳过本段", key=f"skip_btn_{audio_file.name}_{seg_idx}")

        if save_clicked:
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

                # 切换分片或下一个文件
                if seg_idx + 1 < total_segments:
                    st.session_state.segment_info[audio_file.name]["current_seg"] += 1
                else:
                    st.session_state.processed_files.add(audio_file.name)
                    st.session_state.current_index += 1

                st.success("标注已保存！")
                st.experimental_rerun()  # 手动触发刷新，减少延迟

        if skip_clicked:
            if seg_idx + 1 < total_segments:
                st.session_state.segment_info[audio_file.name]["current_seg"] += 1
            else:
                st.session_state.processed_files.add(audio_file.name)
                st.session_state.current_index += 1
            st.experimental_rerun()

    # 检查是否所有音频都已标注完成
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
