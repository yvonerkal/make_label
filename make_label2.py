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
import hashlib  # 用于生成音频数据的唯一哈希值


# 工具函数优化：解决图表缓存和闪现问题
# ======== 工具函数 =========
@st.cache_data
def load_audio(file):
    # 为不同文件生成唯一缓存键
    file_key = hashlib.md5(file.getvalue()).hexdigest()
    return librosa.load(file, sr=None)


def generate_spectrogram_image(y, sr, unique_key):
    """生成频谱图，添加unique_key确保缓存唯一"""
    # 强制清除Matplotlib残留状态
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="Spectrogram (dB)")
    fig.tight_layout()
    
    # 使用BytesIO存储图表，避免文件系统操作
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)  # 确保关闭当前图表
    return Image.open(buf)


def generate_waveform_image(y, sr, unique_key):
    """生成波形图，添加unique_key确保缓存唯一"""
    # 强制清除Matplotlib残留状态
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 3))
    librosa.display.waveshow(y, sr=sr)
    ax.set(title="Waveform")
    fig.tight_layout()
    
    # 使用BytesIO存储图表，避免文件系统操作
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)  # 确保关闭当前图表
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
# 添加图表缓存状态，避免重复生成
if "plot_cache" not in st.session_state:
    st.session_state.plot_cache = {}


st.set_page_config(layout="wide")
st.title("青蛙音频标注工具")


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


    # 音频片段下载（优化错误处理）
    annotated_paths = []
    if os.path.exists(csv_path):
        df_tmp = pd.read_csv(csv_path)
        if "segment_index" in df_tmp.columns:
            for idx, row in df_tmp.iterrows():
                try:
                    fname = str(row["segment_index"])
                    if pd.notna(fname) and fname.strip():
                        full_path = os.path.join(output_dir, fname)
                        if os.path.exists(full_path):
                            annotated_paths.append(full_path)
                except Exception as e:
                    st.warning(f"处理路径时出错: {str(e)}")
        else:
            st.warning("CSV文件缺少 'segment_index' 列，无法生成音频包")


    if annotated_paths:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for path in annotated_paths:
                zip_file.write(path, os.path.basename(path))
        zip_buffer.seek(0)
        st.download_button(
            label="🎵 下载标注音频 (ZIP)",
            data=zip_buffer,
            file_name="annotated_audio_segments.zip",
            mime="application/zip"
        )


    # 标注状态显示
    if uploaded_files:
        with st.expander("✅ 已标注音频", expanded=True):
            st.write([f.name for f in uploaded_files if f.name in st.session_state.processed_files])
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

        # 初始化音频片段信息
        if audio_file.name not in st.session_state.segment_info:
            st.session_state.segment_info[audio_file.name] = {"current_seg": 0, "total_seg": total_segments}
        seg_info = st.session_state.segment_info[audio_file.name]
        seg_idx = seg_info["current_seg"]

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")

        # 计算当前片段的时间范围
        start_sec = seg_idx * SEGMENT_DURATION
        end_sec = min((seg_idx + 1) * SEGMENT_DURATION, total_duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]

        # 生成当前片段的唯一标识（用于图表缓存）
        segment_key = f"{audio_file.name}_{seg_idx}_{hashlib.md5(segment_y).hexdigest()}"

        # 布局：左侧音频信息，右侧标签
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            # 播放当前音频片段
            st.subheader("🎧 播放当前音频片段")
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            st.audio(audio_bytes, format="audio/wav", start_time=0)

            # 波形图和频谱图（使用缓存避免重复生成）
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 波形图")
                # 检查缓存，不存在则生成并缓存
                if f"wave_{segment_key}" not in st.session_state.plot_cache:
                    st.session_state.plot_cache[f"wave_{segment_key}"] = generate_waveform_image(segment_y, sr, segment_key)
                st.image(st.session_state.plot_cache[f"wave_{segment_key}"], caption="Waveform", use_container_width=True)

            with col2:
                st.markdown("#### 🎞️ 频谱图")
                if f"spec_{segment_key}" not in st.session_state.plot_cache:
                    st.session_state.plot_cache[f"spec_{segment_key}"] = generate_spectrogram_image(segment_y, sr, segment_key)
                st.image(st.session_state.plot_cache[f"spec_{segment_key}"], caption="Spectrogram (dB)", use_container_width=True)


        with col_labels:
            st.markdown("### 物种标签（可多选）")
            species_list = ["北方狭口蛙", "黑斑侧褶蛙", "金线蛙", "牛蛙", "饰纹姬蛙", "中华蟾蜍", "泽蛙", "其他"]
            current_key_prefix = f"{audio_file.name}_{seg_idx}"

            # 重置标签选择状态（仅当切换片段时）
            if (st.session_state.last_audio_file != audio_file.name or 
                st.session_state.last_seg_idx != seg_idx):
                for label in species_list:
                    st.session_state[f"label_{label}_{current_key_prefix}"] = False
                st.session_state.last_audio_file = audio_file.name
                st.session_state.last_seg_idx = seg_idx

            # 收集选中的标签
            selected_labels = []
            for label in species_list:
                key = f"label_{label}_{current_key_prefix}"
                if key not in st.session_state:
                    st.session_state[key] = False
                checked = st.checkbox(label, key=key, value=st.session_state[key])
                st.session_state[key] = checked
                if checked:
                    selected_labels.append(label)

            # 显示选中状态
            if selected_labels:
                st.success(f"已选标签: {', '.join(selected_labels)}")
            else:
                st.info("请选择至少一个标签")

            # 操作按钮
            col_save, col_skip = st.columns(2)
            with col_save:
                save_clicked = st.button("保存本段标注", key=f"save_{current_key_prefix}")
            with col_skip:
                skip_clicked = st.button("跳过本段", key=f"skip_{current_key_prefix}")


        # 保存逻辑
        if save_clicked:
            if not selected_labels:
                st.warning("❗请先选择至少一个标签！")
            else:
                # 保存音频片段
                segment_filename = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}.wav"
                segment_path = os.path.join(output_dir, segment_filename)
                sf.write(segment_path, segment_y, sr)

                # 保存标注到CSV
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

                # 切换到下一段或下一个文件
                if seg_idx + 1 < total_segments:
                    st.session_state.segment_info[audio_file.name]["current_seg"] += 1
                else:
                    st.session_state.processed_files.add(audio_file.name)
                    st.session_state.current_index += 1

                # 清除当前片段的图表缓存（避免残留）
                if f"wave_{segment_key}" in st.session_state.plot_cache:
                    del st.session_state.plot_cache[f"wave_{segment_key}"]
                if f"spec_{segment_key}" in st.session_state.plot_cache:
                    del st.session_state.plot_cache[f"spec_{segment_key}"]

                st.success("标注已保存！")
                st.rerun()


        # 跳过逻辑
        if skip_clicked:
            if seg_idx + 1 < total_segments:
                st.session_state.segment_info[audio_file.name]["current_seg"] += 1
            else:
                st.session_state.processed_files.add(audio_file.name)
                st.session_state.current_index += 1

            # 清除当前片段的图表缓存
            if f"wave_{segment_key}" in st.session_state.plot_cache:
                del st.session_state.plot_cache[f"wave_{segment_key}"]
            if f"spec_{segment_key}" in st.session_state.plot_cache:
                del st.session_state.plot_cache[f"spec_{segment_key}"]

            st.rerun()


    # 所有音频标注完成
    else:
        st.success("🎉 所有上传的音频已标注完成！")

else:
    st.info("请先在左侧上传音频文件 (.wav)")
