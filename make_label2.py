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
if "segment_info" not in st.session_state:
    st.session_state.segment_info = {}
if "last_audio_file" not in st.session_state:
    st.session_state.last_audio_file = None
if "last_seg_idx" not in st.session_state:
    st.session_state.last_seg_idx = -1
# 新增：标签相关状态（从优化版本整合）
if "dynamic_species_list" not in st.session_state:
    st.session_state["dynamic_species_list"] = [
        "北方狭口蛙", "黑斑侧褶蛙", "金线蛙", "牛蛙", "饰纹姬蛙", "中华蟾蜍", "泽蛙", "其他"  # 默认标签，可被上传文件覆盖
    ]
if "current_selected_labels" not in st.session_state:
    st.session_state.current_selected_labels = set()


st.set_page_config(layout="wide")
st.title("青蛙音频标注工具")


# ======== 标签管理组件（从优化版本整合，确保不影响保存功能） =========
def label_management_component():
    """侧边栏标签上传组件，动态更新标签列表"""
    with st.sidebar:
        st.markdown("### 🏷️ 标签设置")
        # 使用表单避免频繁重运行，确保保存功能稳定
        with st.form("label_form", clear_on_submit=True):
            label_file = st.file_uploader("上传标签文件（每行一个）", type=["txt"], key="label_file")
            submit_label = st.form_submit_button("加载标签")
            
            if submit_label and label_file:
                try:
                    # 读取并解码标签文件
                    content = label_file.read().decode("utf-8")
                    species_list = [line.strip() for line in content.split("\n") if line.strip()]
                    if species_list:
                        st.session_state["dynamic_species_list"] = species_list
                        st.success(f"成功加载 {len(species_list)} 个标签！")
                        st.rerun()  # 刷新页面使标签生效
                    else:
                        st.error("标签文件为空，请检查内容")
                except UnicodeDecodeError:
                    st.error("标签文件编码错误，请使用UTF-8编码")
                except Exception as e:
                    st.error(f"标签文件读取失败：{str(e)}")


# ======== 侧边栏（整合保存版本和优化版本） =========
with st.sidebar:
    # 先加载标签管理组件（新增）
    label_management_component()
    
    # 保留原版本的音频上传和输出设置（确保保存功能稳定）
    uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True)
    output_dir = "E:/Frog audio classification/uploaded_audios"  # 保留原路径，确保保存逻辑不变
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, encoding="utf-8")
    else:
        df_old = pd.DataFrame(columns=["filename", "segment_index", "start_time", "end_time", "labels"])

    # 下载区域（保留原版本，确保稳定）
    st.markdown("### 📥 下载标注结果")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                label="📄 下载标注CSV文件",
                data=f,
                file_name="annotations.csv",
                mime="text/csv"
            )

    # 音频片段下载（保留原版本的稳定逻辑）
    annotated_paths = []
    if os.path.exists(csv_path):
        df_tmp = pd.read_csv(csv_path)
        if "segment_index" in df_tmp.columns:
            for idx, row in df_tmp.iterrows():
                try:
                    fname = str(row["segment_index"])
                    if pd.notna(fname) and fname.strip() != "":
                        full_path = os.path.join(output_dir, fname)
                        if os.path.exists(full_path):
                            annotated_paths.append(full_path)
                except Exception as e:
                    st.warning(f"处理音频片段路径时出错: {str(e)}")
        else:
            st.warning("CSV文件中缺少 'segment_index' 列，无法生成音频下载包")

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

    # 标注状态显示（保留原版本）
    if uploaded_files:
        with st.expander("✅ 已标注音频", expanded=True):
            for f in uploaded_files:
                if f.name in st.session_state.processed_files:
                    st.write(f.name)
        with st.expander("🕓 未标注音频", expanded=True):
            st.write([f.name for f in uploaded_files if f.name not in st.session_state.processed_files])


# ======== 主处理区域（以原版本为基础，整合标签功能） =========
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

        # 计算当前片段的时间范围（保留原版本逻辑，确保保存正确）
        start_sec = seg_idx * SEGMENT_DURATION
        end_sec = min((seg_idx + 1) * SEGMENT_DURATION, total_duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]

        # 布局：左侧音频信息，右侧标签（整合优化版本的布局）
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            # 播放音频段（保留原版本稳定逻辑）
            st.subheader("🎧 播放当前音频片段")
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            st.audio(audio_bytes, format="audio/wav", start_time=0)

            # 波形图 + 频谱图（保留原版本）
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 波形图")
                wave_img = generate_waveform_image(segment_y, sr)
                st.image(wave_img, caption="Waveform", use_container_width=True)
            with col2:
                st.markdown("#### 🎞️ 频谱图")
                spec_img = generate_spectrogram_image(segment_y, sr)
                st.image(spec_img, caption="Spectrogram (dB)", use_container_width=True)

        with col_labels:  # 右侧标签区域，使用动态标签列表
            st.markdown("### 物种标签（可多选）")
            species_list = st.session_state["dynamic_species_list"]  # 从session获取动态标签
            current_key_prefix = f"{audio_file.name}_{seg_idx}"

            # 切换片段时重置复选框状态（整合优化版本的状态管理）
            if (st.session_state.last_audio_file != audio_file.name
                    or st.session_state.last_seg_idx != seg_idx):
                st.session_state.current_selected_labels = set()  # 重置已选标签
                st.session_state.last_audio_file = audio_file.name
                st.session_state.last_seg_idx = seg_idx

            # 渲染复选框并收集选中的标签（使用动态标签列表）
            selected_labels = []
            for label in species_list:
                key = f"label_checkbox_{label}_{current_key_prefix}"
                if key not in st.session_state:
                    st.session_state[key] = False
                checked = st.checkbox(label, key=key, value=st.session_state[key])
                if checked != st.session_state[key]:
                    st.session_state[key] = checked
                if st.session_state[key]:
                    selected_labels.append(label)

            # 显示已选标签（优化显示位置，保留原版本的提示逻辑）
            if selected_labels:
                st.success(f"已选标签: {', '.join(selected_labels)}")
            else:
                st.info("请选择至少一个标签")

            # 操作按钮（保留原版本的稳定逻辑）
            col_save, col_skip = st.columns(2)
            with col_save:
                save_clicked = st.button("保存本段标注", key=f"save_btn_{current_key_prefix}")
            with col_skip:
                skip_clicked = st.button("跳过本段", key=f"skip_btn_{current_key_prefix}")

        # 保存逻辑（完全保留原版本，确保稳定）
        if save_clicked:
            if not selected_labels:
                st.warning("❗请先选择至少一个物种标签！")
            else:
                # 保存分片音频（原版本验证有效的逻辑）
                segment_filename = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}.wav"
                segment_path = os.path.join(output_dir, segment_filename)
                sf.write(segment_path, segment_y, sr)

                # 保存到CSV（原版本验证有效的逻辑）
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

                # 切换分片或下一个文件（原版本逻辑）
                if seg_idx + 1 < total_segments:
                    st.session_state.segment_info[audio_file.name]["current_seg"] += 1
                else:
                    st.session_state.processed_files.add(audio_file.name)
                    st.session_state.current_index += 1

                st.success("标注已保存！")
                st.rerun()

        if skip_clicked:  # 保留原版本的跳过逻辑
            if seg_idx + 1 < total_segments:
                st.session_state.segment_info[audio_file.name]["current_seg"] += 1
            else:
                st.session_state.processed_files.add(audio_file.name)
                st.session_state.current_index += 1
            st.rerun()

    # 检查是否所有音频都已标注完成（保留原版本逻辑）
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
