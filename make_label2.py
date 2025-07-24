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
# 初始化动态标签列表（关键修复：确保默认标签存在）
if "dynamic_species_list" not in st.session_state:
    st.session_state["dynamic_species_list"] = [
        "北方狭口蛙", "黑斑侧褶蛙", "金线蛙", "牛蛙", "饰纹姬蛙", "中华蟾蜍", "泽蛙", "其他"
    ]


st.set_page_config(layout="wide")
st.title("青蛙音频标注工具")


# ======== 侧边栏 =========
with st.sidebar:
    uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True)
    output_dir = "E:/Frog audio classification/uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path, encoding="utf-8")
    else:
        df_old = pd.DataFrame(columns=["filename", "segment_index", "start_time", "end_time", "labels"])

    # 上传标签文件（核心修复：优化标签读取和状态更新）
    st.markdown("### 🏷️ 标签设置")
    label_file = st.file_uploader("上传标签文件（每行一个标签）", type=["txt"], key="label_file_uploader")

    if label_file is not None:  # 检测到新的标签文件上传
        try:
            # 读取并解码标签（支持中文）
            content = label_file.read().decode("utf-8")  # 一次性读取所有内容并解码
            species_list = [line.strip() for line in content.split("\n") if line.strip()]  # 按行分割
            if species_list:  # 确保标签列表不为空
                st.session_state["dynamic_species_list"] = species_list
                st.success(f"成功加载 {len(species_list)} 个标签！")
                # 刷新页面使标签生效（使用st.rerun而非实验性方法）
                st.rerun()
            else:
                st.warning("标签文件为空，请检查文件内容")
        except UnicodeDecodeError:
            st.error("标签文件编码错误，请使用UTF-8编码的TXT文件")
        except Exception as e:
            st.error(f"标签文件读取失败：{str(e)}")

    # 显示当前使用的标签列表（方便用户确认）
    st.markdown("#### 当前标签列表")
    st.write(st.session_state["dynamic_species_list"])

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

        # 布局：左侧音频信息，右侧标签
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            # 播放当前音频片段
            st.subheader("🎧 播放当前音频片段")
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            st.audio(audio_bytes, format="audio/wav", start_time=0)

            # 波形图 + 频谱图
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 波形图")
                wave_img = generate_waveform_image(segment_y, sr)
                st.image(wave_img, caption="Waveform", use_container_width=True)
            with col2:
                st.markdown("#### 🎞️ 频谱图")
                spec_img = generate_spectrogram_image(segment_y, sr)
                st.image(spec_img, caption="Spectrogram (dB)", use_container_width=True)

        # 关键修复：从session_state获取最新的标签列表
        species_list = st.session_state["dynamic_species_list"]
        with col_labels:
            st.markdown("### 物种标签（可多选）")
            current_key_prefix = f"{audio_file.name}_{seg_idx}"
            
            # 标签搜索功能
            search_query = st.text_input("🔍 搜索标签", value="", key=f"search_{current_key_prefix}")
            filtered_species = [label for label in species_list if search_query.lower() in label.lower()]

            # 切换片段时重置复选框状态
            if (st.session_state.last_audio_file != audio_file.name
                    or st.session_state.last_seg_idx != seg_idx):
                for label in species_list:
                    key = f"label_checkbox_{label}_{current_key_prefix}"
                    st.session_state[key] = False
                st.session_state.last_audio_file = audio_file.name
                st.session_state.last_seg_idx = seg_idx

            # 渲染筛选后的标签复选框
            selected_labels = []
            for label in filtered_species:
                key = f"label_checkbox_{label}_{current_key_prefix}"
                if key not in st.session_state:
                    st.session_state[key] = False
                # 保持复选框状态与session同步
                checked = st.checkbox(label, key=key, value=st.session_state[key])
                if checked != st.session_state[key]:
                    st.session_state[key] = checked
                if checked:
                    selected_labels.append(label)

            # 显示已选标签
            if selected_labels:
                st.success(f"已选标签: {', '.join(selected_labels)}")
            else:
                st.info("请选择至少一个标签")

            # 操作按钮
            col_save, col_skip = st.columns(2)
            with col_save:
                save_clicked = st.button("保存本段标注", key=f"save_btn_{current_key_prefix}")
            with col_skip:
                skip_clicked = st.button("跳过本段", key=f"skip_btn_{current_key_prefix}")

        # 保存逻辑
        if save_clicked:
            if not selected_labels:
                st.warning("❗请先选择至少一个物种标签！")
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

                st.success("标注已保存！")
                st.rerun()

        if skip_clicked:
            if seg_idx + 1 < total_segments:
                st.session_state.segment_info[audio_file.name]["current_seg"] += 1
            else:
                st.session_state.processed_files.add(audio_file.name)
                st.session_state.current_index += 1
            st.rerun()

    # 所有音频标注完成
    else:
        st.success("🎉 所有上传的音频都已标注完成！")

else:
    st.info("请先在左侧上传至少一个音频文件")
