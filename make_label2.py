import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import os
import io
import zipfile
from io import BytesIO


# ======== 工具函数 =========
@st.cache_data  # 缓存音频加载结果，避免重运行时重复加载
def load_audio(file):
    return librosa.load(file, sr=None)


@st.cache_data  # 缓存图表生成结果
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


@st.cache_data  # 缓存图表生成结果
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
if "dynamic_species_list" not in st.session_state:
    st.session_state["dynamic_species_list"] = [
        "北方狭口蛙", "黑斑侧褶蛙", "金线蛙", "牛蛙", "饰纹姬蛙", "中华蟾蜍", "泽蛙", "其他"
    ]
if "current_selected_labels" not in st.session_state:
    st.session_state.current_selected_labels = set()
# 新增：标记是否需要更新标签，避免全页面重运行
if "needs_label_update" not in st.session_state:
    st.session_state.needs_label_update = False


st.set_page_config(layout="wide")
st.title("青蛙音频标注工具")


# ======== 标签更新函数（核心优化） =========
def update_labels(label_file):
    """处理标签文件并更新，不触发全页面重运行"""
    try:
        content = label_file.read().decode("utf-8")
        species_list = [line.strip() for line in content.split("\n") if line.strip()]
        if species_list:
            st.session_state["dynamic_species_list"] = species_list
            st.session_state.needs_label_update = True  # 标记需要更新标签显示
            return True, f"成功加载 {len(species_list)} 个标签！"
        else:
            return False, "标签文件为空，请检查内容"
    except UnicodeDecodeError:
        return False, "标签文件编码错误，请使用UTF-8"
    except Exception as e:
        return False, f"读取失败：{str(e)}"


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

    # 标签设置（优化后）
    st.markdown("### 🏷️ 标签设置")
    # 使用回调函数处理标签上传，避免立即重运行
    label_file = st.file_uploader(
        "上传标签文件（每行一个标签）", 
        type=["txt"], 
        key="label_file_uploader",
        on_change=lambda: update_labels(label_file) if label_file else None
    )

    # 显示标签更新结果（即时反馈）
    if label_file:
        success, msg = update_labels(label_file)
        if success:
            st.success(msg)
        else:
            st.error(msg)

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
    if os.path.exists(csv_path) and "segment_index" in pd.read_csv(csv_path).columns:
        df_tmp = pd.read_csv(csv_path)
        for idx, row in df_tmp.iterrows():
            try:
                fname = str(row["segment_index"])
                if pd.notna(fname) and fname.strip():
                    full_path = os.path.join(output_dir, fname)
                    if os.path.exists(full_path):
                        annotated_paths.append(full_path)
            except Exception as e:
                st.warning(f"处理路径时出错: {str(e)}")

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


# ======== 主处理区域 =========
SEGMENT_DURATION = 5.0  # 每段时长（秒）


if uploaded_files:
    unprocessed = [f for f in uploaded_files if not is_fully_annotated(f)]

    if st.session_state.current_index < len(unprocessed):
        audio_file = unprocessed[st.session_state.current_index]
        # 仅在首次加载或音频文件变化时才重新加载（利用缓存减少重复计算）
        if (st.session_state.last_audio_file != audio_file.name or 
            f"audio_{audio_file.name}" not in st.session_state):
            y, sr = load_audio(audio_file)
            st.session_state[f"audio_{audio_file.name}"] = (y, sr)  # 缓存音频数据
        else:
            y, sr = st.session_state[f"audio_{audio_file.name}"]  # 直接使用缓存

        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / SEGMENT_DURATION))

        if audio_file.name not in st.session_state.segment_info:
            st.session_state.segment_info[audio_file.name] = {"current_seg": 0, "total_seg": total_segments}
        seg_info = st.session_state.segment_info[audio_file.name]
        seg_idx = seg_info["current_seg"]

        # 切换片段时重置选中标签
        current_segment_key = f"{audio_file.name}_{seg_idx}"
        if (st.session_state.last_audio_file != audio_file.name
                or st.session_state.last_seg_idx != seg_idx):
            st.session_state.current_selected_labels = set()
            st.session_state.last_audio_file = audio_file.name
            st.session_state.last_seg_idx = seg_idx

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
            st.subheader("🎧 播放当前音频片段")
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            st.audio(audio_bytes, format="audio/wav", start_time=0)

            # 波形图和频谱图（使用缓存减少重绘时间）
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 波形图")
                if f"wave_{current_segment_key}" not in st.session_state:
                    st.session_state[f"wave_{current_segment_key}"] = generate_waveform_image(segment_y, sr)
                st.image(st.session_state[f"wave_{current_segment_key}"], caption="Waveform", use_container_width=True)
            with col2:
                st.markdown("#### 🎞️ 频谱图")
                if f"spec_{current_segment_key}" not in st.session_state:
                    st.session_state[f"spec_{current_segment_key}"] = generate_spectrogram_image(segment_y, sr)
                st.image(st.session_state[f"spec_{current_segment_key}"], caption="Spectrogram (dB)", use_container_width=True)

        # 获取标签列表（若有更新则立即生效）
        species_list = st.session_state["dynamic_species_list"]
        with col_labels:
            st.markdown("### 物种标签（可多选）")
            current_key_prefix = current_segment_key

            # 标签搜索
            search_query = st.text_input("🔍 搜索标签", value="", key=f"search_{current_key_prefix}")
            filtered_species = [label for label in species_list if search_query.lower() in label.lower()]

            # 显示已选标签
            st.info(f"已选标签数：{len(st.session_state.current_selected_labels)}")
            if st.session_state.current_selected_labels:
                st.success(f"已选标签: {', '.join(st.session_state.current_selected_labels)}")
            else:
                st.info("请选择至少一个标签")

            # 渲染标签复选框
            for label in filtered_species:
                key = f"label_{label}_{current_key_prefix}"
                is_selected = label in st.session_state.current_selected_labels
                if st.checkbox(label, key=key, value=is_selected):
                    st.session_state.current_selected_labels.add(label)
                else:
                    st.session_state.current_selected_labels.discard(label)

            # 操作按钮
            st.markdown("### 🛠️ 操作")
            col_save, col_skip = st.columns(2)
            with col_save:
                if st.button("保存本段标注", key=f"save_{current_key_prefix}"):
                    selected_labels = list(st.session_state.current_selected_labels)
                    if not selected_labels:
                        st.warning("❗请先选择至少一个标签！")
                    else:
                        try:
                            segment_filename = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}.wav"
                            segment_path = os.path.join(output_dir, segment_filename)
                            sf.write(segment_path, segment_y, sr)

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

                            if seg_idx + 1 < total_segments:
                                st.session_state.segment_info[audio_file.name]["current_seg"] += 1
                            else:
                                st.session_state.processed_files.add(audio_file.name)
                                st.session_state.current_index += 1

                            st.success("标注已保存！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败：{str(e)}")

            with col_skip:
                if st.button("跳过本段", key=f"skip_{current_key_prefix}"):
                    if seg_idx + 1 < total_segments:
                        st.session_state.segment_info[audio_file.name]["current_seg"] += 1
                    else:
                        st.session_state.processed_files.add(audio_file.name)
                        st.session_state.current_index += 1
                    st.rerun()

    else:
        st.success("🎉 所有上传的音频都已标注完成！")

else:
    st.info("请先在左侧上传至少一个音频文件")

# 若标签需要更新，仅刷新标签区域（局部更新）
if st.session_state.needs_label_update:
    st.session_state.needs_label_update = False
    st.experimental_rerun()  # 轻量级重运行，仅更新必要部分
