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
if "dynamic_species_list" not in st.session_state:
    st.session_state["dynamic_species_list"] = [
        "北方狭口蛙", "黑斑侧褶蛙", "金线蛙", "牛蛙", "饰纹姬蛙", "中华蟾蜍", "泽蛙", "其他"
    ]
if "current_selected_labels" not in st.session_state:
    st.session_state.current_selected_labels = set()
# 标记当前标签区域是否需要重新渲染
if "refresh_label_area" not in st.session_state:
    st.session_state.refresh_label_area = False


st.set_page_config(layout="wide")
st.title("青蛙音频标注工具")


# ======== 标签处理函数 =========
def process_label_file(label_file):
    """处理标签文件并更新状态，不触发全页面重运行"""
    try:
        content = label_file.read().decode("utf-8")
        species_list = [line.strip() for line in content.split("\n") if line.strip()]
        if species_list:
            st.session_state["dynamic_species_list"] = species_list
            st.session_state.refresh_label_area = True  # 仅标记标签区域需要刷新
            return True, f"加载成功：{len(species_list)}个标签"
        else:
            return False, "标签文件为空"
    except Exception as e:
        return False, f"错误：{str(e)}"


# ======== 侧边栏 =========
with st.sidebar:
    uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True)
    output_dir = "E:/Frog audio classification/uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")
    df_old = pd.read_csv(csv_path, encoding="utf-8") if os.path.exists(csv_path) else pd.DataFrame(
        columns=["filename", "segment_index", "start_time", "end_time", "labels"]
    )

    # 标签上传（核心优化：仅处理标签，不触发音频重处理）
    st.markdown("### 🏷️ 标签设置")
    label_file = st.file_uploader(
        "上传标签文件（每行一个）", 
        type=["txt"], 
        key="label_uploader",
        on_change=lambda: process_label_file(label_file) if label_file else None
    )

    # 显示标签上传状态
    if label_file:
        success, msg = process_label_file(label_file)
        st.success(msg) if success else st.error(msg)

    # 显示当前标签列表（快速预览）
    st.markdown("#### 当前标签预览")
    st.write(st.session_state["dynamic_species_list"][:5] + (["..."] if len(st.session_state["dynamic_species_list"]) > 5 else []))

    # 下载区域（保持不变）
    st.markdown("### 📥 下载结果")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button("📄 下载CSV", f, "annotations.csv", "text/csv")

    annotated_paths = []
    if os.path.exists(csv_path) and "segment_index" in pd.read_csv(csv_path).columns:
        for _, row in pd.read_csv(csv_path).iterrows():
            try:
                fname = str(row["segment_index"])
                if fname and os.path.exists(os.path.join(output_dir, fname)):
                    annotated_paths.append(os.path.join(output_dir, fname))
            except:
                pass
    if annotated_paths:
        with zipfile.ZipFile(zip_buf := BytesIO(), "w") as zf:
            for p in annotated_paths:
                zf.write(p, os.path.basename(p))
        zip_buf.seek(0)
        st.download_button("🎵 下载音频片段", zip_buf, "annotated_segments.zip", "application/zip")


# ======== 主处理区域 =========
if uploaded_files:
    unprocessed = [f for f in uploaded_files if not is_fully_annotated(f)]

    if st.session_state.current_index < len(unprocessed):
        audio_file = unprocessed[st.session_state.current_index]
        # 音频数据缓存（避免重复加载）
        audio_cache_key = f"audio_{audio_file.name}"
        if audio_cache_key not in st.session_state:
            st.session_state[audio_cache_key] = load_audio(audio_file)
        y, sr = st.session_state[audio_cache_key]

        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / 5.0))
        seg_idx = st.session_state.segment_info.get(audio_file.name, {"current_seg": 0})["current_seg"]
        current_segment_key = f"{audio_file.name}_{seg_idx}"

        # 切换片段时重置选中标签
        if (st.session_state.last_audio_file != audio_file.name or 
            st.session_state.last_seg_idx != seg_idx):
            st.session_state.current_selected_labels = set()
            st.session_state.last_audio_file = audio_file.name
            st.session_state.last_seg_idx = seg_idx
            st.session_state.refresh_label_area = True  # 切换片段时刷新标签区域

        st.header(f"标注：{audio_file.name}（第 {seg_idx+1}/{total_segments} 段）")

        # 左侧：音频和图表（不随标签更新重渲染）
        col_main, col_labels = st.columns([3, 1])
        with col_main:
            st.subheader("🎧 播放片段")
            audio_bytes = BytesIO()
            sf.write(audio_bytes, y[int(seg_idx*5*sr):int(min((seg_idx+1)*5, total_duration)*sr)], sr, "WAV")
            st.audio(audio_bytes, "audio/wav")

            # 图表缓存（避免标签更新时重绘）
            wave_cache_key = f"wave_{current_segment_key}"
            spec_cache_key = f"spec_{current_segment_key}"
            if wave_cache_key not in st.session_state:
                st.session_state[wave_cache_key] = generate_waveform_image(
                    y[int(seg_idx*5*sr):int(min((seg_idx+1)*5, total_duration)*sr)], sr
                )
            if spec_cache_key not in st.session_state:
                st.session_state[spec_cache_key] = generate_spectrogram_image(
                    y[int(seg_idx*5*sr):int(min((seg_idx+1)*5, total_duration)*sr)], sr
                )

            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state[wave_cache_key], "波形图", use_container_width=True)
            with col2:
                st.image(st.session_state[spec_cache_key], "频谱图", use_container_width=True)

        # 右侧：标签区域（核心优化：用容器隔离，仅标签更新时重渲染）
        with col_labels:
            # 创建标签专用容器，仅当需要刷新时重渲染
            label_container = st.container()
            with label_container:
                if st.session_state.refresh_label_area:
                    st.session_state.refresh_label_area = False  # 重置标记

                    st.markdown("### 物种标签（可多选）")
                    search_query = st.text_input("🔍 搜索标签", "", key=f"search_{current_segment_key}")
                    filtered_labels = [
                        lbl for lbl in st.session_state["dynamic_species_list"] 
                        if search_query.lower() in lbl.lower()
                    ]

                    st.info(f"已选：{len(st.session_state.current_selected_labels)}个")
                    if st.session_state.current_selected_labels:
                        st.success(f"已选：{', '.join(st.session_state.current_selected_labels)}")

                    # 渲染标签复选框
                    for lbl in filtered_labels:
                        key = f"lbl_{lbl}_{current_segment_key}"
                        is_selected = lbl in st.session_state.current_selected_labels
                        if st.checkbox(lbl, key=key, value=is_selected):
                            st.session_state.current_selected_labels.add(lbl)
                        else:
                            st.session_state.current_selected_labels.discard(lbl)

                    # 操作按钮
                    st.markdown("### 🛠️ 操作")
                    col_save, col_skip = st.columns(2)
                    with col_save:
                        if st.button("保存本段", key=f"save_{current_segment_key}"):
                            if not st.session_state.current_selected_labels:
                                st.warning("请至少选一个标签！")
                            else:
                                try:
                                    seg_fn = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}.wav"
                                    sf.write(os.path.join(output_dir, seg_fn), 
                                            y[int(seg_idx*5*sr):int(min((seg_idx+1)*5, total_duration)*sr)], sr)
                                    df_old = pd.concat([df_old, pd.DataFrame([{
                                        "filename": audio_file.name,
                                        "segment_index": seg_fn,
                                        "start_time": round(seg_idx*5, 3),
                                        "end_time": round(min((seg_idx+1)*5, total_duration), 3),
                                        "labels": ",".join(st.session_state.current_selected_labels)
                                    }])], ignore_index=True)
                                    df_old.to_csv(csv_path, index=False, encoding="utf-8-sig")
                                    if seg_idx + 1 < total_segments:
                                        st.session_state.segment_info[audio_file.name]["current_seg"] += 1
                                    else:
                                        st.session_state.processed_files.add(audio_file.name)
                                        st.session_state.current_index += 1
                                    st.success("保存成功！")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"保存失败：{e}")
                    with col_skip:
                        if st.button("跳过本段", key=f"skip_{current_segment_key}"):
                            if seg_idx + 1 < total_segments:
                                st.session_state.segment_info[audio_file.name]["current_seg"] += 1
                            else:
                                st.session_state.processed_files.add(audio_file.name)
                                st.session_state.current_index += 1
                            st.rerun()

    else:
        st.success("🎉 所有音频已标注完成！")

else:
    st.info("请先上传.wav音频文件")
