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
@st.cache_data(show_spinner=False)
def load_audio(file):
    return librosa.load(file, sr=None)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


# ======== Session 状态初始化 =========
# 关键修改：移除默认标签，初始化为空列表
if "dynamic_species_list" not in st.session_state:
    st.session_state["dynamic_species_list"] = []  # 无默认标签
if "current_selected_labels" not in st.session_state:
    st.session_state.current_selected_labels = set()
if "audio_state" not in st.session_state:
    st.session_state.audio_state = {
        "processed_files": set(),
        "current_index": 0,
        "segment_info": {},
        "last_audio_file": None,
        "last_seg_idx": -1,
        "annotations": []
    }

st.set_page_config(layout="wide")
st.title("🐸 青蛙音频标注工具")


# ======== 标签组件（完全独立） =========
def label_management_component():
    """标签管理独立组件，不影响音频处理"""
    with st.sidebar:
        st.markdown("### 🏷️ 标签设置")

        # 使用表单避免实时重运行
        with st.form("label_form", clear_on_submit=True):
            label_file = st.file_uploader("上传标签文件（每行一个）", type=["txt"], key="label_file")
            submit_label = st.form_submit_button("加载标签")

            if submit_label and label_file:
                try:
                    content = label_file.read().decode("utf-8")
                    species_list = [line.strip() for line in content.split("\n") if line.strip()]
                    if species_list:
                        st.session_state["dynamic_species_list"] = species_list
                        st.success(f"加载成功！共 {len(species_list)} 个标签")
                        st.rerun()  # 立即刷新页面，更新标签显示
                    else:
                        st.error("标签文件为空，请检查内容")
                except Exception as e:
                    st.error(f"错误：{str(e)}")

        # 显示当前标签（无标签时提示上传）
        st.markdown("#### 当前标签预览")
        if st.session_state["dynamic_species_list"]:
            st.write(st.session_state["dynamic_species_list"][:5] + (
                ["..."] if len(st.session_state["dynamic_species_list"]) > 5 else []))
        else:
            st.info("尚未加载标签，请上传标签文件")  # 无标签时提示

    return st.session_state["dynamic_species_list"]


# ======== 标注标签组件（独立于音频） =========
def annotation_labels_component(current_segment_key):
    """标注标签独立组件，仅处理标签逻辑"""
    species_list = st.session_state["dynamic_species_list"]
    col_labels = st.columns([1])[0]  # 右侧标签列

    with col_labels:
        st.markdown("### 物种标签（可多选）")

        # 无标签时提示上传
        if not species_list:
            st.warning("请先在左侧上传标签文件")
            return None, None  # 无标签时返回空按钮

        # 搜索功能
        search_query = st.text_input("🔍 搜索标签", "", key=f"search_{current_segment_key}")
        filtered_species = [label for label in species_list if search_query.lower() in label.lower()]

        # 已选标签显示
        st.info(f"已选标签数：{len(st.session_state.current_selected_labels)}")
        if st.session_state.current_selected_labels:
            st.success(f"已选：{', '.join(st.session_state.current_selected_labels)}")

        # 渲染标签复选框
        for label in filtered_species:
            key = f"label_{label}_{current_segment_key}"
            is_selected = label in st.session_state.current_selected_labels

            if st.checkbox(label, key=key, value=is_selected):
                st.session_state.current_selected_labels.add(label)
            else:
                st.session_state.current_selected_labels.remove(label)

        # 操作按钮
        st.markdown("### 🛠️ 操作")
        col_save, col_skip = st.columns(2)
        return col_save, col_skip


# ======== 音频处理逻辑 =========
def process_audio():
    """音频处理主逻辑，与标签组件隔离"""
    audio_state = st.session_state.audio_state
    output_dir = "E:/Frog audio classification/uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")
    df_old = pd.read_csv(csv_path, encoding="utf-8") if os.path.exists(csv_path) else pd.DataFrame(
        columns=["filename", "segment_index", "start_time", "end_time", "labels"]
    )

    # 侧边栏音频上传
    with st.sidebar:
        st.markdown("### 🎵 音频上传")
        uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True, key="audio_files")

        # 下载区域
        st.markdown("### 📥 下载结果")
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button("📄 下载标注CSV", f, "annotations.csv", "text/csv")

        # 音频片段下载
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

    if not uploaded_files:
        st.info("请先在左侧上传音频文件")
        return

    # 音频处理逻辑
    unprocessed = [f for f in uploaded_files if not (audio_state["segment_info"].get(f.name) and
                                                     audio_state["segment_info"][f.name]["current_seg"] >=
                                                     audio_state["segment_info"][f.name]["total_seg"])]

    if audio_state["current_index"] < len(unprocessed):
        audio_file = unprocessed[audio_state["current_index"]]
        y, sr = load_audio(audio_file)
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / 5.0))

        # 初始化片段信息
        if audio_file.name not in audio_state["segment_info"]:
            audio_state["segment_info"][audio_file.name] = {"current_seg": 0, "total_seg": total_segments}
        seg_idx = audio_state["segment_info"][audio_file.name]["current_seg"]
        current_segment_key = f"{audio_file.name}_{seg_idx}"

        # 切换片段时重置选中标签
        if (audio_state["last_audio_file"] != audio_file.name or audio_state["last_seg_idx"] != seg_idx):
            st.session_state.current_selected_labels = set()
            audio_state["last_audio_file"] = audio_file.name
            audio_state["last_seg_idx"] = seg_idx

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")

        # 左侧音频信息
        col_main = st.columns([3])[0]
        with col_main:
            st.subheader("🎧 播放当前片段")
            start_sec = seg_idx * 5.0
            end_sec = min((seg_idx + 1) * 5.0, total_duration)
            segment_y = y[int(start_sec * sr):int(end_sec * sr)]
            audio_bytes = BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            st.audio(audio_bytes, format="audio/wav")

            # 波形图和频谱图（使用缓存）
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 波形图")
                st.image(generate_waveform_image(segment_y, sr), use_container_width=True)
            with col2:
                st.markdown("#### 🎞️ 频谱图")
                st.image(generate_spectrogram_image(segment_y, sr), use_container_width=True)

        # 右侧标签（调用独立组件）
        col_save, col_skip = annotation_labels_component(current_segment_key)

        # 保存逻辑（无标签时禁用）
        if col_save and col_skip:  # 仅当标签加载成功时显示按钮
            with col_save:
                if st.button("保存本段标注", key=f"save_{current_segment_key}"):
                    if not st.session_state.current_selected_labels:
                        st.warning("❗请至少选择一个标签")
                    else:
                        try:
                            segment_filename = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}.wav"
                            sf.write(os.path.join(output_dir, segment_filename), segment_y, sr)

                            # 保存标注
                            entry = {
                                "filename": audio_file.name,
                                "segment_index": segment_filename,
                                "start_time": round(start_sec, 3),
                                "end_time": round(end_sec, 3),
                                "labels": ",".join(st.session_state.current_selected_labels)
                            }
                            audio_state["annotations"].append(entry)
                            df_combined = pd.concat([df_old, pd.DataFrame([entry])], ignore_index=True)
                            df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

                            # 切换到下一段
                            if seg_idx + 1 < total_segments:
                                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                            else:
                                audio_state["processed_files"].add(audio_file.name)
                                audio_state["current_index"] += 1

                            st.success("标注已保存！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败：{str(e)}")

            with col_skip:
                if st.button("跳过本段", key=f"skip_{current_segment_key}"):
                    if seg_idx + 1 < total_segments:
                        audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                    else:
                        audio_state["processed_files"].add(audio_file.name)
                        audio_state["current_index"] += 1
                st.rerun()

    else:
        st.success("🎉 所有音频标注完成！")

    # 更新session_state
    st.session_state.audio_state = audio_state


# ======== 主流程 =========
if __name__ == "__main__":
    # 1. 加载标签组件（独立运行，不影响音频）
    label_management_component()
    # 2. 处理音频标注（独立运行，不依赖标签加载）
    process_audio()
