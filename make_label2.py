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
from PIL import Image
import uuid
import time


# ======== 工具函数 =========
@st.cache_data(show_spinner=False)
def load_audio(file):
    return librosa.load(file, sr=None)


def generate_spectrogram_image(y, sr, play_pos=0.0):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="Spectrogram (dB)")
    ax.set_xlim(0, librosa.get_duration(y=y, sr=sr))
    if play_pos > 0:
        ax.axvline(x=play_pos, color='red', linestyle='--', linewidth=2, label=f'播放位置: {play_pos:.2f}s')
        ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def generate_waveform_image(y, sr, play_pos=0.0):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform")
    ax.set_xlim(0, librosa.get_duration(y=y, sr=sr))
    if play_pos > 0:
        ax.axvline(x=play_pos, color='red', linestyle='--', linewidth=2, label=f'播放位置: {play_pos:.2f}s')
        ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# ======== Session 状态初始化 =========
if "dynamic_species_list" not in st.session_state:
    st.session_state["dynamic_species_list"] = []
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
if "filtered_labels_cache" not in st.session_state:
    st.session_state.filtered_labels_cache = {}
# 播放状态（仅与原生audio控件绑定）
if "play_state" not in st.session_state:
    st.session_state.play_state = {
        "is_playing": False,  # 是否正在播放
        "start_time": 0.0,    # 播放开始的系统时间
        "current_pos": 0.0,   # 当前播放位置（秒）
        "duration": 0.0,      # 音频片段总时长
        "segment_key": ""     # 绑定当前片段
    }


st.set_page_config(layout="wide")
st.title("🐸 青蛙音频标注工具")


# ======== 标签管理组件 =========
def label_management_component():
    with st.sidebar:
        st.markdown("### 🏷️ 标签设置")
        with st.form("label_form", clear_on_submit=True):
            label_file = st.file_uploader("上传标签文件（每行一个）", type=["txt"], key="label_file")
            submit_label = st.form_submit_button("加载标签")
            if submit_label and label_file:
                try:
                    species_list = [line.strip() for line in label_file.read().decode("utf-8").split("\n") if line.strip()]
                    if species_list:
                        st.session_state["dynamic_species_list"] = species_list
                        st.success(f"加载成功！共 {len(species_list)} 个标签")
                        st.rerun()
                    else:
                        st.error("标签文件为空")
                except Exception as e:
                    st.error(f"错误：{str(e)}")
        st.markdown("#### 当前标签预览")
        st.write(st.session_state["dynamic_species_list"][:5] + (["..."] if len(st.session_state["dynamic_species_list"]) > 5 else []))
    return st.session_state["dynamic_species_list"]


# ======== 右侧标注标签组件 =========
def annotation_labels_component(current_segment_key):
    species_list = st.session_state["dynamic_species_list"]
    col_labels = st.container()

    with col_labels:
        st.markdown("### 物种标签（可多选）")
        if not species_list:
            st.warning("请先在左侧上传标签文件")
            return None, None

        search_query = st.text_input("🔍 搜索标签", "", key=f"search_{current_segment_key}")
        cache_key = f"{current_segment_key}_{search_query}"
        if cache_key not in st.session_state.filtered_labels_cache:
            st.session_state.filtered_labels_cache[cache_key] = [
                label for label in species_list if search_query.lower() in label.lower()
            ]
        filtered_species = st.session_state.filtered_labels_cache[cache_key]

        for label in filtered_species:
            key = f"label_{label}_{current_segment_key}"
            is_selected = label in st.session_state.current_selected_labels
            if st.checkbox(label, key=key, value=is_selected):
                st.session_state.current_selected_labels.add(label)
            else:
                st.session_state.current_selected_labels.discard(label)

        st.markdown("### 已选标签")
        st.info(f"已选数量：{len(st.session_state.current_selected_labels)}")
        if st.session_state.current_selected_labels:
            st.success(f"标签：{', '.join(st.session_state.current_selected_labels)}")
        else:
            st.info("尚未选择标签")

        st.markdown("### 🛠️ 操作")
        col_save, col_skip = st.columns(2)
        return col_save, col_skip


# ======== 音频处理逻辑（核心优化） =========
def process_audio():
    audio_state = st.session_state.audio_state
    output_dir = "uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    # 安全加载CSV
    try:
        df_old = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(
            columns=["filename", "segment_index", "start_time", "end_time", "labels"]
        )
    except:
        df_old = pd.DataFrame(columns=["filename", "segment_index", "start_time", "end_time", "labels"])

    with st.sidebar:
        st.markdown("### 🎵 音频上传")
        uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True, key="audio_files")
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

    if not uploaded_files:
        st.info("请先上传音频文件")
        return

    unprocessed = [f for f in uploaded_files if not (audio_state["segment_info"].get(f.name) and
                  audio_state["segment_info"][f.name]["current_seg"] >= audio_state["segment_info"][f.name]["total_seg"])]

    if audio_state["current_index"] < len(unprocessed):
        audio_file = unprocessed[audio_state["current_index"]]
        y, sr = load_audio(audio_file)
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / 5.0))
        seg_idx = audio_state["segment_info"].get(audio_file.name, {"current_seg": 0})["current_seg"]
        current_segment_key = f"{audio_file.name}_{seg_idx}"

        # 计算当前片段时间范围
        start_sec, end_sec = seg_idx * 5.0, min((seg_idx + 1) * 5.0, total_duration)
        segment_duration = end_sec - start_sec  # 片段总时长
        segment_y = y[int(start_sec * sr):int(end_sec * sr)]

        # 切换片段时重置播放状态
        if (audio_state["last_audio_file"] != audio_file.name or audio_state["last_seg_idx"] != seg_idx):
            st.session_state.current_selected_labels = set()
            audio_state["last_audio_file"], audio_state["last_seg_idx"] = audio_file.name, seg_idx
            # 重置播放状态（新片段）
            st.session_state.play_state = {
                "is_playing": False,
                "start_time": 0.0,
                "current_pos": 0.0,
                "duration": segment_duration,
                "segment_key": current_segment_key
            }

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            st.subheader("🎧 播放当前片段")
            
            # 生成音频字节流（供原生audio控件使用）
            audio_bytes = BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            audio_bytes.seek(0)

            # 原生音频播放控件（仅用这个，不添加额外按钮）
            audio_player = st.audio(audio_bytes, format="audio/wav", start_time=0)

            # 核心：监听播放状态并计算进度
            # 1. 检测播放开始（从暂停/停止到播放）
            if not st.session_state.play_state["is_playing"]:
                # 假设用户点击了播放按钮，记录开始时间
                # （注：Streamlit无法直接监听audio事件，这里通过状态变化推断）
                st.session_state.play_state["is_playing"] = True
                st.session_state.play_state["start_time"] = time.time()
            # 2. 检测播放结束
            elif st.session_state.play_state["is_playing"]:
                current_pos = time.time() - st.session_state.play_state["start_time"]
                if current_pos >= segment_duration:
                    st.session_state.play_state["is_playing"] = False
                    st.session_state.play_state["current_pos"] = 0.0
                else:
                    st.session_state.play_state["current_pos"] = current_pos

            # 3. 定时刷新页面（播放时每0.1秒刷新一次，更新红线位置）
            if st.session_state.play_state["is_playing"]:
                # 控制刷新频率，避免过度重运行
                if time.time() - st.session_state.play_state["start_time"] > st.session_state.play_state["current_pos"] + 0.1:
                    st.rerun()

            # 波形图（第一行）
            st.markdown("#### 📈 波形图")
            wave_img = generate_waveform_image(
                segment_y, 
                sr, 
                play_pos=st.session_state.play_state["current_pos"]  # 红线位置=当前进度
            )
            st.image(wave_img, use_container_width=True)

            # 频谱图（第二行）
            st.markdown("#### 🎞️ 频谱图")
            spec_img = generate_spectrogram_image(
                segment_y, 
                sr, 
                play_pos=st.session_state.play_state["current_pos"]  # 红线位置=当前进度
            )
            st.image(spec_img, use_container_width=True)

        with col_labels:
            col_save, col_skip = annotation_labels_component(current_segment_key)

            if col_save and col_skip:
                with col_save:
                    if st.button("保存本段标注", key=f"save_{current_segment_key}"):
                        if not st.session_state.current_selected_labels:
                            st.warning("❗请至少选择一个标签")
                            return

                        try:
                            # 保存音频片段
                            base_name = os.path.splitext(audio_file.name)[0]
                            unique_id = uuid.uuid4().hex[:8]
                            segment_filename = f"{base_name}_seg{seg_idx}_{unique_id}.wav"
                            segment_path = os.path.join(output_dir, segment_filename)
                            sf.write(segment_path, segment_y, sr)

                            # 保存CSV
                            clean_labels = [label.replace("/", "").replace("\\", "") for label in st.session_state.current_selected_labels]
                            entry = {
                                "filename": audio_file.name,
                                "segment_index": segment_filename,
                                "start_time": round(start_sec, 3),
                                "end_time": round(end_sec, 3),
                                "labels": ",".join(clean_labels)
                            }
                            new_df = pd.DataFrame([entry])
                            combined_df = pd.concat([df_old, new_df], ignore_index=True) if not df_old.empty else new_df
                            combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                            # 更新状态
                            if seg_idx + 1 < total_segments:
                                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                            else:
                                audio_state["processed_files"].add(audio_file.name)
                                audio_state["current_index"] += 1

                            st.success(f"成功保存标注！文件: {segment_filename}")
                            st.rerun()

                        except Exception as e:
                            st.error(f"保存失败: {str(e)}")

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

    st.session_state.audio_state = audio_state


# ======== 主流程 =========
if __name__ == "__main__":
    label_management_component()
    process_audio()
