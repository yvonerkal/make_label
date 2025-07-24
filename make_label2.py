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


# ======== 工具函数（新增红线绘制功能） =========
@st.cache_data(show_spinner=False)
def load_audio(file):
    return librosa.load(file, sr=None)


def generate_spectrogram_image(y, sr, play_pos=0.0):
    """生成带播放进度红线的频谱图"""
    fig, ax = plt.subplots(figsize=(5, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="Spectrogram (dB)")
    # 绘制播放进度红线（红色虚线）
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
    """生成带播放进度红线的波形图"""
    fig, ax = plt.subplots(figsize=(5, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title="Waveform")
    # 绘制播放进度红线（红色虚线）
    if play_pos > 0:
        ax.axvline(x=play_pos, color='red', linestyle='--', linewidth=2, label=f'播放位置: {play_pos:.2f}s')
        ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


# ======== Session 状态初始化（新增播放状态跟踪） =========
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
# 新增：播放状态跟踪变量
if "play_state" not in st.session_state:
    st.session_state.play_state = {
        "is_playing": False,       # 是否正在播放
        "start_time": 0.0,         # 播放开始的系统时间
        "current_pos": 0.0,        # 当前播放位置（秒）
        "segment_duration": 0.0,   # 当前片段总时长
        "current_segment_key": ""  # 当前片段标识（避免跨片段干扰）
    }


st.set_page_config(layout="wide")
st.title("🐸 青蛙音频标注工具")


# ======== 标签管理组件（保持不变） =========
def label_management_component():
    with st.sidebar:
        st.markdown("### 🏷️ 标签设置")
        with st.form("label_form", clear_on_submit=True):
            label_file = st.file_uploader("上传标签文件（每行一个）", type=["txt"], key="label_file")
            submit_label = st.form_submit_button("加载标签")
            if submit_label and label_file:
                try:
                    species_list = [line.strip() for line in label_file.read().decode("utf-8").split("\n") if
                                    line.strip()]
                    if species_list:
                        st.session_state["dynamic_species_list"] = species_list
                        st.success(f"加载成功！共 {len(species_list)} 个标签")
                        st.rerun()
                    else:
                        st.error("标签文件为空")
                except Exception as e:
                    st.error(f"错误：{str(e)}")
        st.markdown("#### 当前标签预览")
        st.write(st.session_state["dynamic_species_list"][:5] + (
            ["..."] if len(st.session_state["dynamic_species_list"]) > 5 else []))
    return st.session_state["dynamic_species_list"]


# ======== 右侧标注标签组件（保持不变） =========
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


# ======== 音频处理逻辑（核心修改：添加红线跟随） =========
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
                                                     audio_state["segment_info"][f.name]["current_seg"] >=
                                                     audio_state["segment_info"][f.name]["total_seg"])]

    if audio_state["current_index"] < len(unprocessed):
        audio_file = unprocessed[audio_state["current_index"]]
        y, sr = load_audio(audio_file)
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / 5.0))
        seg_idx = audio_state["segment_info"].get(audio_file.name, {"current_seg": 0})["current_seg"]
        current_segment_key = f"{audio_file.name}_{seg_idx}"

        # 计算当前片段时间范围
        start_sec = seg_idx * 5.0
        end_sec = min((seg_idx + 1) * 5.0, total_duration)
        segment_duration = end_sec - start_sec  # 当前片段总时长
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
                "segment_duration": segment_duration,
                "current_segment_key": current_segment_key
            }

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            st.subheader("🎧 播放当前片段（红线将跟随进度）")
            
            # 生成音频字节流
            audio_bytes = BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            audio_bytes.seek(0)

            # 播放控制：通过按钮触发播放状态（因Streamlit无法直接监听audio事件）
            col_play, col_reset = st.columns(2)
            with col_play:
                if st.button("▶️ 开始播放（点击后红线跟随）", key=f"play_btn_{current_segment_key}"):
                    # 开始播放时记录系统时间
                    st.session_state.play_state = {
                        "is_playing": True,
                        "start_time": time.time(),
                        "current_pos": 0.0,
                        "segment_duration": segment_duration,
                        "current_segment_key": current_segment_key
                    }
            with col_reset:
                if st.button("⏹️ 停止播放", key=f"stop_btn_{current_segment_key}"):
                    # 停止播放时重置状态
                    st.session_state.play_state = {
                        "is_playing": False,
                        "start_time": 0.0,
                        "current_pos": 0.0,
                        "segment_duration": segment_duration,
                        "current_segment_key": current_segment_key
                    }

            # 显示原生音频播放器
            st.audio(audio_bytes, format="audio/wav")

            # 计算当前播放位置（核心逻辑）
            current_pos = 0.0
            if st.session_state.play_state["is_playing"] and st.session_state.play_state["current_segment_key"] == current_segment_key:
                # 计算已播放时间（当前系统时间 - 开始时间）
                elapsed = time.time() - st.session_state.play_state["start_time"]
                current_pos = min(elapsed, segment_duration)  # 不超过片段时长
                st.session_state.play_state["current_pos"] = current_pos

                # 定时刷新（每0.1秒一次，确保红线流畅移动）
                if elapsed < segment_duration and (elapsed % 0.1 < 0.02):  # 控制刷新频率
                    st.rerun()  # 刷新页面更新红线位置
                # 播放结束后自动停止
                if current_pos >= segment_duration:
                    st.session_state.play_state["is_playing"] = False

            # 显示波形图（带红线）
            col1, col2 = st.columns(2)
            with col1:
                wave_img = generate_waveform_image(segment_y, sr, play_pos=current_pos)
                st.image(wave_img, caption="波形图（红线为当前播放位置）", use_container_width=True)

            # 显示频谱图（带红线）
            with col2:
                spec_img = generate_spectrogram_image(segment_y, sr, play_pos=current_pos)
                st.image(spec_img, caption="频谱图（红线为当前播放位置）", use_container_width=True)

            # 显示当前播放进度（辅助信息）
            st.write(f"当前播放位置: {current_pos:.2f}s / 总时长: {segment_duration:.2f}s")

        with col_labels:
            col_save, col_skip = annotation_labels_component(current_segment_key)

            if col_save and col_skip:
                with col_save:
                    if st.button("保存本段标注", key=f"save_{current_segment_key}"):
                        try:
                            # 1. 检查标签
                            if not st.session_state.current_selected_labels:
                                st.warning("❗请至少选择一个标签")
                                return

                            # 2. 确保输出目录存在
                            os.makedirs(output_dir, exist_ok=True)

                            # 3. 生成安全的文件名
                            base_name = os.path.splitext(audio_file.name)[0]
                            try:
                                base_name = base_name.encode('utf-8').decode('utf-8')
                            except:
                                base_name = "audio_segment"

                            unique_id = uuid.uuid4().hex[:8]
                            segment_filename = f"{base_name}_seg{seg_idx}_{unique_id}.wav"
                            segment_path = os.path.join(output_dir, segment_filename)

                            # 4. 保存音频文件
                            try:
                                with sf.SoundFile(segment_path, 'w', samplerate=sr, channels=1) as f:
                                    f.write(segment_y)
                            except Exception as audio_error:
                                st.error(f"音频保存失败: {str(audio_error)}")
                                return

                            # 5. 准备CSV条目
                            clean_labels = [label.replace("/", "").replace("\\", "") for label in
                                            st.session_state.current_selected_labels]
                            entry = {
                                "filename": audio_file.name,
                                "segment_index": segment_filename,
                                "start_time": round(start_sec, 3),
                                "end_time": round(end_sec, 3),
                                "labels": ",".join(clean_labels)
                            }

                            # 6. 更新CSV
                            try:
                                new_df = pd.DataFrame([entry])
                                if os.path.exists(csv_path):
                                    existing_df = pd.read_csv(csv_path)
                                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                                else:
                                    combined_df = new_df

                                combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                            except Exception as csv_error:
                                st.error(f"CSV保存失败: {str(csv_error)}")
                                if os.path.exists(segment_path):
                                    os.remove(segment_path)
                                return

                            # 7. 更新状态
                            if audio_file.name not in audio_state["segment_info"]:
                                audio_state["segment_info"][audio_file.name] = {
                                    "current_seg": 0,
                                    "total_seg": total_segments
                                }

                            if seg_idx + 1 < total_segments:
                                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                            else:
                                audio_state["processed_files"].add(audio_file.name)
                                audio_state["current_index"] += 1

                            st.session_state.audio_state = audio_state
                            st.success(f"成功保存标注！文件: {segment_filename}")
                            st.balloons()
                            st.rerun()

                        except Exception as e:
                            st.error(f"保存过程中发生错误: {str(e)}")

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
