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

# 音频预加载和自动播放功能
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
if "auto_play" not in st.session_state:
    st.session_state.auto_play = True  # 自动播放开关
if "audio_context" not in st.session_state:
    st.session_state.audio_context = None  # 音频上下文
if "audio_unlocked" not in st.session_state:
    st.session_state.audio_unlocked = False  # 音频是否解锁（用户已交互）
if "preloaded_audios" not in st.session_state:
    st.session_state.preloaded_audios = {}  # 预加载的音频

st.set_page_config(layout="wide")

st.title("🐸 青蛙音频标注工具")

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

    # 音频片段下载
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

    # 自动播放开关
    st.session_state.auto_play = st.checkbox("自动播放音频", value=st.session_state.auto_play)

    # 标注状态显示
    if uploaded_files:
        with st.expander("✅ 已标注音频", expanded=True):
            for f in uploaded_files:
                if f.name in st.session_state.processed_files:
                    st.write(f.name)
        with st.expander("🕓 未标注音频", expanded=True):
            st.write([f.name for f in uploaded_files if f.name not in st.session_state.processed_files])

# ======== 主处理区域 =========
SEGMENT_DURATION = 5.0  # 每段时长（秒）

# 音频解锁函数 - 首次用户交互时调用
def unlock_audio():
    if not st.session_state.audio_unlocked:
        st.session_state.audio_unlocked = True
        st.info("🔊 音频已解锁，现在可以自动播放")
        st.experimental_rerun()

# 预加载当前和下一个音频片段
def preload_audio(audio_file, seg_idx):
    key = f"{audio_file.name}_{seg_idx}"
    if key not in st.session_state.preloaded_audios:
        y, sr = load_audio(audio_file)
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        # 计算当前段落的时间范围
        start_sec = seg_idx * SEGMENT_DURATION
        end_sec = min((seg_idx + 1) * SEGMENT_DURATION, total_duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]
        
        # 生成音频数据
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, segment_y, sr, format='WAV')
        audio_data = audio_bytes.getvalue()
        
        st.session_state.preloaded_audios[key] = {
            'data': audio_data,
            'sr': sr
        }
        
        # 预加载下一个片段
        next_seg_idx = seg_idx + 1
        next_key = f"{audio_file.name}_{next_seg_idx}"
        if next_seg_idx < int(np.ceil(total_duration / SEGMENT_DURATION)) and next_key not in st.session_state.preloaded_audios:
            next_start_sec = next_seg_idx * SEGMENT_DURATION
            next_end_sec = min((next_seg_idx + 1) * SEGMENT_DURATION, total_duration)
            next_start_sample = int(next_start_sec * sr)
            next_end_sample = int(next_end_sec * sr)
            next_segment_y = y[next_start_sample:next_end_sample]
            
            next_audio_bytes = io.BytesIO()
            sf.write(next_audio_bytes, next_segment_y, sr, format='WAV')
            next_audio_data = next_audio_bytes.getvalue()
            
            st.session_state.preloaded_audios[next_key] = {
                'data': next_audio_data,
                'sr': sr
            }

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

        # 预加载当前和下一个音频片段
        preload_audio(audio_file, seg_idx)

        # 计算当前段落的时间范围
        start_sec = seg_idx * SEGMENT_DURATION
        end_sec = min((seg_idx + 1) * SEGMENT_DURATION, total_duration)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment_y = y[start_sample:end_sample]

        # 布局调整：左侧音频信息，右侧标签和操作
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            # 播放音频段
            st.subheader("🎧 播放当前音频片段")
            
            # 获取预加载的音频数据
            key = f"{audio_file.name}_{seg_idx}"
            audio_data = st.session_state.preloaded_audios[key]['data']
            sr = st.session_state.preloaded_audios[key]['sr']
            
            # 生成随机ID，确保每次加载新音频时ID不同
            audio_id = f"audio_{uuid.uuid4()}"
            
            # 音频解锁按钮（仅在未解锁时显示）
            if not st.session_state.audio_unlocked:
                st.warning("""
                🔇 为了启用自动播放功能，请先点击下方按钮解锁音频。
                这是浏览器的安全限制，只需要点击一次即可。
                """)
                if st.button("解锁音频播放", key="unlock_button"):
                    unlock_audio()
            
            # 使用HTML音频元素
            if st.session_state.audio_unlocked and st.session_state.auto_play:
                # 已解锁且启用自动播放
                st.markdown(f"""
                <audio id="{audio_id}" autoplay controls>
                    <source src="data:audio/wav;base64,{audio_data.hex()}" type="audio/wav">
                    您的浏览器不支持音频播放。
                </audio>
                <script>
                    // 音频自动播放逻辑
                    var audio = document.getElementById('{audio_id}');
                    audio.play().catch(e => {{
                        console.log('自动播放被阻止:', e);
                    }});
                </script>
                """, unsafe_allow_html=True)
                st.info("✅ 自动播放已启用")
            else:
                # 未解锁或禁用自动播放
                st.audio(audio_data, format="audio/wav", start_time=0)
                if st.session_state.audio_unlocked:
                    st.info("🔇 自动播放已禁用，点击播放按钮手动播放音频")
                else:
                    st.info("🔒 音频已锁定，请先解锁以启用自动播放")

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

        with col_labels:  # 右侧区域：标签选择 + 操作按钮
            st.markdown("### 🐸 物种标签（可多选）")
            species_list = ["Rana", "Hyla", "Bufo", "Fejervarya", "Microhyla", "Other"]
            current_key_prefix = f"{audio_file.name}_{seg_idx}"

            # 切换片段时重置复选框状态
            if (st.session_state.last_audio_file != audio_file.name
                    or st.session_state.last_seg_idx != seg_idx):
                for label in species_list:
                    key = f"label_checkbox_{label}_{current_key_prefix}"
                    st.session_state[key] = False
                st.session_state.last_audio_file = audio_file.name
                st.session_state.last_seg_idx = seg_idx

            # 渲染复选框并收集选中的标签
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

            # 显示已选标签
            if selected_labels:
                st.success(f"已选标签: {', '.join(selected_labels)}")
            else:
                st.info("请选择至少一个标签")

            # 操作按钮
            st.markdown("### 🛠️ 操作")
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
                st.experimental_rerun()

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
