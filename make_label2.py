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
import tempfile

# ======== 工具函数 =========
@st.cache_data(show_spinner=False)
def load_audio(file):
    # 使用临时文件避免Streamlit缓存锁定
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(file.getbuffer())
        temp_path = f.name
    y, sr = librosa.load(temp_path, sr=None)
    os.unlink(temp_path)  # 加载后立即删除临时文件
    return y, sr

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

# ======== 音频处理逻辑 =========
def process_audio():
    audio_state = st.session_state.audio_state
    
    # 使用侧边栏输入自定义保存目录（增强灵活性）
    with st.sidebar:
        st.markdown("### 💾 保存设置")
        output_dir = st.text_input("保存目录", "uploaded_audios")
    
    # 验证目录可写性
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, ".permission_test")
        with open(test_file, 'w') as f:
            f.write('test')
        os.unlink(test_file)
    except Exception as e:
        st.error(f"目录不可写：{output_dir} - {str(e)}")
        return
    
    csv_path = os.path.join(output_dir, "annotations.csv")
    
    # 安全加载CSV（处理空文件或格式错误）
    try:
        df_old = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(
            columns=["filename", "segment_index", "start_time", "end_time", "labels"]
        )
    except Exception as e:
        st.warning(f"CSV文件损坏，将创建新文件：{str(e)}")
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
        
        # 验证音频文件是否有效
        try:
            y, sr = load_audio(audio_file)
        except Exception as e:
            st.error(f"无法加载音频文件：{audio_file.name} - {str(e)}")
            st.stop()
            
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / 5.0))
        seg_idx = audio_state["segment_info"].get(audio_file.name, {"current_seg": 0})["current_seg"]
        current_segment_key = f"{audio_file.name}_{seg_idx}"

        if (audio_state["last_audio_file"] != audio_file.name or audio_state["last_seg_idx"] != seg_idx):
            st.session_state.current_selected_labels = set()
            audio_state["last_audio_file"], audio_state["last_seg_idx"] = audio_file.name, seg_idx

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            st.subheader("🎧 播放当前片段")
            start_sec, end_sec = seg_idx * 5.0, min((seg_idx + 1) * 5.0, total_duration)
            segment_y = y[int(start_sec * sr):int(end_sec * sr)]
            
            # 检查片段是否有效
            if len(segment_y) == 0:
                st.warning("当前片段为空，自动跳至下一段")
                if seg_idx + 1 < total_segments:
                    audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                else:
                    audio_state["processed_files"].add(audio_file.name)
                    audio_state["current_index"] += 1
                st.rerun()
            
            # 保存片段到内存
            audio_bytes = BytesIO()
            sf.write(audio_bytes, segment_y, sr, format='WAV')
            audio_bytes.seek(0)
            st.audio(audio_bytes, format="audio/wav")

            col1, col2 = st.columns(2)
            with col1:
                st.image(generate_waveform_image(segment_y, sr), caption="波形图", use_container_width=True)
            with col2:
                st.image(generate_spectrogram_image(segment_y, sr), caption="频谱图", use_container_width=True)

        with col_labels:
            col_save, col_skip = annotation_labels_component(current_segment_key)

            if col_save and col_skip:
                with col_save:
                    if st.button("保存本段标注", key=f"save_{current_segment_key}"):
                        if not st.session_state.current_selected_labels:
                            st.warning("❗请至少选择一个标签")
                            return

                        try:
                            # 使用临时文件中转，避免直接写入目标位置
                            base_name = os.path.splitext(audio_file.name)[0]
                            safe_filename = "".join([c for c in base_name if c.isalnum() or c in ['_', '-']])
                            unique_id = uuid.uuid4().hex[:8]
                            segment_filename = f"{safe_filename}_seg{seg_idx}_{unique_id}.wav"
                            segment_path = os.path.join(output_dir, segment_filename)
                            
                            # 创建临时文件并写入
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            temp_file.close()
                            sf.write(temp_file.name, segment_y, sr)
                            
                            # 验证临时文件是否成功写入
                            if not os.path.exists(temp_file.name):
                                raise Exception("临时文件创建失败")
                            
                            # 移动文件到目标位置（原子操作，减少冲突）
                            os.replace(temp_file.name, segment_path)
                            
                            # 验证最终文件是否存在
                            if not os.path.exists(segment_path):
                                raise Exception(f"文件移动失败：{segment_path}")
                            
                            # 准备CSV条目
                            clean_labels = [label.replace("/", "").replace("\\", "") for label in
                                            st.session_state.current_selected_labels]
                            entry = {
                                "filename": audio_file.name,
                                "segment_index": segment_filename,
                                "start_time": round(start_sec, 3),
                                "end_time": round(end_sec, 3),
                                "labels": ",".join(clean_labels)
                            }
                            
                            # 安全写入CSV（使用临时文件）
                            temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                            temp_csv.close()
                            df_combined = pd.concat([df_old, pd.DataFrame([entry])], ignore_index=True)
                            df_combined.to_csv(temp_csv.name, index=False, encoding="utf-8-sig")
                            os.replace(temp_csv.name, csv_path)
                            
                            # 验证CSV写入成功
                            if not os.path.exists(csv_path) or len(pd.read_csv(csv_path)) != len(df_combined):
                                raise Exception("CSV写入失败")
                            
                            # 更新状态（最后执行，确保文件已保存）
                            if seg_idx + 1 < total_segments:
                                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                            else:
                                audio_state["processed_files"].add(audio_file.name)
                                audio_state["current_index"] += 1
                            
                            st.success(f"标注已保存：{segment_filename}")
                            st.rerun()
                            
                        except PermissionError as e:
                            st.error(f"权限错误：无法写入文件 - {str(e)}")
                        except FileNotFoundError as e:
                            st.error(f"路径错误：文件或目录不存在 - {str(e)}")
                        except Exception as e:
                            st.error(f"保存失败：{str(e)}")
                            # 打印详细错误堆栈（仅在调试模式下可见）
                            import traceback
                            st.write(traceback.format_exc())

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
