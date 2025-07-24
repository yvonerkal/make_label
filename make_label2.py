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
import zipfile
from io import BytesIO
from PIL import Image
import uuid
from pypinyin import lazy_pinyin

# ======== 工具函数 =========
@st.cache_data(show_spinner=False)
def load_audio(file):
    return librosa.load(file, sr=None)

@st.cache_data(show_spinner=False)
def generate_spectrogram_image(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="Spectrogram (dB)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

@st.cache_data(show_spinner=False)
def generate_waveform_image(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr)
    ax.set(title="Waveform")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def get_pinyin_abbr(text):
    return ''.join([p[0] for p in lazy_pinyin(text) if p])

def get_full_pinyin(text):
    return ''.join(lazy_pinyin(text))

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
if "annotation_mode" not in st.session_state:
    st.session_state.annotation_mode = "分段标注"
if "canvas_boxes" not in st.session_state:
    st.session_state.canvas_boxes = []
if "spec_image" not in st.session_state:
    st.session_state.spec_image = None

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
        st.write(st.session_state["dynamic_species_list"][:5] + (
            ["..."] if len(st.session_state["dynamic_species_list"]) > 5 else []))
        
        # 标注模式选择
        st.session_state.annotation_mode = st.radio(
            "标注模式",
            ["分段标注", "频谱图画框"],
            index=0 if st.session_state.annotation_mode == "分段标注" else 1
        )
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

        # 搜索框
        search_query = st.text_input("🔍 搜索标签（支持中文、拼音首字母、全拼）", "", key=f"search_{current_segment_key}")

        # 生成缓存键
        cache_key = f"{current_segment_key}_{search_query}"

        # 缓存搜索结果
        if cache_key not in st.session_state.filtered_labels_cache:
            filtered_species = []
            if search_query:
                search_lower = search_query.lower()
                for label in species_list:
                    label_lower = label.lower()
                    if (search_lower in label_lower or
                            search_lower in get_pinyin_abbr(label) or
                            search_lower in get_full_pinyin(label)):
                        filtered_species.append(label)
            else:
                filtered_species = species_list.copy()
            st.session_state.filtered_labels_cache[cache_key] = filtered_species

        filtered_species = st.session_state.filtered_labels_cache[cache_key]

        # 显示标签选择框
        for label in filtered_species:
            key = f"label_{label}_{current_segment_key}"
            is_selected = label in st.session_state.current_selected_labels
            if st.checkbox(label, key=key, value=is_selected):
                st.session_state.current_selected_labels.add(label)
            else:
                st.session_state.current_selected_labels.discard(label)

        st.markdown("### 已选标签")
        st.info(f"已选数量：{len(st.session_state.current_selected_labels)}")
        
        # 操作按钮
        col_save, col_skip = st.columns(2)
        return col_save, col_skip

# ======== 频谱图画框组件 ========
def spectral_annotation_component(y, sr, current_segment_key):
    col_main, col_labels = st.columns([3, 1])
    
    with col_main:
        st.subheader("🎧 频谱图画框标注")
        
        # 生成频谱图
        if st.session_state.spec_image is None:
            st.session_state.spec_image = generate_spectrogram_image(y, sr)
        
        # 显示频谱图并添加画布
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="red",
            background_image=st.session_state.spec_image,
            height=400,
            width=800,
            drawing_mode="rect",
            key=f"canvas_{current_segment_key}",
            update_streamlit=True
        )
        
        # 处理画框结果
        if canvas_result.json_data is not None:
            new_boxes = []
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    new_boxes.append({
                        "left": obj["left"],
                        "top": obj["top"],
                        "width": obj["width"],
                        "height": obj["height"]
                    })
            st.session_state.canvas_boxes = new_boxes
        
        # 音频播放器
        audio_bytes = BytesIO()
        sf.write(audio_bytes, y, sr, format='WAV')
        st.audio(audio_bytes, format="audio/wav")
        
        # 操作按钮
        if st.button("撤销上一个标注框"):
            if st.session_state.canvas_boxes:
                st.session_state.canvas_boxes.pop()
                st.rerun()
    
    with col_labels:
        # 使用原有的标签选择组件
        col_save, col_skip = annotation_labels_component(current_segment_key)
        
        if st.session_state.canvas_boxes:
            st.markdown("### 当前标注框")
            for i, box in enumerate(st.session_state.canvas_boxes):
                st.write(f"{i+1}. 位置: ({box['left']:.0f}, {box['top']:.0f}), 大小: {box['width']:.0f}x{box['height']:.0f}")
        
        if col_save and st.button("保存所有标注", key=f"save_boxes_{current_segment_key}"):
            return True
    
    return False

# ======== 音频处理主逻辑 ========
def process_audio():
    audio_state = st.session_state.audio_state
    output_dir = "uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    try:
        df_old = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(
            columns=["filename", "segment_index", "start_time", "end_time", "labels", "box_data"]
        )
    except:
        df_old = pd.DataFrame(columns=["filename", "segment_index", "start_time", "end_time", "labels", "box_data"])

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

        if (audio_state["last_audio_file"] != audio_file.name or audio_state["last_seg_idx"] != seg_idx):
            st.session_state.current_selected_labels = set()
            st.session_state.canvas_boxes = []
            st.session_state.spec_image = None
            audio_state["last_audio_file"], audio_state["last_seg_idx"] = audio_file.name, seg_idx

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        
        start_sec, end_sec = seg_idx * 5.0, min((seg_idx + 1) * 5.0, total_duration)
        segment_y = y[int(start_sec * sr):int(end_sec * sr)]

        if st.session_state.annotation_mode == "频谱图画框":
            if spectral_annotation_component(segment_y, sr, current_segment_key):
                save_spectral_annotations(audio_file, seg_idx, start_sec, end_sec, segment_y, sr, output_dir)
        else:
            col_main, col_labels = st.columns([3, 1])
            
            with col_main:
                st.subheader("🎧 播放当前片段")
                audio_bytes = BytesIO()
                sf.write(audio_bytes, segment_y, sr, format='WAV')
                st.audio(audio_bytes, format="audio/wav")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(generate_waveform_image(segment_y, sr), caption="波形图", use_container_width=True)
                with col2:
                    st.image(generate_spectrogram_image(segment_y, sr), caption="频谱图", use_container_width=True)

            with col_labels:
                col_save, col_skip = annotation_labels_component(current_segment_key)

                if col_save and st.button("保存本段标注", key=f"save_{current_segment_key}"):
                    save_segment_annotation(audio_file, seg_idx, start_sec, end_sec, segment_y, sr, output_dir)
                
                if col_skip and st.button("跳过本段", key=f"skip_{current_segment_key}"):
                    if seg_idx + 1 < total_segments:
                        audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                    else:
                        audio_state["processed_files"].add(audio_file.name)
                        audio_state["current_index"] += 1
                    st.rerun()

    else:
        st.success("🎉 所有音频标注完成！")

    st.session_state.audio_state = audio_state

# ======== 保存函数 ========
def save_segment_annotation(audio_file, seg_idx, start_sec, end_sec, segment_y, sr, output_dir):
    csv_path = os.path.join(output_dir, "annotations.csv")
    
    try:
        if not st.session_state.current_selected_labels:
            st.warning("❗请至少选择一个标签")
            return

        base_name = os.path.splitext(audio_file.name)[0]
        try:
            base_name = base_name.encode('utf-8').decode('utf-8')
        except:
            base_name = "audio_segment"

        unique_id = uuid.uuid4().hex[:8]
        segment_filename = f"{base_name}_seg{seg_idx}_{unique_id}.wav"
        segment_path = os.path.join(output_dir, segment_filename)

        with sf.SoundFile(segment_path, 'w', samplerate=sr, channels=1) as f:
            f.write(segment_y)

        clean_labels = [label.replace("/", "").replace("\\", "") for label in st.session_state.current_selected_labels]
        entry = {
            "filename": audio_file.name,
            "segment_index": segment_filename,
            "start_time": round(start_sec, 3),
            "end_time": round(end_sec, 3),
            "labels": ",".join(clean_labels),
            "box_data": None
        }

        new_df = pd.DataFrame([entry])
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        audio_state = st.session_state.audio_state
        if audio_file.name not in audio_state["segment_info"]:
            audio_state["segment_info"][audio_file.name] = {
                "current_seg": 0,
                "total_seg": int(np.ceil(librosa.get_duration(y=segment_y, sr=sr) / 5.0))
            }

        if seg_idx + 1 < audio_state["segment_info"][audio_file.name]["total_seg"]:
            audio_state["segment_info"][audio_file.name]["current_seg"] += 1
        else:
            audio_state["processed_files"].add(audio_file.name)
            audio_state["current_index"] += 1

        st.session_state.audio_state = audio_state
        st.session_state.current_selected_labels = set()
        st.success(f"成功保存标注！文件: {segment_filename}")
        st.balloons()
        st.rerun()

    except Exception as e:
        st.error(f"保存过程中发生错误: {str(e)}")

def save_spectral_annotations(audio_file, seg_idx, segment_start, segment_end, segment_y, sr, output_dir):
    csv_path = os.path.join(output_dir, "annotations.csv")
    
    try:
        if not st.session_state.current_selected_labels or not st.session_state.canvas_boxes:
            st.warning("请至少添加一个标注框并选择标签")
            return

        base_name = os.path.splitext(audio_file.name)[0]
        try:
            base_name = base_name.encode('utf-8').decode('utf-8')
        except:
            base_name = "audio_segment"

        label = list(st.session_state.current_selected_labels)[0]  # 取第一个选中的标签
        
        # 生成唯一文件名
        unique_id = uuid.uuid4().hex[:8]
        segment_filename = f"{base_name}_seg{seg_idx}_box_{unique_id}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        
        # 保存音频片段
        with sf.SoundFile(segment_path, 'w', samplerate=sr, channels=1) as f:
            f.write(segment_y)
        
        # 创建记录
        entry = {
            "filename": audio_file.name,
            "segment_index": segment_filename,
            "start_time": round(segment_start, 3),
            "end_time": round(segment_end, 3),
            "labels": label,
            "box_data": str(st.session_state.canvas_boxes)
        }

        # 保存到CSV
        new_df = pd.DataFrame([entry])
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 更新状态
        audio_state = st.session_state.audio_state
        if seg_idx + 1 < audio_state["segment_info"][audio_file.name]["total_seg"]:
            audio_state["segment_info"][audio_file.name]["current_seg"] += 1
        else:
            audio_state["processed_files"].add(audio_file.name)
            audio_state["current_index"] += 1

        st.session_state.audio_state = audio_state
        st.session_state.canvas_boxes = []
        st.session_state.current_selected_labels = set()
        st.session_state.spec_image = None
        st.success("成功保存标注框！")
        st.balloons()
        st.rerun()

    except Exception as e:
        st.error(f"保存过程中发生错误: {str(e)}")

if __name__ == "__main__":
    # 安装额外依赖: pip install streamlit-drawable-canvas
    from streamlit_drawable_canvas import st_canvas
    
    label_management_component()
    process_audio()
