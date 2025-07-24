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

# 确保pinyin库正确导入
try:
    from pinyin import pinyin
except ImportError:
    st.error("❌ 请安装pinyin库：pip install pinyin")
    st.stop()


# ======== 拼音首字母处理（核心组件） =========
class PinyinHandler:
    @staticmethod
    def get_initial(char):
        if len(char) != 1:
            return ""
        if not '\u4e00' <= char <= '\u9fff':  # 仅处理汉字
            return ""
        try:
            py_result = pinyin(char)
            if py_result and isinstance(py_result[0], list) and py_result[0]:
                return py_result[0][0].lower()[0]
            return ""
        except:
            return ""

    @staticmethod
    def label_to_initial(label):
        return ''.join([PinyinHandler.get_initial(c) for c in label if '\u4e00' <= c <= '\u9fff'])


# ======== 模糊搜索函数 =========
def fuzzy_search(labels, query):
    if not query:
        return labels
    query_clean = query.lower().strip()
    matched = []
    for label in labels:
        label_initial = PinyinHandler.label_to_initial(label)
        label_lower = label.lower()
        if query_clean == label_initial:
            matched.append((label, 3))
        elif query_clean in label_initial:
            matched.append((label, 2))
        elif query_clean in label_lower:
            matched.append((label, 1))
    if matched:
        unique_matched = {label: prio for label, prio in matched}
        return sorted(unique_matched.keys(), key=lambda x: -unique_matched[x])
    return []


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
    librosa.display.waveshow(y, sr=sr, ax=ax)
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


# ======== 标签管理组件（确保在调用前定义） =========
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

        search_query = st.text_input(
            "🔍 搜索标签（示例：输入'yy'找'鸳鸯'）",
            "",
            key=f"search_{current_segment_key}"
        )

        cache_key = f"{current_segment_key}_{search_query}"
        if cache_key not in st.session_state.filtered_labels_cache:
            st.session_state.filtered_labels_cache[cache_key] = fuzzy_search(
                species_list,
                search_query
            )
        filtered_species = st.session_state.filtered_labels_cache[cache_key]

        st.info(f"找到 {len(filtered_species)} 个匹配标签（共 {len(species_list)} 个）")

        with st.container(height=300):
            for label in filtered_species:
                label_initial = PinyinHandler.label_to_initial(label)
                display_text = f"{label}（首字母：{label_initial}）" if label_initial else label
                
                key = f"label_{label}_{current_segment_key}"
                is_selected = label in st.session_state.current_selected_labels
                if st.checkbox(display_text, key=key, value=is_selected):
                    st.session_state.current_selected_labels.add(label)
                else:
                    st.session_state.current_selected_labels.discard(label)

        st.markdown("### 已选标签")
        st.info(f"已选数量：{len(st.session_state.current_selected_labels)}")
        if st.session_state.current_selected_labels:
            st.success("标签：\n" + ", ".join(st.session_state.current_selected_labels).replace(", ", "\n"))
        else:
            st.info("尚未选择标签")

        st.markdown("### 🛠️ 操作")
        col_save, col_skip = st.columns(2)
        return col_save, col_skip


# ======== 音频处理逻辑 =========
def process_audio():
    audio_state = st.session_state.audio_state
    output_dir = "uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

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

        if (audio_state["last_audio_file"] != audio_file.name or audio_state["last_seg_idx"] != seg_idx):
            st.session_state.current_selected_labels = set()
            audio_state["last_audio_file"], audio_state["last_seg_idx"] = audio_file.name, seg_idx

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        col_main, col_labels = st.columns([3, 1])

        with col_main:
            st.subheader("🎧 播放当前片段")
            start_sec, end_sec = seg_idx * 5.0, min((seg_idx + 1) * 5.0, total_duration)
            segment_y = y[int(start_sec * sr):int(end_sec * sr)]
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

            if col_save and col_skip:
                with col_save:
                    if st.button("保存本段标注", key=f"save_{current_segment_key}"):
                        try:
                            if not st.session_state.current_selected_labels:
                                st.warning("❗请至少选择一个标签")
                                return

                            base_name = os.path.splitext(audio_file.name)[0]
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
                                "labels": ",".join(clean_labels)
                            }

                            new_df = pd.DataFrame([entry])
                            if os.path.exists(csv_path):
                                existing_df = pd.read_csv(csv_path)
                                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                            else:
                                combined_df = new_df

                            combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                            if audio_file.name not in audio_state["segment_info"]:
                                audio_state["segment_info"][audio_file.name] = {"current_seg": 0, "total_seg": total_segments}

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


# ======== 主流程（确保函数调用在定义之后） =========
if __name__ == "__main__":
    # 确保label_management_component已定义再调用
    label_management_component()
    process_audio()
