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
import re
from io import BytesIO
from PIL import Image
import uuid
from pinyin import pinyin  # 导入pinyin库


# ======== 拼音首字母转换工具（基于pinyin库） =========
class PinyinConverter:
    @classmethod
    def get_initial(cls, char):
        """获取单个字符的拼音首字母（小写）"""
        if len(char) != 1:
            return ''  # 非单个字符返回空
        # 字母直接返回小写
        if char.isalpha():
            return char.lower()
        # 汉字通过pinyin库获取首字母
        try:
            # 例如：pinyin("牛") → [['niu']] → 首字母 'n'
            return pinyin(char)[0][0][0].lower()
        except:
            return ''  # 转换失败返回空

    @classmethod
    def get_label_initial(cls, label):
        """获取标签的拼音首字母字符串（如"牛蛙" → "nw"）"""
        return ''.join([cls.get_initial(c) for c in label])


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


# ======== 模糊搜索函数（基于pinyin库） =========
def fuzzy_search(labels, query):
    """
    增强版模糊搜索：
    1. 标签包含查询字符串（不区分大小写）
    2. 标签拼音首字母包含查询字符串（如"nw"匹配"牛蛙"）
    3. 查询字符串是标签的子序列（如"牛亚"匹配"牛蛙_亚种A"）
    """
    if not query:
        return labels
    
    query_lower = query.lower()
    matched = []
    
    for label in labels:
        label_lower = label.lower()
        
        # 规则1：直接包含匹配
        if query_lower in label_lower:
            matched.append(label)
            continue
        
        # 规则2：拼音首字母匹配（基于pinyin库）
        label_initial = PinyinConverter.get_label_initial(label)
        if query_lower in label_initial:
            matched.append(label)
            continue
        
        # 规则3：子序列匹配
        it = iter(label_lower)
        if all(c in it for c in query_lower):
            matched.append(label)
            continue
    
    return matched


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
st.title("🐸 青蛙音频标注工具（基于pinyin库）")


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
        # 提示用户已安装pinyin库
        st.info("✅ 已启用完整拼音首字母搜索（依赖pinyin库）")
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

        # 搜索框（提示拼音首字母功能）
        search_query = st.text_input(
            "🔍 搜索标签（支持拼音首字母，如输入'nw'找'牛蛙'）", 
            "", 
            key=f"search_{current_segment_key}"
        )

        # 缓存搜索结果
        cache_key = f"{current_segment_key}_{search_query}"
        if cache_key not in st.session_state.filtered_labels_cache:
            st.session_state.filtered_labels_cache[cache_key] = fuzzy_search(
                species_list, 
                search_query
            )
        filtered_species = st.session_state.filtered_labels_cache[cache_key]

        # 显示匹配数量
        st.info(f"找到 {len(filtered_species)} 个匹配标签（共 {len(species_list)} 个）")

        # 带滚动条的标签选择区（显示拼音首字母）
        with st.container(height=300):
            for label in filtered_species:
                label_initial = PinyinConverter.get_label_initial(label)
                display_text = f"{label}（首字母：{label_initial}）" if label_initial else label
                
                key = f"label_{label}_{current_segment_key}"
                is_selected = label in st.session_state.current_selected_labels
                if st.checkbox(display_text, key=key, value=is_selected):
                    st.session_state.current_selected_labels.add(label)
                else:
                    st.session_state.current_selected_labels.discard(label)

        # 已选标签展示
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

                            os.makedirs(output_dir, exist_ok=True)
                            base_name = os.path.splitext(audio_file.name)[0]
                            unique_id = uuid.uuid4().hex[:8]
                            segment_filename = f"{base_name}_seg{seg_idx}_{unique_id}.wav"
                            segment_path = os.path.join(output_dir, segment_filename)

                            try:
                                with sf.SoundFile(segment_path, 'w', samplerate=sr, channels=1) as f:
                                    f.write(segment_y)
                            except Exception as audio_error:
                                st.error(f"音频保存失败: {str(audio_error)}")
                                return

                            clean_labels = [label.replace("/", "").replace("\\", "") for label in
                                            st.session_state.current_selected_labels]
                            entry = {
                                "filename": audio_file.name,
                                "segment_index": segment_filename,
                                "start_time": round(start_sec, 3),
                                "end_time": round(end_sec, 3),
                                "labels": ",".join(clean_labels)
                            }

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
                            st.error(f"错误详情: {repr(e)}")

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
