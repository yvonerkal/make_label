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
from pinyin import pinyin  # 确保已安装：pip install pinyin


# ======== 核心优化：拼音首字母转换（解决匹配问题） =========
class PinyinHandler:
    @staticmethod
    def get_initial(char):
        """
        获取单个字符的拼音首字母（处理各种边缘情况）
        返回值：小写首字母（如"牛"→"n"，"W"→"w"，未识别→""）
        """
        # 处理空字符或长字符串
        if not char or len(char) != 1:
            return ""
        
        # 处理字母（直接返回小写）
        if char.isalpha():
            return char.lower()
        
        # 处理数字和符号（直接返回，不参与首字母匹配）
        if char.isdigit() or char in "._-()[]{}@#$%^&*":
            return ""
        
        # 处理汉字（使用pinyin库，取第一个拼音的首字母）
        try:
            # pinyin("牛") 返回 [['niu']] → 提取首字母 'n'
            py_list = pinyin(char)
            if py_list and len(py_list[0]) > 0:
                first_pinyin = py_list[0][0].lower()  # 取第一个拼音并小写
                return first_pinyin[0] if first_pinyin else ""  # 返回首字母
            return ""
        except Exception as e:
            # 转换失败时返回空（避免报错）
            return ""

    @staticmethod
    def label_to_initial(label):
        """将标签转换为拼音首字母字符串（如"牛蛙_2型"→"nw"）"""
        return ''.join([PinyinHandler.get_initial(c) for c in label])


# ======== 优化：模糊搜索函数（提高匹配成功率） =========
def fuzzy_search(labels, query):
    """
    增强版模糊搜索，解决匹配不上的问题：
    1. 忽略查询中的空格和特殊字符
    2. 支持首字母部分匹配（如"n"匹配"牛蛙"）
    3. 优先返回精确匹配结果
    """
    if not query:
        return labels  # 空查询返回所有标签
    
    # 预处理查询：去除空格和特殊字符，转为小写
    query_clean = ''.join([c.lower() for c in query if c.isalnum()])
    if not query_clean:
        return labels
    
    matched = []
    for label in labels:
        label_clean = label.lower()
        label_initial = PinyinHandler.label_to_initial(label)  # 标签首字母串
        
        # 规则1：标签包含查询字符串（原始字符匹配）
        if query_clean in label_clean:
            matched.append((label, 3))  # 优先级3（最高）
            continue
        
        # 规则2：标签首字母包含查询字符串（首字母匹配）
        if query_clean in label_initial:
            matched.append((label, 2))  # 优先级2
            continue
        
        # 规则3：查询是标签首字母的子序列（如"nw"匹配"nww"）
        it = iter(label_initial)
        if all(c in it for c in query_clean):
            matched.append((label, 1))  # 优先级1
            continue
    
    # 按优先级排序（精确匹配在前），去重后返回
    if matched:
        # 去重并按优先级排序
        unique_matched = list({label: prio for label, prio in matched}.items())
        unique_matched.sort(key=lambda x: -x[1])  # 优先级高的排前面
        return [item[0] for item in unique_matched]
    return []  # 无匹配时返回空列表


# ======== 工具函数（保持不变） =========
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
st.title("🐸 青蛙音频标注工具（拼音匹配优化版）")


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
        st.info("✅ 已启用拼音首字母搜索（支持汉字、字母混合查询）")
        st.markdown("#### 当前标签预览")
        st.write(st.session_state["dynamic_species_list"][:5] + (["..."] if len(st.session_state["dynamic_species_list"]) > 5 else []))
    return st.session_state["dynamic_species_list"]


# ======== 右侧标注标签组件（显示匹配细节） =========
def annotation_labels_component(current_segment_key):
    species_list = st.session_state["dynamic_species_list"]
    col_labels = st.container()

    with col_labels:
        st.markdown("### 物种标签（可多选）")
        if not species_list:
            st.warning("请先在左侧上传标签文件")
            return None, None

        # 搜索框（提示使用方法）
        search_query = st.text_input(
            "🔍 搜索标签（示例：输入'nw'找'牛蛙'，输入'牛'也可）",
            "",
            key=f"search_{current_segment_key}"
        )

        # 缓存搜索结果（提高性能）
        cache_key = f"{current_segment_key}_{search_query}"
        if cache_key not in st.session_state.filtered_labels_cache:
            st.session_state.filtered_labels_cache[cache_key] = fuzzy_search(
                species_list,
                search_query
            )
        filtered_species = st.session_state.filtered_labels_cache[cache_key]

        # 显示匹配信息（帮助用户理解结果）
        st.info(
            f"找到 {len(filtered_species)} 个匹配标签 "
            f"（总标签数：{len(species_list)}）"
        )

        # 显示标签及首字母（方便用户核对）
        with st.container(height=300):  # 固定高度+滚动条
            for label in filtered_species:
                label_initial = PinyinHandler.label_to_initial(label)
                # 显示标签和其首字母（如"牛蛙（首字母：nw）"）
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


# ======== 音频处理逻辑（保持不变） =========
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
