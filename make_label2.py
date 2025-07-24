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

# 确保pinyin库正确导入（必选）
try:
    from pinyin import pinyin
except ImportError:
    st.error("请先安装pinyin库：pip install pinyin")
    st.stop()


# ======== 核心：拼音首字母提取（简化逻辑，确保正确） =========
def get_pinyin_initial(label):
    """
    提取标签的拼音首字母串（如"鸳鸯" → "yy"）
    逻辑：对每个汉字取第一个拼音的首字母，非汉字忽略
    """
    initials = []
    for char in label:
        # 只处理汉字（Unicode范围：\u4e00-\u9fff）
        if '\u4e00' <= char <= '\u9fff':
            try:
                # pinyin("鸳") → [['yuan']] → 取首字母'y'
                py = pinyin(char)[0][0].lower()  # 取第一个拼音并小写
                initials.append(py[0])  # 取首字母
            except:
                # 生僻字转换失败则跳过
                continue
    return ''.join(initials)  # 组合成首字母串


# ======== 核心：搜索匹配（仅保留首字母和字符匹配，确保生效） =========
def search_labels(labels, query):
    """
    搜索逻辑：
    1. 空查询返回所有标签
    2. 首字母匹配（如"yy" → "鸳鸯"）
    3. 字符包含匹配（如"鸳" → "鸳鸯"）
    """
    if not query:
        return labels
    
    query = query.lower().strip()
    matched = []
    for label in labels:
        # 提取首字母串
        label_initial = get_pinyin_initial(label)
        # 标签字符（小写）
        label_lower = label.lower()
        
        # 规则1：首字母完全匹配（优先级最高）
        if query == label_initial:
            matched.append((label, 2))
        # 规则2：首字母包含匹配
        elif query in label_initial:
            matched.append((label, 1))
        # 规则3：字符包含匹配
        elif query in label_lower:
            matched.append((label, 0))
    
    # 按优先级排序，去重后返回
    if matched:
        # 去重（保留优先级最高的）
        unique = {}
        for label, prio in matched:
            if label not in unique or prio > unique[label]:
                unique[label] = prio
        # 按优先级排序
        return sorted(unique.keys(), key=lambda x: -unique[x])
    return []


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
        "last_seg_idx": -1
    }
if "search_cache" not in st.session_state:
    st.session_state.search_cache = {}

st.set_page_config(layout="wide")
st.title("🐸 青蛙音频标注工具（拼音首字母修复版）")


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
                    st.session_state["dynamic_species_list"] = species_list
                    st.success(f"加载成功！共 {len(species_list)} 个标签")
                    st.rerun()
                except Exception as e:
                    st.error(f"错误：{str(e)}")
        # 调试：显示标签首字母提取结果（方便排查）
        if st.session_state["dynamic_species_list"]:
            st.markdown("#### 标签首字母预览（调试用）")
            preview = [f"{label} → {get_pinyin_initial(label)}" for label in st.session_state["dynamic_species_list"][:3]]
            st.write("\n".join(preview) + ("..." if len(st.session_state["dynamic_species_list"]) > 3 else ""))
    return st.session_state["dynamic_species_list"]


# ======== 右侧标注组件（显示首字母，确保匹配可见） =========
def annotation_component(current_key):
    labels = st.session_state["dynamic_species_list"]
    if not labels:
        st.warning("请先上传标签文件")
        return None, None

    # 搜索框
    query = st.text_input(
        "🔍 搜索标签（示例：输入'yy'找'鸳鸯'）",
        "",
        key=f"query_{current_key}"
    )

    # 缓存搜索结果
    cache_key = f"{current_key}_{query}"
    if cache_key not in st.session_state.search_cache:
        st.session_state.search_cache[cache_key] = search_labels(labels, query)
    results = st.session_state.search_cache[cache_key]

    # 显示匹配信息
    st.info(f"匹配结果：{len(results)}/{len(labels)} 个标签")

    # 标签选择区（带首字母显示）
    with st.container(height=300):
        for label in results:
            initial = get_pinyin_initial(label)
            # 显示：标签（首字母：xx）
            display = f"{label}（首字母：{initial}）" if initial else label
            key = f"label_{label}_{current_key}"
            checked = label in st.session_state.current_selected_labels
            if st.checkbox(display, key=key, value=checked):
                st.session_state.current_selected_labels.add(label)
            else:
                st.session_state.current_selected_labels.discard(label)

    # 已选标签
    st.markdown("### 已选标签")
    st.write(f"数量：{len(st.session_state.current_selected_labels)}")
    if st.session_state.current_selected_labels:
        st.success(", ".join(st.session_state.current_selected_labels))

    # 操作按钮
    col_save, col_skip = st.columns(2)
    return col_save, col_skip


# ======== 音频处理逻辑 =========
def process_audio():
    audio_state = st.session_state.audio_state
    output_dir = "uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    # 加载CSV
    try:
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(
            columns=["filename", "segment", "start", "end", "labels"]
        )
    except:
        df = pd.DataFrame(columns=["filename", "segment", "start", "end", "labels"])

    # 侧边栏：上传和下载
    with st.sidebar:
        st.markdown("### 🎵 音频上传")
        files = st.file_uploader("上传.wav文件", type="wav", accept_multiple_files=True, key="audios")
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button("下载标注结果", f, "annotations.csv")

    if not files:
        st.info("请上传音频文件")
        return

    # 处理当前音频段
    current_idx = audio_state["current_index"]
    if current_idx < len(files):
        file = files[current_idx]
        y, sr = load_audio(file)
        total_dur = librosa.get_duration(y=y, sr=sr)
        total_seg = int(np.ceil(total_dur / 5))
        seg_idx = audio_state["segment_info"].get(file.name, {"current": 0})["current"]
        current_key = f"{file.name}_{seg_idx}"

        # 切换段时重置选择
        if audio_state["last_audio_file"] != file.name or audio_state["last_seg_idx"] != seg_idx:
            st.session_state.current_selected_labels = set()
            audio_state["last_audio_file"] = file.name
            audio_state["last_seg_idx"] = seg_idx

        # 显示当前段信息
        st.header(f"处理：{file.name}（第 {seg_idx+1}/{total_seg} 段）")
        col_main, col_annot = st.columns([3, 1])

        with col_main:
            # 音频播放
            start = seg_idx * 5
            end = min(start + 5, total_dur)
            seg_y = y[int(start*sr):int(end*sr)]
            audio_buf = BytesIO()
            sf.write(audio_buf, seg_y, sr, format="WAV")
            st.audio(audio_buf, format="audio/wav")

            # 波形图和频谱图
            col1, col2 = st.columns(2)
            with col1:
                st.image(generate_waveform_image(seg_y, sr), caption="波形图", use_container_width=True)
            with col2:
                st.image(generate_spectrogram_image(seg_y, sr), caption="频谱图", use_container_width=True)

        with col_annot:
            save_btn, skip_btn = annotation_component(current_key)

            # 保存按钮
            if save_btn.button("保存标注", key=f"save_{current_key}"):
                if not st.session_state.current_selected_labels:
                    st.warning("请至少选择一个标签")
                    return
                try:
                    # 保存音频段
                    seg_name = f"{file.name}_seg{seg_idx}_{uuid.uuid4().hex[:6]}.wav"
                    sf.write(os.path.join(output_dir, seg_name), seg_y, sr)
                    # 保存到CSV
                    new_row = {
                        "filename": file.name,
                        "segment": seg_name,
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "labels": ",".join(st.session_state.current_selected_labels)
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    # 更新状态
                    if seg_idx + 1 < total_seg:
                        audio_state["segment_info"][file.name] = {"current": seg_idx + 1}
                    else:
                        audio_state["processed_files"].add(file.name)
                        audio_state["current_index"] += 1
                    st.success("保存成功！")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败：{str(e)}")

            # 跳过按钮
            if skip_btn.button("跳过", key=f"skip_{current_key}"):
                if seg_idx + 1 < total_seg:
                    audio_state["segment_info"][file.name] = {"current": seg_idx + 1}
                else:
                    audio_state["current_index"] += 1
                st.rerun()

    else:
        st.success("所有音频处理完成！")

    st.session_state.audio_state = audio_state


# ======== 主流程 =========
if __name__ == "__main__":
    label_management_component()
    process_audio()
