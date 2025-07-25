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
import sys

sys.setrecursionlimit(10000)  # 增加递归深度限制


# ======== 工具函数 =========
@st.cache_data(show_spinner=False)
def load_audio(file):
    return librosa.load(file, sr=None)


def generate_spectrogram_data(y, sr):
    """生成频谱图数据及坐标轴范围（用于坐标转换）"""
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    times = librosa.times_like(D, sr=sr)  # 时间轴
    frequencies = librosa.fft_frequencies(sr=sr)  # 频率轴
    return D, times, frequencies


def generate_spectrogram_image(D, times, frequencies):
    """生成带坐标的频谱图（确保可正常显示）"""
    # 强制matplotlib使用Agg后端，避免显示问题
    plt.switch_backend('Agg')
    
    fig = plt.figure(figsize=(12, 6), dpi=100)
    img = librosa.display.specshow(
        D,
        sr=frequencies[-1] * 2,
        x_axis='time',
        y_axis='log',
    )
    plt.xlim(times[0], times[-1])
    plt.ylim(frequencies[0], frequencies[-1])
    plt.colorbar(format='%+2.0f dB')
    plt.title('频谱图（可画框标注）')
    plt.tight_layout(pad=0)

    # 保存图像到内存
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')  # 确保图像为RGB模式
    plt.close(fig)  # 强制关闭，释放资源
    return img


@st.cache_data(show_spinner=False)
def generate_waveform_image(y, sr):
    plt.switch_backend('Agg')
    plt.figure(figsize=(12, 3), dpi=100)
    librosa.display.waveshow(y, sr=sr)
    plt.title('波形图')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img


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
    }
if "filtered_labels_cache" not in st.session_state:
    st.session_state.filtered_labels_cache = {}
if "canvas_boxes" not in st.session_state:
    st.session_state.canvas_boxes = []
if "spec_params" not in st.session_state:
    st.session_state.spec_params = {"times": None, "frequencies": None, "img_size": (0, 0)}

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
                    st.session_state["dynamic_species_list"] = species_list
                    st.success(f"加载成功！共 {len(species_list)} 个标签")
                    st.rerun()
                except Exception as e:
                    st.error(f"错误：{str(e)}")
        st.markdown("#### 当前标签预览")
        st.write(st.session_state["dynamic_species_list"][:5] + (
            ["..."] if len(st.session_state["dynamic_species_list"]) > 5 else []))

        # 标注模式选择
        st.session_state.annotation_mode = st.radio(
            "标注模式",
            ["分段标注", "频谱图画框"],
            index=0 if st.session_state.get("annotation_mode") == "分段标注" else 1
        )
    return st.session_state["dynamic_species_list"]


# ======== 频谱图画框+标签关联组件 =========
def spectral_annotation_component(y, sr, current_segment_key):
    # 生成频谱图数据
    D, times, frequencies = generate_spectrogram_data(y, sr)
    
    # 生成频谱图（不使用session_state缓存，避免状态问题）
    spec_image = generate_spectrogram_image(D, times, frequencies)
    
    # 强制显示测试（确认图像可显示）
    # st.image(spec_image, caption="测试频谱图", use_container_width=True)
    
    # 获取图像尺寸（确保有效）
    img_width, img_height = spec_image.size
    if img_width <= 0 or img_height <= 0:
        st.error("频谱图尺寸无效，无法显示")
        return False, False

    # 更新频谱图参数
    st.session_state.spec_params = {
        "times": times,
        "frequencies": frequencies,
        "img_size": (img_width, img_height)
    }

    # 主区域布局
    col_main, col_labels = st.columns([3, 1])

    with col_main:
        st.subheader("🎧 频谱图画框标注（点击画布绘制矩形）")

        # 1. 音频播放（上方）
        st.markdown("#### 音频播放")
        audio_bytes = BytesIO()
        sf.write(audio_bytes, y, sr, format='WAV')
        st.audio(audio_bytes, format="audio/wav", start_time=0)

        # 2. 频谱图画布区域（核心修复：确保图像正确传递给canvas）
        st.markdown("#### 频谱图（可绘制矩形框）")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=spec_image,  # 直接使用生成的图像，不通过session_state
            height=img_height,  # 确保尺寸与图像一致
            width=img_width,
            drawing_mode="rect",
            key=f"canvas_{current_segment_key}",
            update_streamlit=True,
            display_toolbar=True
        )

        # 处理画框
        if canvas_result.json_data is not None:
            st.session_state.canvas_boxes = [
                {
                    "pixel": {
                        "left": obj["left"],
                        "top": obj["top"],
                        "width": obj["width"],
                        "height": obj["height"]
                    },
                    "label": None
                }
                for obj in canvas_result.json_data["objects"]
                if obj["type"] == "rect"
            ]

        # 3. 操作按钮
        st.markdown("#### 操作")
        button_row = st.columns([1, 1, 2])
        with button_row[0]:
            refresh_clicked = st.button("刷新频谱图", key="refresh_spec")
        with button_row[1]:
            save_clicked = st.button("保存画框标注", key=f"save_boxes_{current_segment_key}")
        with button_row[2]:
            skip_clicked = st.button("跳过本段", key=f"skip_box_{current_segment_key}")

        # 刷新逻辑
        if refresh_clicked:
            st.rerun()  # 直接刷新页面，避免状态缓存问题

    # 右侧标签管理
    with col_labels:
        st.markdown("### 框标签管理")
        species_list = st.session_state["dynamic_species_list"]
        if not species_list:
            st.warning("请先在左侧上传标签文件")
            return save_clicked, skip_clicked

        if st.session_state.canvas_boxes:
            for i, box in enumerate(st.session_state.canvas_boxes):
                st.markdown(f"#### 框 {i + 1}")
                time_freq = pixel_to_time_freq(box["pixel"])
                st.write(f"时间范围：{time_freq['start']:.2f} - {time_freq['end']:.2f} 秒")
                st.write(f"频率范围：{time_freq['min']:.0f} - {time_freq['max']:.0f} Hz")

                # 标签搜索和选择（保持原有逻辑）
                search_query = st.text_input(
                    "搜索标签", "", key=f"box_search_{i}",
                    placeholder="输入中文/拼音首字母"
                )
                filtered = []
                if search_query:
                    q = search_query.lower()
                    for label in species_list:
                        if q in label.lower() or q in get_pinyin_abbr(label).lower() or q in get_full_pinyin(label).lower():
                            filtered.append(label)
                else:
                    filtered = species_list

                selected_label = st.selectbox(
                    f"选择框 {i + 1} 的标签",
                    filtered,
                    index=filtered.index(box["label"]) if box["label"] in filtered else 0,
                    key=f"box_label_{i}"
                )
                if selected_label != box["label"]:
                    st.session_state.canvas_boxes[i]["label"] = selected_label
                    st.session_state.canvas_boxes = st.session_state.canvas_boxes

    return save_clicked, skip_clicked


# ======== 像素坐标→时间/频率转换函数 =========
def pixel_to_time_freq(pixel_coords):
    times = st.session_state.spec_params["times"]
    frequencies = st.session_state.spec_params["frequencies"]
    img_width, img_height = st.session_state.spec_params["img_size"]

    # 时间转换
    total_time = times[-1] - times[0]
    time_per_pixel = total_time / img_width
    start_time = times[0] + pixel_coords["left"] * time_per_pixel
    end_time = start_time + pixel_coords["width"] * time_per_pixel

    # 频率转换
    total_freq = frequencies[-1] - frequencies[0]
    freq_per_pixel = total_freq / img_height
    max_freq = frequencies[-1] - pixel_coords["top"] * freq_per_pixel
    min_freq = max_freq - pixel_coords["height"] * freq_per_pixel

    return {
        "start": round(max(0, start_time), 3),
        "end": round(min(5, end_time), 3),
        "min": round(max(0, min_freq), 1),
        "max": round(min(frequencies[-1], max_freq), 1)
    }


# ======== 音频处理主逻辑 =========
def process_audio():
    audio_state = st.session_state.audio_state
    output_dir = "annotated_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    # 初始化CSV
    try:
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=[
                "filename", "segment_index", "box_id",
                "start_time", "end_time", "min_freq", "max_freq", "label"
            ]).to_csv(csv_path, index=False, encoding='utf_8_sig')
        pd.read_csv(csv_path, encoding='utf_8_sig')  # 验证文件可读取
    except Exception as e:
        st.error(f"CSV文件错误：{str(e)}")
        return

    with st.sidebar:
        st.markdown("### 🎵 音频上传")
        uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True, key="audio_files")
        st.markdown("### 📥 下载结果")
        
        # 优化下载逻辑：仅当文件存在时显示下载按钮
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, "rb") as f:
                st.download_button(
                    "📄 下载标注结果", 
                    f, 
                    "annotations.csv", 
                    "text/csv; charset=utf-8",
                    key="download_csv"
                )
        
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 1:  # 排除CSV文件
            with zipfile.ZipFile(zip_buf := BytesIO(), "w") as zf:
                for f in os.listdir(output_dir):
                    if f.endswith(".wav"):
                        zf.write(os.path.join(output_dir, f), f)
            zip_buf.seek(0)
            st.download_button(
                "🎵 下载音频片段", 
                zip_buf, 
                "annotated_segments.zip", 
                "application/zip",
                key="download_audio"
            )

    if not uploaded_files:
        st.info("请先上传音频文件")
        return

    # 处理未标注音频
    unprocessed = [f for f in uploaded_files if f.name not in audio_state["processed_files"]]
    if unprocessed:
        audio_file = unprocessed[0]
        y, sr = load_audio(audio_file)
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / 5.0))

        # 初始化分段信息
        if audio_file.name not in audio_state["segment_info"]:
            audio_state["segment_info"][audio_file.name] = {"current_seg": 0, "total_seg": total_segments}
        seg_idx = audio_state["segment_info"][audio_file.name]["current_seg"]
        current_segment_key = f"{audio_file.name}_{seg_idx}"

        # 切换音频/分段时重置状态
        if (audio_state["last_audio_file"] != audio_file.name or
                audio_state["last_seg_idx"] != seg_idx):
            st.session_state.current_selected_labels = set()
            st.session_state.canvas_boxes = []
            audio_state["last_audio_file"] = audio_file.name
            audio_state["last_seg_idx"] = seg_idx

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        start_sec, end_sec = seg_idx * 5.0, min((seg_idx + 1) * 5.0, total_duration)
        
        # 时间取整优化：开始时间向下取整，结束时间向上取整
        start_sec_floor = np.floor(start_sec * 10) / 10  # 保留1位小数向下取整
        end_sec_ceil = np.ceil(end_sec * 10) / 10        # 保留1位小数向上取整
        segment_y = y[int(start_sec_floor * sr):int(end_sec_ceil * sr)]  # 使用取整后的时间

        # 标注模式处理
        if st.session_state.annotation_mode == "频谱图画框":
            save_clicked, skip_clicked = spectral_annotation_component(segment_y, sr, current_segment_key)

            if save_clicked:
                if not st.session_state.canvas_boxes:
                    st.warning("请先绘制至少一个框")
                    return
                if any(box["label"] is None for box in st.session_state.canvas_boxes):
                    st.warning("请为所有框添加标签")
                    return

                # 保存音频片段
                base_name = os.path.splitext(audio_file.name)[0]
                unique_id = uuid.uuid4().hex[:8]
                segment_filename = f"{base_name}_seg{seg_idx}_{unique_id}.wav"
                segment_path = os.path.join(output_dir, segment_filename)
                sf.write(segment_path, segment_y, sr)

                # 保存标注到CSV
                entries = []
                for box_id, box in enumerate(st.session_state.canvas_boxes):
                    time_freq = pixel_to_time_freq(box["pixel"])
                    # 时间取整优化
                    entries.append({
                        "filename": audio_file.name,
                        "segment_index": segment_filename,
                        "box_id": box_id,
                        "start_time": np.floor(time_freq["start"] * 10) / 10,  # 向下取整
                        "end_time": np.ceil(time_freq["end"] * 10) / 10,      # 向上取整
                        "min_freq": time_freq["min"],
                        "max_freq": time_freq["max"],
                        "label": box["label"]
                    })
                pd.DataFrame(entries).to_csv(
                    csv_path, mode='a', header=False, index=False, encoding='utf_8_sig'
                )

                # 更新状态
                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                if audio_state["segment_info"][audio_file.name]["current_seg"] >= total_segments:
                    audio_state["processed_files"].add(audio_file.name)
                st.success(f"成功保存 {len(entries)} 个框标注！")
                st.balloons()
                st.rerun()

            if skip_clicked:
                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                if audio_state["segment_info"][audio_file.name]["current_seg"] >= total_segments:
                    audio_state["processed_files"].add(audio_file.name)
                st.rerun()

        else:
            # 分段标注逻辑
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
                    st.image(generate_spectrogram_image(*generate_spectrogram_data(segment_y, sr)), caption="频谱图",
                             use_container_width=True)

            with col_labels:
                col_save, col_skip = annotation_labels_component(current_segment_key)
                if col_save and st.button("保存分段标注", key=f"save_seg_{current_segment_key}"):
                    save_segment_annotation(audio_file, seg_idx, start_sec_floor, end_sec_ceil, segment_y, sr, output_dir)
                if col_skip and st.button("跳过本段", key=f"skip_seg_{current_segment_key}"):
                    audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                    if audio_state["segment_info"][audio_file.name]["current_seg"] >= total_segments:
                        audio_state["processed_files"].add(audio_file.name)
                    st.rerun()

    else:
        st.success("🎉 所有音频标注完成！")

    st.session_state.audio_state = audio_state


# ======== 分段标注保存函数 =========
def save_segment_annotation(audio_file, seg_idx, start_sec, end_sec, segment_y, sr, output_dir):
    csv_path = os.path.join(output_dir, "annotations.csv")
    if not st.session_state.current_selected_labels:
        st.warning("请至少选择一个标签")
        return

    base_name = os.path.splitext(audio_file.name)[0]
    unique_id = uuid.uuid4().hex[:8]
    segment_filename = f"{base_name}_seg{seg_idx}_{unique_id}.wav"
    segment_path = os.path.join(output_dir, segment_filename)
    sf.write(segment_path, segment_y, sr)

    # 时间取整优化
    entry = {
        "filename": audio_file.name,
        "segment_index": segment_filename,
        "box_id": None,
        "start_time": start_sec,  # 已取整
        "end_time": end_sec,      # 已取整
        "min_freq": None,
        "max_freq": None,
        "label": ",".join(st.session_state.current_selected_labels)
    }
    pd.DataFrame([entry]).to_csv(csv_path, mode='a', header=False, index=False, encoding='utf_8_sig')

    audio_state = st.session_state.audio_state
    audio_state["segment_info"][audio_file.name]["current_seg"] += 1
    if audio_state["segment_info"][audio_file.name]["current_seg"] >= audio_state["segment_info"][audio_file.name]["total_seg"]:
        audio_state["processed_files"].add(audio_file.name)
    st.success(f"成功保存分段标注！")
    st.balloons()
    st.rerun()


# ======== 分段标注标签组件 =========
def annotation_labels_component(current_segment_key):
    species_list = st.session_state["dynamic_species_list"]
    with st.container():
        st.markdown("### 物种标签（可多选）")
        if not species_list:
            st.warning("请先在左侧上传标签文件")
            return None, None

        search_query = st.text_input("🔍 搜索标签", "", key=f"search_{current_segment_key}")
        cache_key = f"{current_segment_key}_{search_query}"

        if cache_key not in st.session_state.filtered_labels_cache:
            filtered_species = []
            if search_query:
                search_lower = search_query.lower()
                for label in species_list:
                    if (search_lower in label.lower() or
                            search_lower in get_pinyin_abbr(label) or
                            search_lower in get_full_pinyin(label)):
                        filtered_species.append(label)
            else:
                filtered_species = species_list.copy()
            st.session_state.filtered_labels_cache[cache_key] = filtered_species

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
        col_save, col_skip = st.columns(2)
        return col_save, col_skip


if __name__ == "__main__":
    label_management_component()
    process_audio()
