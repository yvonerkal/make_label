# 保存、下载数据的方式
# 频谱图显示问题
# 保存分割数据时，开始时间点往前，结束时间点往后取整
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
    times = librosa.times_like(D, sr=sr)  # 时间轴：0-5秒（5秒片段）
    frequencies = librosa.fft_frequencies(sr=sr)  # 频率轴：0到sr/2（奈奎斯特频率）
    return D, times, frequencies


def generate_spectrogram_image(D, times, frequencies):
    """生成带坐标的频谱图（确保x/y轴范围明确）"""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)  # 固定尺寸
    img = librosa.display.specshow(
        D,
        sr=frequencies[-1] * 2,
        x_axis='time',
        y_axis='log',
        ax=ax
    )
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(frequencies[0], frequencies[-1])
    ax.set_title('频谱图（可画框标注）')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()

    buf = io.BytesIO()
    # 👉 改为白底不透明背景（加 facecolor='white'）
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='white')
    buf.seek(0)
    plt.close(fig)
    if img.mode != "RGB":
        img = img.convert("RGB")


    return Image.open(buf)



@st.cache_data(show_spinner=False)
def generate_waveform_image(y, sr):
    plt.figure(figsize=(12, 3), dpi=100)
    librosa.display.waveshow(y, sr=sr)
    plt.title('波形图')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
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
    }
if "filtered_labels_cache" not in st.session_state:
    st.session_state.filtered_labels_cache = {}
if "canvas_boxes" not in st.session_state:  # 存储带标签的画框：{像素坐标, 时间频率, 标签}
    st.session_state.canvas_boxes = []
if "spec_params" not in st.session_state:  # 存储频谱图参数（用于坐标转换）
    st.session_state.spec_params = {"times": None, "frequencies": None, "img_size": (0, 0)}
if "spec_image" not in st.session_state:  # 缓存频谱图以避免重复生成
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
    # 生成频谱图数据（时间、频率范围）
    D, times, frequencies = generate_spectrogram_data(y, sr)

    # 缓存频谱图，避免重复生成
    if st.session_state.spec_image is None:
        spec_image = generate_spectrogram_image(D, times, frequencies)
        st.session_state.spec_image = spec_image
    else:
        spec_image = st.session_state.spec_image

    st.session_state.spec_params = {
        "times": times,  # 0-5秒的时间轴
        "frequencies": frequencies,  # 频率轴（0到sr/2）
        "img_size": (spec_image.width, spec_image.height)  # 频谱图尺寸（像素）
    }

    # 主区域布局：左侧为操作区（固定结构），右侧为标签区（可滚动）
    col_main, col_labels = st.columns([3, 1])

    with col_main:
        st.subheader("🎧 频谱图画框标注（点击画布绘制矩形）")
        

        # 1. 音频播放移到频谱图上方
        st.markdown("#### 音频播放")
        audio_bytes = BytesIO()
        sf.write(audio_bytes, y, sr, format='WAV')
        st.audio(audio_bytes, format="audio/wav", start_time=0)

        # 2. 频谱图画布区域
        # DEBUG: 临时显示频谱图
        st.image(spec_image, caption="频谱图 DEBUG 显示", use_column_width=True)
        st.markdown("#### 频谱图（可绘制矩形框）")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # 半透明橙色
            stroke_width=2,
            stroke_color="#FF0000",  # 红色边框
            background_color="#eee",
            background_image=spec_image,
            height=spec_image.height,  # 画布高度=频谱图高度
            width=spec_image.width,  # 画布宽度=频谱图宽度
            drawing_mode="rect",  # 仅允许画矩形
            key=f"canvas_{current_segment_key}",
            update_streamlit=True,  # 启用自动更新
            display_toolbar=True  # 显示工具栏
        )

        # 处理画布上的画框
        if canvas_result.json_data is not None:
            st.session_state.canvas_boxes = [
                {
                    "pixel": {  # 像素坐标（画布上的位置）
                        "left": obj["left"],
                        "top": obj["top"],
                        "width": obj["width"],
                        "height": obj["height"]
                    },
                    "label": None  # 初始无标签
                }
                for obj in canvas_result.json_data["objects"]
                if obj["type"] == "rect"
            ]

        # 3. 刷新按钮和操作按钮组（固定在频谱图下方）
        st.markdown("#### 操作")
        button_row = st.columns([1, 1, 2])  # 调整按钮宽度比例
        with button_row[0]:
            refresh_clicked = st.button("刷新频谱图", key="refresh_spec")
        with button_row[1]:
            save_clicked = st.button("保存画框标注", key=f"save_boxes_{current_segment_key}")
        with button_row[2]:
            skip_clicked = st.button("跳过本段", key=f"skip_box_{current_segment_key}")

        # 处理刷新逻辑
        if refresh_clicked:
            st.session_state.spec_image = None
            st.rerun()

    # 右侧标签管理区域（可滚动，不影响左侧按钮位置）
    with col_labels:
        st.markdown("### 框标签管理")
        species_list = st.session_state["dynamic_species_list"]
        if not species_list:
            st.warning("请先在左侧上传标签文件")
            return save_clicked, skip_clicked

        # 显示所有画框并关联标签
        if st.session_state.canvas_boxes:
            for i, box in enumerate(st.session_state.canvas_boxes):
                st.markdown(f"#### 框 {i + 1}")

                # 转换像素坐标为实际时间和频率
                time_freq = pixel_to_time_freq(box["pixel"])
                st.write(f"时间范围：{time_freq['start']:.2f} - {time_freq['end']:.2f} 秒")
                st.write(f"频率范围：{time_freq['min']:.0f} - {time_freq['max']:.0f} Hz")

                # 为当前框选择标签
                search_query = st.text_input(
                    "搜索标签", "", key=f"box_search_{i}",
                    placeholder="输入中文/拼音首字母"
                )
                # 过滤标签（支持中文、拼音首字母、全拼）
                filtered = []
                if search_query:
                    q = search_query.lower()
                    for label in species_list:
                        if q in label.lower() or q in get_pinyin_abbr(label).lower() or q in get_full_pinyin(
                                label).lower():
                            filtered.append(label)
                else:
                    filtered = species_list

                # 选择标签并保存
                selected_label = st.selectbox(
                    f"选择框 {i + 1} 的标签",
                    filtered,
                    index=filtered.index(box["label"]) if box["label"] in filtered else 0,
                    key=f"box_label_{i}"
                )
                # 更新当前框的标签
                if selected_label != box["label"]:
                    st.session_state.canvas_boxes[i]["label"] = selected_label
                    st.session_state.canvas_boxes = st.session_state.canvas_boxes  # 触发状态更新

    return save_clicked, skip_clicked


# ======== 像素坐标→时间/频率转换函数 =========
def pixel_to_time_freq(pixel_coords):
    """将画布上的像素坐标转换为实际的时间（秒）和频率（Hz）"""
    times = st.session_state.spec_params["times"]
    frequencies = st.session_state.spec_params["frequencies"]
    img_width, img_height = st.session_state.spec_params["img_size"]

    # 时间范围（x轴）：画布左→右 = 0→5秒
    total_time = times[-1] - times[0]  # 总时长（5秒）
    time_per_pixel = total_time / img_width  # 每个像素对应的时间
    start_time = times[0] + pixel_coords["left"] * time_per_pixel
    end_time = start_time + pixel_coords["width"] * time_per_pixel

    # 频率范围（y轴）：画布上→下 = 高频→低频（因为频谱图y轴是倒的）
    total_freq = frequencies[-1] - frequencies[0]  # 总频率范围（0到sr/2）
    freq_per_pixel = total_freq / img_height  # 每个像素对应的频率
    max_freq = frequencies[-1] - pixel_coords["top"] * freq_per_pixel
    min_freq = max_freq - pixel_coords["height"] * freq_per_pixel

    return {
        "start": round(max(0, start_time), 3),  # 确保不小于0
        "end": round(min(5, end_time), 3),  # 确保不超过5秒
        "min": round(max(0, min_freq), 1),
        "max": round(min(frequencies[-1], max_freq), 1)
    }


# ======== 音频处理主逻辑 =========
def process_audio():
    audio_state = st.session_state.audio_state
    output_dir = "annotated_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    # 初始化CSV（包含画框的时间、频率、标签）
    try:
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=[
                "filename", "segment_index", "box_id",
                "start_time", "end_time", "min_freq", "max_freq", "label"
            ]).to_csv(csv_path, index=False, encoding='utf_8_sig')
        try:
            df_old = pd.read_csv(csv_path, encoding='utf_8_sig')
        except Exception as e:
            st.error(f"CSV文件错误：{str(e)}")
            return
    except Exception as e:
        st.error(f"CSV文件错误：{str(e)}")
        return

    with st.sidebar:
        st.markdown("### 🎵 音频上传")
        uploaded_files = st.file_uploader("上传音频文件 (.wav)", type=["wav"], accept_multiple_files=True, key="audio_files")
        st.markdown("### 📥 下载结果")
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button("📄 下载标注结果", f, "annotations.csv", "text/csv; charset=utf-8")
        if os.path.exists(output_dir):
            with zipfile.ZipFile(zip_buf := BytesIO(), "w") as zf:
                for f in os.listdir(output_dir):
                    if f.endswith(".wav"):
                        zf.write(os.path.join(output_dir, f), f)
            zip_buf.seek(0)
            st.download_button("🎵 下载音频片段", zip_buf, "annotated_segments.zip", "application/zip")

    if not uploaded_files:
        st.info("请先上传音频文件")
        return

    # 获取未处理的音频
    unprocessed = [f for f in uploaded_files if f.name not in audio_state["processed_files"]]

    if unprocessed:
        audio_file = unprocessed[0]
        y, sr = load_audio(audio_file)
        total_duration = librosa.get_duration(y=y, sr=sr)
        total_segments = int(np.ceil(total_duration / 5.0))

        # 初始化当前音频的分段信息
        if audio_file.name not in audio_state["segment_info"]:
            audio_state["segment_info"][audio_file.name] = {"current_seg": 0, "total_seg": total_segments}
        seg_idx = audio_state["segment_info"][audio_file.name]["current_seg"]
        current_segment_key = f"{audio_file.name}_{seg_idx}"

        # 切换音频/分段时重置状态
        if (audio_state["last_audio_file"] != audio_file.name or
                audio_state["last_seg_idx"] != seg_idx):
            st.session_state.current_selected_labels = set()
            st.session_state.canvas_boxes = []
            st.session_state.spec_image = None  # 重置频谱图缓存
            audio_state["last_audio_file"] = audio_file.name
            audio_state["last_seg_idx"] = seg_idx

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        start_sec, end_sec = seg_idx * 5.0, min((seg_idx + 1) * 5.0, total_duration)
        segment_y = y[int(start_sec * sr):int(end_sec * sr)]  # 当前5秒片段

        # 根据模式选择标注方式
        if st.session_state.annotation_mode == "频谱图画框":
            save_clicked, skip_clicked = spectral_annotation_component(segment_y, sr, current_segment_key)

            # 保存画框标注
            if save_clicked:
                # 检查是否所有框都有标签
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

                # 保存每个框的信息到CSV
                entries = []
                for box_id, box in enumerate(st.session_state.canvas_boxes):
                    time_freq = pixel_to_time_freq(box["pixel"])
                    entries.append({
                        "filename": audio_file.name,
                        "segment_index": segment_filename,
                        "box_id": box_id,
                        "start_time": time_freq["start"],
                        "end_time": time_freq["end"],
                        "min_freq": time_freq["min"],
                        "max_freq": time_freq["max"],
                        "label": box["label"]
                    })
                pd.DataFrame(entries).to_csv(csv_path, mode='a', header=False, index=False, encoding='utf_8_sig')

                # 更新状态，进入下一段
                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                if audio_state["segment_info"][audio_file.name]["current_seg"] >= total_segments:
                    audio_state["processed_files"].add(audio_file.name)
                st.session_state.spec_image = None  # 重置频谱图缓存
                st.success(f"成功保存 {len(entries)} 个框标注！")
                st.balloons()
                st.rerun()

            # 跳过当前段
            if skip_clicked:
                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
                if audio_state["segment_info"][audio_file.name]["current_seg"] >= total_segments:
                    audio_state["processed_files"].add(audio_file.name)
                st.session_state.spec_image = None  # 重置频谱图缓存
                st.rerun()

        else:
            # 原有分段标注逻辑
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
                    save_segment_annotation(audio_file, seg_idx, start_sec, end_sec, segment_y, sr, output_dir)
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

    entry = {
        "filename": audio_file.name,
        "segment_index": segment_filename,
        "box_id": None,
        "start_time": round(start_sec, 3),
        "end_time": round(end_sec, 3),
        "min_freq": None,
        "max_freq": None,
        "label": ",".join(st.session_state.current_selected_labels)
    }
    pd.DataFrame([entry]).to_csv(csv_path, mode='a', header=False, index=False, encoding='utf_8_sig')

    audio_state = st.session_state.audio_state
    audio_state["segment_info"][audio_file.name]["current_seg"] += 1
    if audio_state["segment_info"][audio_file.name]["current_seg"] >= audio_state["segment_info"][audio_file.name][
        "total_seg"]:
        audio_state["processed_files"].add(audio_file.name)
    st.success(f"成功保存分段标注！")
    st.balloons()
    st.rerun()


# ======== 原有分段标注标签组件 =========
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
