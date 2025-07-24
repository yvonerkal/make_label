# 在原有代码基础上新增以下内容

# ======== 新增画框标注相关函数 =========
@st.cache_data(show_spinner=False)
def generate_interactive_spectrogram(y, sr):
    """生成交互式频谱图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title="Spectrogram (Click and drag to annotate)")
    return fig, ax

def draw_rectangle(fig, ax, start_time, end_time, low_freq, high_freq, label=None):
    """在频谱图上绘制矩形框"""
    width = end_time - start_time
    height = high_freq - low_freq
    rect = plt.Rectangle((start_time, low_freq), width, height,
                        linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if label:
        ax.text(start_time, high_freq, label, color='white', 
               backgroundcolor='red', fontsize=10)
    return fig

# ======== 修改音频处理逻辑 =========
def process_audio():
    audio_state = st.session_state.audio_state
    output_dir = "uploaded_audios"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "annotations.csv")

    # 新增标注模式切换
    annotation_mode = st.sidebar.radio("标注模式", ["分段标注", "频谱图画框"], index=0)
    
    # ... (保留原有的文件上传和下载逻辑)

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
            # 初始化画框标注状态
            if "boxes" not in st.session_state:
                st.session_state.boxes = []

        st.header(f"标注音频: {audio_file.name} - 第 {seg_idx + 1}/{total_segments} 段")
        
        if annotation_mode == "频谱图画框":
            process_spectral_annotation(y, sr, audio_file, seg_idx, output_dir)
        else:
            process_segment_annotation(y, sr, audio_file, seg_idx, total_duration, current_segment_key, output_dir)

    else:
        st.success("🎉 所有音频标注完成！")

    st.session_state.audio_state = audio_state

def process_spectral_annotation(y, sr, audio_file, seg_idx, output_dir):
    """处理频谱图画框标注"""
    col_main, col_labels = st.columns([3, 1])
    
    with col_main:
        st.subheader("🎧 频谱图画框标注")
        start_sec, end_sec = seg_idx * 5.0, min((seg_idx + 1) * 5.0, librosa.get_duration(y=y, sr=sr))
        segment_y = y[int(start_sec * sr):int(end_sec * sr)]
        
        # 生成交互式频谱图
        fig, ax = generate_interactive_spectrogram(segment_y, sr)
        
        # 显示已有标注框
        for box in st.session_state.get("boxes", []):
            fig = draw_rectangle(fig, ax, box["start_time"], box["end_time"], 
                               box["low_freq"], box["high_freq"], box["label"])
        
        # 显示频谱图
        st.pyplot(fig)
        
        # 音频播放器
        audio_bytes = BytesIO()
        sf.write(audio_bytes, segment_y, sr, format='WAV')
        st.audio(audio_bytes, format="audio/wav")
        
        # 画框控件
        with st.form("box_form"):
            st.markdown("### 添加标注框")
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.number_input("开始时间(s)", min_value=0.0, max_value=5.0, step=0.1)
                end_time = st.number_input("结束时间(s)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            with col2:
                low_freq = st.number_input("最低频率(Hz)", min_value=0, max_value=sr//2, value=1000, step=100)
                high_freq = st.number_input("最高频率(Hz)", min_value=0, max_value=sr//2, value=3000, step=100)
            
            if st.form_submit_button("添加标注框"):
                if end_time <= start_time:
                    st.error("结束时间必须大于开始时间")
                elif high_freq <= low_freq:
                    st.error("最高频率必须大于最低频率")
                else:
                    st.session_state.boxes.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "low_freq": low_freq,
                        "high_freq": high_freq,
                        "label": ""
                    })
                    st.rerun()
        
        # 撤销按钮
        if st.button("撤销上一个标注框") and st.session_state.boxes:
            st.session_state.boxes.pop()
            st.rerun()
    
    with col_labels:
        # 为每个框选择标签
        if st.session_state.boxes:
            st.markdown("### 标注框标签")
            for i, box in enumerate(st.session_state.boxes):
                if not box["label"]:
                    label = st.selectbox(
                        f"标注框{i+1} ({box['start_time']:.1f}s-{box['end_time']:.1f}s)",
                        st.session_state["dynamic_species_list"],
                        key=f"box_label_{i}"
                    )
                    st.session_state.boxes[i]["label"] = label
            
            # 保存所有标注
            if st.button("保存所有标注"):
                save_spectral_annotations(audio_file, seg_idx, start_sec, output_dir)
        else:
            st.info("请先添加标注框")

def save_spectral_annotations(audio_file, seg_idx, start_sec, output_dir):
    """保存频谱图标注结果"""
    try:
        # 为每个标注框保存单独的音频片段
        for i, box in enumerate(st.session_state.boxes):
            if not box["label"]:
                st.error(f"请为标注框{i+1}选择标签")
                return
            
            # 生成唯一文件名
            unique_id = uuid.uuid4().hex[:8]
            segment_filename = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}_box{i}_{unique_id}.wav"
            
            # 保存标注信息到CSV
            entry = {
                "filename": audio_file.name,
                "segment_index": segment_filename,
                "start_time": round(start_sec + box["start_time"], 3),
                "end_time": round(start_sec + box["end_time"], 3),
                "low_freq": box["low_freq"],
                "high_freq": box["high_freq"],
                "labels": box["label"]
            }
            
            # 添加到CSV文件
            csv_path = os.path.join(output_dir, "annotations.csv")
            df = pd.DataFrame([entry])
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
        
        st.success("标注保存成功！")
        st.session_state.boxes = []  # 清空当前标注
        st.rerun()
    
    except Exception as e:
        st.error(f"保存失败: {str(e)}")

# ... (保留原有的process_segment_annotation函数和其他代码)
