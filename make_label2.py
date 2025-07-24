# 修改后的process_audio函数中的保存部分
with col_save:
    if st.button("保存本段标注", key=f"save_{current_segment_key}"):
        # 1. 检查标签
        if not st.session_state.current_selected_labels:
            st.warning("❗请至少选择一个标签")
            return

        # 2. 确保输出目录存在
        try:
            os.makedirs(output_dir, exist_ok=True)
            # 测试目录是否可写
            test_file = os.path.join(output_dir, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            st.error(f"无法写入目录 {output_dir}: {str(e)}")
            return

        try:
            # 3. 生成安全的文件名
            base_name = os.path.splitext(os.path.basename(audio_file.name))[0]
            # 移除文件名中的非法字符
            safe_base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            unique_id = uuid.uuid4().hex[:8]
            segment_filename = f"{safe_base_name}_seg{seg_idx}_{unique_id}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            
            # 确保音频数据有效
            if len(segment_y) == 0:
                st.error("音频片段为空，无法保存")
                return

            # 4. 保存音频文件
            with sf.SoundFile(segment_path, 'w', samplerate=sr, channels=1) as f:
                f.write(segment_y)

            # 5. 准备CSV条目
            clean_labels = [label.replace("/", "").replace("\\", "") for label in
                           st.session_state.current_selected_labels]
            entry = {
                "filename": audio_file.name,
                "segment_index": segment_filename,
                "start_time": round(start_sec, 3),
                "end_time": round(end_sec, 3),
                "labels": ",".join(clean_labels)
            }

            # 6. 更新CSV
            try:
                if os.path.exists(csv_path):
                    df_old = pd.read_csv(csv_path)
                else:
                    df_old = pd.DataFrame(columns=["filename", "segment_index", "start_time", "end_time", "labels"])
                
                df_combined = pd.concat([df_old, pd.DataFrame([entry])], ignore_index=True)
                df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
            except Exception as e:
                st.error(f"CSV保存失败: {str(e)}")
                return

            # 7. 更新状态
            current_segment_info = audio_state["segment_info"].get(audio_file.name, {})
            if current_segment_info.get("current_seg", 0) != seg_idx:
                st.error("片段索引不匹配，可能已被修改，请重试")
                return

            if seg_idx + 1 < total_segments:
                audio_state["segment_info"][audio_file.name]["current_seg"] += 1
            else:
                audio_state["processed_files"].add(audio_file.name)
                audio_state["current_index"] += 1

            st.success(f"标注已保存到: {segment_path}")
            st.rerun()

        except Exception as e:
            st.error(f"保存失败: {str(e)}")
            # 打印更详细的错误信息
            st.text(f"错误详情: {repr(e)}")
            if os.path.exists(segment_path):
                st.text(f"文件已存在: {os.path.getsize(segment_path)} bytes")
