with col_save:
    if st.button("保存本段标注", key=f"save_{current_segment_key}"):
        # 1. 检查标签
        if not st.session_state.current_selected_labels:
            st.warning("❗请至少选择一个标签")
            continue
        
        # 2. 检查输出目录
        if not os.path.exists(output_dir):
            st.error(f"保存目录不存在：{output_dir}，请修改路径")
            continue
        
        try:
            # 3. 保存音频片段
            segment_filename = f"{os.path.splitext(audio_file.name)[0]}_seg{seg_idx}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            if len(segment_y) == 0:
                st.error("音频片段为空，无法保存")
                continue
            sf.write(segment_path, segment_y, sr)
            
            # 4. 准备CSV条目（清洗标签）
            clean_labels = [label.replace("/", "").replace("\\", "") for label in st.session_state.current_selected_labels]
            entry = {
                "filename": audio_file.name,
                "segment_index": segment_filename,
                "start_time": round(start_sec, 3),
                "end_time": round(end_sec, 3),
                "labels": ",".join(clean_labels)
            }
            
            # 5. 更新CSV
            if df_old.empty:
                df_combined = pd.DataFrame([entry])
            else:
                df_combined = pd.concat([df_old, pd.DataFrame([entry])], ignore_index=True)
            df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
            
            # 6. 更新状态（验证索引）
            current_segment_info = audio_state["segment_info"].get(audio_file.name, {})
            if current_segment_info.get("current_seg", 0) != seg_idx:
                st.error("片段已被修改，请重试")
                continue
            
            if seg_idx + 1 < total_segments:
                audio_state["segment_info"][audio_file.name]["current_seg"] = seg_idx + 1
            else:
                audio_state["processed_files"].add(audio_file.name)
                audio_state["current_index"] += 1
            
            st.success("标注已保存！")
            st.rerun()
            
        except PermissionError:
            st.error(f"无写入权限：请检查目录 '{output_dir}' 的权限")
        except FileNotFoundError:
            st.error(f"路径不存在：'{output_dir}'")
        except Exception as e:
            st.error(f"保存失败：{str(e)}（请检查文件是否被占用或格式正确）")
