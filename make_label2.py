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
# 确保pinyin库正确导入（若导入失败会明确提示）
try:
    from pinyin import pinyin
except ImportError:
    st.error("❌ 未安装pinyin库，请先运行：pip install pinyin")
    st.stop()


# ======== 核心修复：拼音首字母转换（解决显示错误） =========
class PinyinHandler:
    @staticmethod
    def get_initial(char):
        """
        修复：正确提取单个汉字的拼音首字母
        例："鸳" → "y"，"鸯" → "y"，非汉字返回空
        """
        if len(char) != 1:
            return ""  # 非单个字符返回空
        
        # 仅处理汉字（忽略字母、数字、符号）
        if not '\u4e00' <= char <= '\u9fff':
            return ""
        
        try:
            # 调用pinyin库获取拼音（如"鸳" → [['yuan']]）
            py_result = pinyin(char)
            # 提取第一个拼音的首字母（小写）
            if py_result and isinstance(py_result[0], list) and py_result[0]:
                first_pinyin = py_result[0][0].lower()
                return first_pinyin[0] if first_pinyin else ""
            return ""
        except:
            return ""  # 转换失败返回空

    @staticmethod
    def label_to_initial(label):
        """
        修复：将标签转换为拼音首字母串
        例："鸳鸯" → "yy"，"鸳鸯_3型" → "yy"
        """
        # 仅提取标签中汉字的首字母，忽略其他字符
        return ''.join([PinyinHandler.get_initial(c) for c in label if '\u4e00' <= c <= '\u9fff'])


# ======== 修复：模糊搜索函数（基于正确的首字母） =========
def fuzzy_search(labels, query):
    if not query:
        return labels
    
    query_clean = query.lower().strip()
    matched = []
    
    for label in labels:
        # 提取标签的拼音首字母串（如"鸳鸯" → "yy"）
        label_initial = PinyinHandler.label_to_initial(label)
        # 标签原始字符（小写）
        label_lower = label.lower()
        
        # 规则1：拼音首字母完全匹配（如"yy"匹配"鸳鸯"）
        if query_clean == label_initial:
            matched.append((label, 3))  # 最高优先级
            continue
        
        # 规则2：拼音首字母包含查询（如"y"匹配"鸳鸯"）
        if query_clean in label_initial:
            matched.append((label, 2))
            continue
        
        # 规则3：标签包含查询字符（如"鸳"匹配"鸳鸯"）
        if query_clean in label_lower:
            matched.append((label, 1))
            continue
    
    # 按优先级排序，去重后返回
    if matched:
        unique_matched = {label: prio for label, prio in matched}
        return sorted(unique_matched.keys(), key=lambda x: -unique_matched[x])
    return []


# ======== 其他函数保持不变（仅修改标注组件的显示逻辑） =========
def annotation_labels_component(current_segment_key):
    species_list = st.session_state["dynamic_species_list"]
    col_labels = st.container()

    with col_labels:
        st.markdown("### 物种标签（可多选）")
        if not species_list:
            st.warning("请先在左侧上传标签文件")
            return None, None

        search_query = st.text_input(
            "🔍 搜索标签（示例：输入'yy'找'鸳鸯'，输入'鸳'也可）",
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

        # 修复：正确显示首字母（如"鸳鸯(首字母:yy)"）
        with st.container(height=300):
            for label in filtered_species:
                label_initial = PinyinHandler.label_to_initial(label)
                # 仅在首字母非空时显示
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


# ======== 其他代码保持不变（工具函数、音频处理等） =========
# （此处省略与之前相同的工具函数、音频处理逻辑等代码，保持不变）


# ======== 主流程 =========
if __name__ == "__main__":
    label_management_component()
    process_audio()
