import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import re
import platform


# 检测操作系统并设置中文字体
def set_chinese_font():
    system = platform.system()

    if system == 'Windows':
        # Windows系统
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
    elif system == 'Darwin':
        # macOS系统
        matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC']
        matplotlib.rcParams['axes.unicode_minus'] = False
    elif system == 'Linux':
        # Linux系统
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False

    # 添加更多的备选字体
    matplotlib.rcParams['font.sans-serif'] += ['SimHei', 'Microsoft YaHei',
                                               'KaiTi', 'FangSong', 'STHeiti']

# 设置中文
set_chinese_font()

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# ================ 页面设置 ================
st.set_page_config(
    page_title="多维指标分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E40AF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
    }
    .stDownloadButton button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<div class="main-header">📊 多维指标分析系统</div>', unsafe_allow_html=True)

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results_cache' not in st.session_state:
    st.session_state.results_cache = {}

# ================ 工具函数 ================

def clean_group_name(name):
    """清洗公司名称，去除空格和特殊字符，统一格式"""
    if pd.isna(name) or str(name).strip() == "":
        return "无"

    # 转换为字符串并去除首尾空格
    name_str = str(name).strip()

    # 替换常见的全角空格和特殊空格为普通空格
    name_str = name_str.replace('　', ' ')  # 全角空格
    name_str = name_str.replace('\t', ' ')  # 制表符
    name_str = name_str.replace('\n', ' ')  # 换行符

    # 去除多余的空格（多个空格合并为一个）
    name_str = re.sub(r'\s+', ' ', name_str)

    return name_str.strip()

def get_column_info(df, column_index):
    """获取列信息"""
    if column_index < len(df.columns):
        return df.columns[column_index]
    else:
        st.error(f"列索引{column_index}超出范围，文件只有{len(df.columns)}列")
        return None

def calculate_all_metrics(df, group_column_idx, group_column_name):
    """计算所有指标"""
    # 列索引设置
    col_7_idx = 6  # 质检列
    col_8_idx = 7  # 问题数列
    col_22_idx = 21  # 是否整改列
    col_23_idx = 22  # 是否问责列
    col_32_idx = 31  # 非空判断列

    # 检查列是否存在
    for idx in [group_column_idx, col_7_idx, col_8_idx, col_22_idx, col_23_idx, col_32_idx]:
        if idx >= len(df.columns):
            st.error(f"列索引{idx}超出范围，文件只有{len(df.columns)}列")
            return pd.DataFrame()

        # 从指定列提取，清洗并去重
        df_copy = df.copy()

        # 清洗
        df_copy['清洗后分组'] = df_copy.iloc[:, group_column_idx].apply(clean_group_name)

        # 获取唯一的分组名称
        groups = df_copy['清洗后分组'].unique().tolist()

        # 移除可能存在的重复（由于清洗后可能重复）
        groups = list(set(groups))

    results = []

    for group in groups:
        # 其他维度：使用清洗后的列进行筛选
        df_copy = df.copy()
        df_copy['清洗后维护单位'] = df_copy.iloc[:, group_column_idx].apply(clean_group_name)
        group_data = df_copy[df_copy['清洗后维护单位'] == group]

        # 计算各项指标
        metrics = calculate_single_group_metrics(group_data, col_7_idx, col_8_idx,
                                                 col_22_idx, col_23_idx, col_32_idx)

        results.append({group_column_name: group, **metrics})

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)

    # 排序
    return sort_results_df(results_df, group_column_name)


def calculate_single_group_metrics(group_data, col_7_idx, col_8_idx, col_22_idx, col_23_idx, col_32_idx):
    """计算单个分组的指标"""
    # 1. 计算问责扣罚率
    denominator_penalty = group_data[group_data.iloc[:, col_32_idx].notna()].shape[0]
    numerator_penalty = group_data[group_data.iloc[:, col_23_idx] == "是"].shape[0]
    penalty_rate = (denominator_penalty / numerator_penalty * 100) if denominator_penalty > 0 else 0

    # 2. 计算质检通过率
    denominator_quality = group_data.shape[0]
    numerator_quality = group_data[group_data.iloc[:, col_7_idx] == "是"].shape[0]
    quality_rate = (numerator_quality / denominator_quality * 100) if denominator_quality > 0 else 0

    # 3. 计算问题比
    try:
        group_data_col8_numeric = pd.to_numeric(group_data.iloc[:, col_8_idx], errors='coerce')
        total_issues = group_data_col8_numeric.sum()
        issue_ratio = total_issues / denominator_quality if denominator_quality > 0 else 0
    except Exception:
        issue_ratio = 0
        total_issues = 0

    # 4. 计算整改率
    denominator_rectification = group_data[group_data.iloc[:, col_7_idx] == "否"].shape[0]
    numerator_rectification = group_data[group_data.iloc[:, col_22_idx] == "是"].shape[0]
    rectification_rate = (
                numerator_rectification / denominator_rectification * 100) if denominator_rectification > 0 else 0

    # 5. 计算问责率
    denominator_accountability = group_data[group_data.iloc[:, col_7_idx] == "否"].shape[0]
    numerator_accountability = group_data[group_data.iloc[:, col_23_idx] == "是"].shape[0]
    accountability_rate = (
                numerator_accountability / denominator_accountability * 100) if denominator_accountability > 0 else 0

    return {
        "问责扣罚数量": numerator_penalty,
        "问责扣罚总数": denominator_penalty,
        "问责扣罚率(%)": round(penalty_rate, 2),
        "质检通过数量": numerator_quality,
        "质检总数": denominator_quality,
        "质检通过率(%)": round(quality_rate, 2),
        "问题总数": float(total_issues),
        "问题比": round(issue_ratio, 2),
        "整改完成数量": numerator_rectification,
        "需整改总数": denominator_rectification,
        "整改率(%)": round(rectification_rate, 2),
        "问责数量": numerator_accountability,
        "问责总数": denominator_accountability,
        "问责率(%)": round(accountability_rate, 2)
    }


def sort_results_df(results_df, group_column_name):
    """对结果DataFrame进行排序"""
    if group_column_name == "维护单位":
        # 对于维护单位，将"无"放在最后，其他按名称排序
        results_df["排序键"] = results_df["维护单位"].apply(lambda x: (1, x) if x == "无" else (0, x))
        results_df = results_df.sort_values("排序键").drop("排序键", axis=1)
    else:
        # 对于专业室和专项检查，将"无"放在最后，其他按名称排序
        results_df["排序键"] = results_df[group_column_name].apply(lambda x: (1, x) if x == "无" else (0, x))
        results_df = results_df.sort_values("排序键").drop("排序键", axis=1)

    return results_df


def create_bar_chart(results_df, group_column_name, metric_config, max_bars=30):
    """创建单个柱状图"""
    # 如果数据太多，只显示前max_bars个
    display_df = results_df.copy()
    if len(display_df) > max_bars:
        display_df = display_df.head(max_bars)

    fig, ax = plt.subplots(figsize=(12, 6))
    groups = display_df[group_column_name].tolist()
    metric_name = metric_config["name"]

    # 设置颜色
    colors = getattr(plt.cm, metric_config["color_map"])(
        np.linspace(metric_config["color_range"][0], metric_config["color_range"][1], len(groups))
    )

    # 创建柱状图
    bars = ax.bar(display_df[group_column_name], display_df[metric_name], color=colors)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{height:.2f}{"%" if "%" in metric_name else ""}',
                ha='center', va='bottom', fontsize=9)

    # 设置图表属性
    ax.set_xlabel(group_column_name, fontsize=11)
    ax.set_ylabel(metric_config["ylabel"], fontsize=11)
    ax.set_title(f"{metric_config['title']} (显示前{len(groups)}个)", fontsize=14, fontweight='bold')

    # 设置Y轴范围
    if len(display_df[metric_name]) > 0:
        max_rate = display_df[metric_name].max()
        y_max = max_rate * 1.2 if max_rate > 0 else 10
        ax.set_ylim(0, y_max)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def display_metrics_summary(results_df, group_column_name):
    """显示指标统计摘要"""
    metrics_summary = ["问责扣罚率(%)", "质检通过率(%)", "问题比", "整改率(%)", "问责率(%)"]

    for metric in metrics_summary:
        if metric in results_df.columns:
            st.markdown(f"**{metric}统计**")
            col1, col2, col3 = st.columns(3)

            with col1:
                max_idx = results_df[metric].idxmax()
                max_value = results_df.loc[max_idx, metric]
                max_group = results_df.loc[max_idx, group_column_name]
                st.metric(f"最高{metric}", f"{max_value:.2f}", f"{max_group}")

            with col2:
                min_idx = results_df[metric].idxmin()
                min_value = results_df.loc[min_idx, metric]
                min_group = results_df.loc[min_idx, group_column_name]
                st.metric(f"最低{metric}", f"{min_value:.2f}", f"{min_group}")

            with col3:
                avg_value = results_df[metric].mean()
                st.metric(f"平均{metric}", f"{avg_value:.2f}")


def display_dimension_analysis(df, dimension_name, column_idx, column_display_name):
    """显示维度分析"""
    st.header(f"{dimension_name}多维指标分析")

    # 检查列是否存在
    if len(df.columns) <= column_idx:
        st.error(f"Excel文件列数不足，请确认文件格式正确")
        st.info(f"文件实际列数：{len(df.columns)}，需要至少{column_idx + 1}列")
        return None

    # 获取列名
    column_name = get_column_info(df, column_idx)
    if column_name is None:
        st.error(f"第{column_idx + 1}列不存在")
        return None

    st.info(f"使用的{column_display_name}列：第{column_idx + 1}列({column_name})")

    # 计算指标
    results_df = calculate_all_metrics(df, column_idx, dimension_name)

    # 检查是否计算成功
    if results_df.empty:
        st.error("指标计算失败，请检查数据格式")
        return None

    # 显示结果表格
    st.subheader(f"各{dimension_name}多维指标统计结果")

    # 显示数据量信息
    st.info(f"共统计到 {len(results_df)} 个{dimension_name}")

    # 添加搜索功能
    search_term = st.text_input(f"搜索{dimension_name}", "", key=f"search_{dimension_name}")
    if search_term:
        filtered_df = results_df[results_df[dimension_name].str.contains(search_term, case=False, na=False)]
        st.info(f"找到 {len(filtered_df)} 个包含 '{search_term}' 的{dimension_name}")
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.dataframe(results_df, use_container_width=True)

    # 下载结果按钮
    csv = results_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label=f"下载{dimension_name}统计结果(CSV)",
        data=csv,
        file_name=f"{dimension_name}多维指标统计结果.csv",
        mime="text/csv",
        key=f"download_{dimension_name}"
    )

    # 指标配置
    metrics_config = [
        {"name": "问责扣罚率(%)", "title": f"各{dimension_name}问责扣罚率", "ylabel": "扣罚率(%)", "color_map": "Reds",
         "color_range": (0.3, 0.9)},
        {"name": "质检通过率(%)", "title": f"各{dimension_name}质检通过率", "ylabel": "通过率(%)",
         "color_map": "Greens", "color_range": (0.3, 0.9)},
        {"name": "问题比", "title": f"各{dimension_name}问题比", "ylabel": "问题比", "color_map": "Oranges",
         "color_range": (0.3, 0.9)},
        {"name": "整改率(%)", "title": f"各{dimension_name}整改率", "ylabel": "整改率(%)", "color_map": "Blues",
         "color_range": (0.3, 0.9)},
        {"name": "问责率(%)", "title": f"各{dimension_name}问责率", "ylabel": "问责率(%)", "color_map": "Purples",
         "color_range": (0.3, 0.9)}
    ]

    # 对于维护单位维度，如果数量太多，提供筛选功能
    display_df = results_df.copy()
    if dimension_name == "维护单位" and len(results_df) > 30:
        st.warning(f"检测到 {len(results_df)} 个维护单位，图表可能较为拥挤，默认显示前30个。")

        col1, col2 = st.columns(2)
        with col1:
            min_data_count = st.slider("最小数据量", 1, 100, 5, key=f"min_data_{dimension_name}")

        with col2:
            top_n = st.slider("显示前N个维护单位", 10, 100, 30, key=f"top_n_{dimension_name}")

        # 获取维护单位的数据量（原始数据）
        cleaned_series = df.iloc[:, column_idx].apply(clean_group_name)
        unit_counts = cleaned_series.value_counts()

        # 筛选维护单位
        filtered_units = []
        for unit in results_df["维护单位"]:
            if unit in unit_counts.index and unit_counts[unit] >= min_data_count:
                filtered_units.append(unit)

        # 按数据量排序并取前N个
        filtered_units = sorted(filtered_units,
                                key=lambda x: unit_counts[x] if x in unit_counts.index else 0,
                                reverse=True)[:top_n]

        display_df = results_df[results_df["维护单位"].isin(filtered_units)]
        st.info(f"显示 {len(display_df)} 个维护单位（数据量≥{min_data_count}，前{top_n}个）")

    # 创建柱状图
    st.subheader(f"{dimension_name}指标柱状图分析")

    # 分三行显示柱状图
    col1, col2 = st.columns(2)
    for i, config in enumerate(metrics_config[:2]):
        with (col1 if i == 0 else col2):
            st.subheader(config["title"])
            try:
                if config["name"] in display_df.columns:
                    fig = create_bar_chart(display_df, dimension_name, config)
                    st.pyplot(fig)
                else:
                    st.error(f"找不到'{config['name']}'列")
            except Exception as e:
                st.error(f"绘制{config['title']}图表时出错: {str(e)}")

    col1, col2 = st.columns(2)
    for i, config in enumerate(metrics_config[2:4]):
        with (col1 if i == 0 else col2):
            st.subheader(config["title"])
            try:
                if config["name"] in display_df.columns:
                    fig = create_bar_chart(display_df, dimension_name, config)
                    st.pyplot(fig)
                else:
                    st.error(f"找不到'{config['name']}'列")
            except Exception as e:
                st.error(f"绘制{config['title']}图表时出错: {str(e)}")

    # 最后一个指标单独一行
    config = metrics_config[4]
    st.subheader(config["title"])
    try:
        if config["name"] in display_df.columns:
            fig = create_bar_chart(display_df, dimension_name, config)
            st.pyplot(fig)
        else:
            st.error(f"找不到'{config['name']}'列")
    except Exception as e:
        st.error(f"绘制{config['title']}图表时出错: {str(e)}")

    # 显示统计摘要
    st.subheader(f"{dimension_name}维度综合统计摘要")
    display_metrics_summary(display_df, dimension_name)

    return results_df
# ================ 主程序 ================
# 设置页面标题
st.title('文件上传Demo')
# 创建文件上传组件
uploaded_file = st.file_uploader('选择文件', type=['xlsx', 'xls'])
if uploaded_file is not None:
    try:
        # 读取Excel文件
        df = pd.read_excel(uploaded_file, dtype=str)
        st.session_state.data = df

        st.success(f"文件上传成功！数据形状：{df.shape}")

        # 显示前几行数据
        with st.expander("查看数据预览"):
            st.dataframe(df.head())

        # 显示列信息
        with st.expander("查看列信息"):
            st.write("文件列信息：")
            for i, col in enumerate(df.columns):
                st.write(f"列{i + 1}: {col}")

        # 维度选择
        dimension = st.selectbox(
            "请选择分析维度",
            ["分公司维度", "维护单位维度", "专业室维度", "专项检查维度"]
        )

        if dimension == "分公司维度":
            # 分公司维度分析
            results_df = display_dimension_analysis(df, "分公司", 13, "分公司")

            if results_df is not None and not results_df.empty:
                with st.expander("查看分公司分布详情"):
                    cleaned_series = df.iloc[:, 13].apply(clean_group_name)
                    company_counts = cleaned_series.value_counts()

                    st.write(f"分公司总数: {len(company_counts)}")

                    if "无" in company_counts.index:
                        st.info(f"空白分公司（标记为'无'）的数据量: {company_counts['无']}")

                    unit_counts_df = pd.DataFrame({
                        "分公司": company_counts.index,
                        "数据量": company_counts.values,
                        "占比(%)": (company_counts.values / len(df) * 100).round(2)
                    })
                    st.dataframe(unit_counts_df, use_container_width=True)

        elif dimension == "维护单位维度":
            # 维护单位维度分析
            results_df = display_dimension_analysis(df, "维护单位", 19, "维护单位")

            # 对于维护单位，额外显示一些分析
            if results_df is not None and not results_df.empty:
                with st.expander("查看维护单位分布详情"):
                    cleaned_series = df.iloc[:, 19].apply(clean_group_name)
                    unit_counts = cleaned_series.value_counts()

                    st.write(f"维护单位总数: {len(unit_counts)}")

                    if "无" in unit_counts.index:
                        st.info(f"空白维护单位（标记为'无'）的数据量: {unit_counts['无']}")

                    unit_counts_df = pd.DataFrame({
                        "维护单位": unit_counts.index,
                        "数据量": unit_counts.values,
                        "占比(%)": (unit_counts.values / len(df) * 100).round(2)
                    })
                    st.dataframe(unit_counts_df, use_container_width=True)

        elif dimension == "专业室维度":
            # 专业室维度分析 (第16列，索引15)
            results_df = display_dimension_analysis(df, "专业室", 15, "专业室")

            # 额外显示专业室分布情况
            if results_df is not None and not results_df.empty:
                with st.expander("查看专业室分布详情"):
                    cleaned_series = df.iloc[:, 15].apply(clean_group_name)
                    room_counts = cleaned_series.value_counts()

                    st.write(f"专业室总数: {len(room_counts)}")

                    if "无" in room_counts.index:
                        st.info(f"空白专业室（标记为'无'）的数据量: {room_counts['无']}")

                    room_counts_df = pd.DataFrame({
                        "专业室": room_counts.index,
                        "数据量": room_counts.values,
                        "占比(%)": (room_counts.values / len(df) * 100).round(2)
                    })

                    st.dataframe(room_counts_df, use_container_width=True)

        elif dimension == "专项检查维度":
            # 专项检查维度分析 (第15列，索引14)
            results_df = display_dimension_analysis(df, "专项检查", 14, "专项检查")

            # 额外显示专项检查分布情况
            if results_df is not None and not results_df.empty:
                with st.expander("查看专项检查分布详情"):
                    cleaned_series = df.iloc[:, 14].apply(clean_group_name)
                    check_counts = cleaned_series.value_counts()

                    st.write(f"专项检查类型总数: {len(check_counts)}")

                    if "无" in check_counts.index:
                        st.info(f"空白专项检查（标记为'无'）的数据量: {check_counts['无']}")

                    check_counts_df = pd.DataFrame({
                        "专项检查": check_counts.index,
                        "数据量": check_counts.values,
                        "占比(%)": (check_counts.values / len(df) * 100).round(2)
                    })

                    st.dataframe(check_counts_df, use_container_width=True)

    except Exception as e:
        st.error(f"读取文件或分析数据时出错：{str(e)}")
        st.error(f"详细错误信息：{repr(e)}")

        # 显示调试信息
        with st.expander("查看调试信息"):
            if 'df' in locals():
                st.write("数据框列名：")
                st.write(list(df.columns))
                st.write("前5行数据：")
                st.write(df.head())

else:
    st.info("请上传Excel文件以开始分析")
