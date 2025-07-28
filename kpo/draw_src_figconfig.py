"""
Matplotlib 绘图配置工具
支持多种预设布局 (manuscript/slides/PCI)，可自定义覆盖参数
"""

import matplotlib.pyplot as plt


# 预定义配置字典
CONFIGURATIONS = {
    # 单栏论文格式
    "manuscript_single": {
        "plot_width": 9.0,        # 总画布宽度 (cm)
        "margin_left": 1.4,       # 左侧边距 (cm)
        "margin_right": 0.3,      # 右侧边距 (cm)
        "margin_bottom": 1.2,     # 底部边距 (cm)
        "margin_top": 0.3,        # 顶部边距 (cm)
        "space_width": 1.5,       # 子图水平间距 (cm)
        "space_height": 1.0,      # 子图垂直间距 (cm)
        "subplot_ratio": 0.9,     # 子图高宽比 (height/width)
        "ftsize": 11             # 基础字体大小
    },
    # 双栏论文格式
    "manuscript_double": {
        "plot_width": 19.0,
        "margin_left": 1.4,
        "margin_right": 0.3,
        "margin_bottom": 1.2,
        "margin_top": 0.3,
        "space_width": 1.5,
        "space_height": 1.0,
        "subplot_ratio": 0.9,
        "ftsize": 11
    },
    # 单栏幻灯片格式
    "slides_single": {
        "plot_width": 9.0,
        "margin_left": 1.8,
        "margin_right": 0.3,
        "margin_bottom": 1.6,
        "margin_top": 0.3,
        "space_width": 1.5,
        "space_height": 1.0,
        "subplot_ratio": 0.9,
        "ftsize": 16
    },
    # 双栏幻灯片格式
    "slides_double": {
        "plot_width": 19.0,
        "margin_left": 1.8,
        "margin_right": 0.3,
        "margin_bottom": 1.6,
        "margin_top": 0.3,
        "space_width": 1.5,
        "space_height": 1.0,
        "subplot_ratio": 0.9,
        "ftsize": 16
    },
    # PCI 期刊单栏格式
    "PCI_single": {
        "plot_width": 6.7,
        "margin_left": 1.1,
        "margin_right": 0.2,
        "margin_bottom": 1.0,
        "margin_top": 0.2,
        "space_width": 1.0,
        "space_height": 1.0,
        "subplot_ratio": 0.9,
        "ftsize": 9
    },
    # PCI 期刊双栏格式
    "PCI_double": {
        "plot_width": 13.7,
        "margin_left": 1.1,
        "margin_right": 0.2,
        "margin_bottom": 1.0,
        "margin_top": 0.2,
        "space_width": 1.0,
        "space_height": 1.0,
        "subplot_ratio": 0.9,
        "ftsize": 9
    },
}


class PlotConfig2:
    """Matplotlib 绘图配置引擎 2.0版本
    
    功能特性：
    - 预置多种学术出版格式
    - 自动计算子图布局
    - 支持每行子图不同高度比例
    - 支持指定行绘制少于列数的子图
    - 支持行间和列间可变间距
    - 支持 LaTeX 数学公式渲染
    - 厘米单位直接输入
    - 灵活的参数覆盖机制
    
    使用示例：
    >>> config = PlotConfig(config_name='manuscript_single', nrow=3, ncol=3)
    >>> config.set_row_config(row=1, ncols=2)  # 第1行只画2个子图
    >>> fig, axs = config.get_multi()
    >>> plt.show()
    """
    
    def __init__(self, config_name='manuscript_single', nrow=1, ncol=1, **kwargs):
        """初始化绘图配置
        
        参数：
        config_name -- 预定义配置名称 (默认: manuscript_single)
        nrow        -- 子图行数 (默认: 1)
        ncol        -- 子图列数 (默认: 1)
        **kwargs    -- 可覆盖预定义配置的任意参数
        """
        
        # 获取基础配置
        config = CONFIGURATIONS.get(config_name, {}).copy()
        
        # 用户自定义参数覆盖
        config.update(kwargs)
        
        # 配置参数绑定到实例
        self.plot_width = config['plot_width']      # 总画布宽度 (cm)
        self.margin_left = config['margin_left']    # 左侧边距 (cm)
        self.margin_right = config['margin_right']  # 右侧边距 (cm)
        self.margin_bottom = config['margin_bottom'] # 底部边距 (cm)
        self.margin_top = config['margin_top']      # 顶部边距 (cm)
        self.ftsize = config['ftsize']              # 基础字号
        
        # 处理高度比例参数
        if isinstance(config['subplot_ratio'], (int, float)):
            self.subplot_ratio = [config['subplot_ratio']] * nrow
        else:
            self.subplot_ratio = config['subplot_ratio']
        
        # 处理行间垂直间距
        if isinstance(config['space_height'], (int, float)):
            self.space_height = [config['space_height']] * (nrow-1) if nrow > 1 else []
        else:
            if len(config['space_height']) != max(0, nrow-1):
                raise ValueError(f"space_height 需要 {max(0, nrow-1)} 个值，但得到 {len(config['space_height'])}")
            self.space_height = config['space_height']
        self.total_space_height = sum(self.space_height)
        
        # 处理列间水平间距
        if isinstance(config['space_width'], (int, float)):
            self.space_width = [config['space_width']] * (ncol-1) if ncol > 1 else []
        else:
            if len(config['space_width']) != max(0, ncol-1):
                raise ValueError(f"space_width 需要 {max(0, ncol-1)} 个值，但得到 {len(config['space_width'])}")
            self.space_width = config['space_width']
        self.total_space_width = sum(self.space_width)
        
        # 布局参数
        self.nrow = nrow
        self.ncol = ncol
        
        # 计算基本子图宽度（所有列等宽时的参考宽度）
        self.base_subplot_width = (self.plot_width 
                                 - self.margin_left - self.margin_right
                                 - self.total_space_width
                                 ) / ncol
        
        # 计算每行高度（基于基本宽度和比例）
        self.row_heights = [
            self.base_subplot_width * ratio 
            for ratio in self.subplot_ratio
        ]
        
        # 计算总画布高度
        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)

        # 每行配置
        self.row_config = {i: ncol for i in range(nrow)}
        self.row_subplot_widths = [self.base_subplot_width] * nrow
        
        # set_row_config 单独针对指定行进行调整
        self.row_width_scales = getattr(self, 'row_width_scales', {})
        self.row_extra_margins = getattr(self, 'row_extra_margins', {})
        
        # 配置 matplotlib 全局参数
        self._configure_matplotlib()
    
    def set_row_config(self, row, ncols, row_width_scale=1.0, row_extra_margin=0.0):
        """设置指定行的子图配置
        参数：
        row   -- 行索引(0开始)
        ncols -- 该行实际绘制的子图数量(必须 ≤ ncol)
        row_width_scale -- 该行总宽度相对于整个画布内容区域的比例(默认1.0)
        row_extra_margin -- 该行额外的左右边距(cm,默认0)
        """
        if ncols > self.ncol or ncols < 1:
            raise ValueError(f"列数 {ncols} 超出范围 (1-{self.ncol})")
        if row < 0 or row >= self.nrow:
            raise IndexError(f"行索引 {row} 超出范围 (0-{self.nrow-1})")
        
        # 更新行配置
        self.row_config[row] = ncols
        
        # 存储额外的配置
        self.row_width_scales[row] = row_width_scale
        self.row_extra_margins[row] = row_extra_margin
        
        # 计算该行可用的总宽度
        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_width_scale - 2 * row_extra_margin
        
        # 计算该行内的列间距总和
        row_space_width = sum(self.space_width[:ncols-1]) if ncols > 1 else 0
        
        # 计算该行每个子图的宽度
        if ncols > 0:
            self.row_subplot_widths[row] = (
                row_available_width - row_space_width
            ) / ncols
        else:
            self.row_subplot_widths[row] = 0
        
        # 重新计算该行高度
        self.row_heights[row] = self.row_subplot_widths[row] * self.subplot_ratio[row]
        
        # 重新计算总高度
        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)
        
    def _configure_matplotlib(self):
        """配置 matplotlib 的全局样式"""
        font = {
            'family': 'serif',
            'weight': 'normal',
            'size': self.ftsize
        }
        plt.rc('text', usetex=True)
        plt.rc('text.latex', 
              preamble=r'\usepackage{amsmath}\usepackage{bm}')
        plt.rc('font', **font)
        plt.rc('xtick', direction='in')
        plt.rc('ytick', direction='in')

    def __cm2inch(self, *tupl):
        """厘米转英寸"""
        inch = 2.54
        return tuple(i/inch for i in tupl)

    def get_fig(self, **kwargs):
        """创建 Figure 对象"""
        figsize = self.__cm2inch(self.plot_width, self.plot_height)
        return plt.figure(figsize=figsize, facecolor='w', **kwargs)

    def get_axes(self, fig, i=0, j=0, **kwargs):
        """在指定位置创建子图"""
        # 获取该行实际子图数量
        ncols_this_row = self.row_config[i]
        
        if j >= ncols_this_row:
            return None
        
        # 计算该行以上所有行的高度和（包括间距）
        height_above = self.margin_top
        for k in range(i):
            height_above += self.row_heights[k]
            if k < len(self.space_height):
                height_above += self.space_height[k]
        
        # 获取该行的额外配置
        row_scale = self.row_width_scales.get(i, 1.0)
        extra_margin = self.row_extra_margins.get(i, 0.0)
        
        # 计算该行可用的总宽度
        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_scale - 2 * extra_margin
        
        # 计算该行两侧的空白（居中放置）
        total_extra_space = content_width - row_available_width
        margin_width = self.margin_left + total_extra_space / 2 + extra_margin
        
        # 添加列偏移（考虑可变列间距）
        for col_idx in range(j):
            margin_width += self.row_subplot_widths[i]
            if col_idx < len(self.space_width):
                margin_width += self.space_width[col_idx]
        
        # 获取当前行高度
        current_row_height = self.row_heights[i]
        
        # 计算相对位置
        y_start = 1.0 - (height_above + current_row_height) / self.plot_height
        
        rect = (
            margin_width / self.plot_width,
            y_start,
            self.row_subplot_widths[i] / self.plot_width,
            current_row_height / self.plot_height
        )
        
        return fig.add_axes(rect, **kwargs)
    
    def get_simple(self):
        """快速获取单子图画布"""
        fig = self.get_fig()
        ax = self.get_axes(fig)
        return fig, ax
    
    def get_multi(self):
        """获取所有子图的画布和坐标轴数组"""
        fig = self.get_fig()
        axs = []
        for i in range(self.nrow):
            row_axs = []
            for j in range(self.row_config[i]):
                ax = self.get_axes(fig, i, j)
                row_axs.append(ax)
            axs.append(row_axs)
        return fig, axs
    
class PlotConfig3:
    """Matplotlib 绘图配置引擎 3.0版本(相比2.0新增功能,完全优于2.0)
    
    功能特性：
    - 预置多种学术出版格式
    - 自动计算子图布局
    - 支持每行子图不同高度比例
    - 支持指定行绘制少于列数的子图
    - 支持行间和列间可变间距
    - 支持 LaTeX 数学公式渲染
    - 厘米单位直接输入
    - 灵活的参数覆盖机制
    
    使用示例：
    >>> config = PlotConfig(config_name='manuscript_single', nrow=3, ncol=3)
    >>> config.set_row_config(row=1, ncols=2)  # 第1行只画2个子图
    >>> fig, axs = config.get_multi()
    >>> plt.show()
    """
    
    def __init__(self, config_name='manuscript_single', nrow=1, ncol=1, **kwargs):
        """初始化绘图配置
        
        参数：
        config_name -- 预定义配置名称 (默认: manuscript_single)
        nrow        -- 子图行数 (默认: 1)
        ncol        -- 子图列数 (默认: 1)
        **kwargs    -- 可覆盖预定义配置的任意参数
        """
        
        # 获取基础配置
        config = CONFIGURATIONS.get(config_name, {}).copy()
        
        # 用户自定义参数覆盖
        config.update(kwargs)
        
        # 配置参数绑定到实例
        self.plot_width = config['plot_width']      # 总画布宽度 (cm)
        self.margin_left = config['margin_left']    # 左侧边距 (cm)
        self.margin_right = config['margin_right']  # 右侧边距 (cm)
        self.margin_bottom = config['margin_bottom'] # 底部边距 (cm)
        self.margin_top = config['margin_top']      # 顶部边距 (cm)
        self.ftsize = config['ftsize']              # 基础字号
        
        # 处理高度比例参数
        if isinstance(config['subplot_ratio'], (int, float)):
            self.subplot_ratio = [config['subplot_ratio']] * nrow
        else:
            self.subplot_ratio = config['subplot_ratio']
        
        # 处理行间垂直间距
        if isinstance(config['space_height'], (int, float)):
            self.space_height = [config['space_height']] * (nrow-1) if nrow > 1 else []
        else:
            if len(config['space_height']) != max(0, nrow-1):
                raise ValueError(f"space_height 需要 {max(0, nrow-1)} 个值，但得到 {len(config['space_height'])}")
            self.space_height = config['space_height']
        self.total_space_height = sum(self.space_height)
        
        # 处理列间水平间距 - 初始化为全局配置
        if isinstance(config['space_width'], (int, float)):
            global_space_width = [config['space_width']] * (ncol-1) if ncol > 1 else []
        else:
            if len(config['space_width']) != max(0, ncol-1):
                raise ValueError(f"space_width 需要 {max(0, ncol-1)} 个值，但得到 {len(config['space_width'])}")
            global_space_width = config['space_width']
        
        # 为每行创建独立的列间距配置（初始使用全局配置）
        self.row_space_widths = {}
        for i in range(nrow):
            self.row_space_widths[i] = global_space_width.copy()
        
        self.total_space_width = sum(global_space_width)
        
        # 布局参数
        self.nrow = nrow
        self.ncol = ncol
        
        # 计算基本子图宽度（所有列等宽时的参考宽度）
        self.base_subplot_width = (self.plot_width 
                                 - self.margin_left - self.margin_right
                                 - self.total_space_width
                                 ) / ncol
        
        # 计算每行高度（基于基本宽度和比例）
        self.row_heights = [
            self.base_subplot_width * ratio 
            for ratio in self.subplot_ratio
        ]
        
        # 计算总画布高度
        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)

        # 每行配置
        self.row_config = {i: ncol for i in range(nrow)}
        self.row_subplot_widths = [self.base_subplot_width] * nrow
        
        # set_row_config 单独针对指定行进行调整
        self.row_width_scales = {}
        self.row_extra_margins = {}
        
        # 配置 matplotlib 全局参数
        self._configure_matplotlib()
    
    def set_row_config(self, row, ncols, row_width_scale=1.0, row_extra_margin=0.0, row_space_width=None):
        """设置指定行的子图配置
        参数：
        row   -- 行索引(0开始)
        ncols -- 该行实际绘制的子图数量(必须 ≤ ncol)
        row_width_scale -- 该行总宽度相对于整个画布内容区域的比例(默认1.0)
        row_extra_margin -- 该行额外的左右边距(cm,默认0)
        row_space_width -- 可选，该行使用的列间距(cm)，可以是标量（所有间距相同）或列表（长度必须为ncols-1）
        """
        if ncols > self.ncol or ncols < 1:
            raise ValueError(f"列数 {ncols} 超出范围 (1-{self.ncol})")
        if row < 0 or row >= self.nrow:
            raise IndexError(f"行索引 {row} 超出范围 (0-{self.nrow-1})")
        
        # 更新行配置
        self.row_config[row] = ncols
        
        # 存储额外的配置
        self.row_width_scales[row] = row_width_scale
        self.row_extra_margins[row] = row_extra_margin
        
        # 处理行特定的列间距
        if row_space_width is not None:
            if ncols <= 1:
                # 单列不需要列间距
                self.row_space_widths[row] = []
            else:
                if isinstance(row_space_width, (int, float)):
                    # 标量值 - 创建等间距列表
                    self.row_space_widths[row] = [row_space_width] * (ncols-1)
                else:
                    if len(row_space_width) != ncols-1:
                        raise ValueError(f"行 {row} 需要 {ncols-1} 个列间距值，但得到 {len(row_space_width)}")
                    self.row_space_widths[row] = row_space_width
        
        # 计算该行内的列间距总和
        row_space_width_sum = sum(self.row_space_widths[row])
        
        # 计算该行可用的总宽度
        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_width_scale - 2 * row_extra_margin
        
        # 计算该行每个子图的宽度
        if ncols > 0:
            self.row_subplot_widths[row] = (
                row_available_width - row_space_width_sum
            ) / ncols
        else:
            self.row_subplot_widths[row] = 0
        
        # 重新计算该行高度
        self.row_heights[row] = self.row_subplot_widths[row] * self.subplot_ratio[row]
        
        # 重新计算总高度
        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)
        
    def _configure_matplotlib(self):
        """配置 matplotlib 的全局样式"""
        font = {
            'family': 'serif',
            'weight': 'normal',
            'size': self.ftsize
        }
        plt.rc('text', usetex=True)
        plt.rc('text.latex', 
              preamble=r'\usepackage{amsmath}\usepackage{bm}')
        plt.rc('font', **font)
        plt.rc('xtick', direction='in')
        plt.rc('ytick', direction='in')

    def __cm2inch(self, *tupl):
        """厘米转英寸"""
        inch = 2.54
        return tuple(i/inch for i in tupl)

    def get_fig(self, **kwargs):
        """创建 Figure 对象"""
        figsize = self.__cm2inch(self.plot_width, self.plot_height)
        return plt.figure(figsize=figsize, facecolor='w', **kwargs)

    def get_axes(self, fig, i=0, j=0, **kwargs):
        """在指定位置创建子图"""
        # 获取该行实际子图数量
        ncols_this_row = self.row_config[i]
        
        if j >= ncols_this_row:
            return None
        
        # 计算该行以上所有行的高度和（包括间距）
        height_above = self.margin_top
        for k in range(i):
            height_above += self.row_heights[k]
            if k < len(self.space_height):
                height_above += self.space_height[k]
        
        # 获取该行的额外配置
        row_scale = self.row_width_scales.get(i, 1.0)
        extra_margin = self.row_extra_margins.get(i, 0.0)
        
        # 计算该行可用的总宽度
        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_scale - 2 * extra_margin
        
        # 计算该行两侧的空白（居中放置）
        total_extra_space = content_width - row_available_width
        margin_width = self.margin_left + total_extra_space / 2 + extra_margin
        
        # 添加列偏移（考虑可变列间距）
        for col_idx in range(j):
            margin_width += self.row_subplot_widths[i]
            # 添加列间距（如果存在）
            if col_idx < len(self.row_space_widths[i]):
                margin_width += self.row_space_widths[i][col_idx]
        
        # 获取当前行高度
        current_row_height = self.row_heights[i]
        
        # 计算相对位置
        y_start = 1.0 - (height_above + current_row_height) / self.plot_height
        
        rect = (
            margin_width / self.plot_width,
            y_start,
            self.row_subplot_widths[i] / self.plot_width,
            current_row_height / self.plot_height
        )
        
        return fig.add_axes(rect, **kwargs)
    
    def get_simple(self):
        """快速获取单子图画布"""
        fig = self.get_fig()
        ax = self.get_axes(fig)
        return fig, ax
    
    def get_multi(self):
        """获取所有子图的画布和坐标轴数组"""
        fig = self.get_fig()
        axs = []
        for i in range(self.nrow):
            row_axs = []
            for j in range(self.row_config[i]):
                ax = self.get_axes(fig, i, j)
                row_axs.append(ax)
            axs.append(row_axs)
        return fig, axs