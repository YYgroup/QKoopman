import matplotlib.pyplot as plt

CONFIGURATIONS = {
    "manuscript_single": {
        "plot_width": 9.0,        #  (cm)
        "margin_left": 1.4,       #  (cm)
        "margin_right": 0.3,      #  (cm)
        "margin_bottom": 1.2,     #  (cm)
        "margin_top": 0.3,        #  (cm)
        "space_width": 1.5,       #  (cm)
        "space_height": 1.0,      #  (cm)
        "subplot_ratio": 0.9,     #  (height/width)
        "ftsize": 11             # 
    },
}


class PlotConfig2:
    def __init__(self, config_name='manuscript_single', nrow=1, ncol=1, **kwargs):
        
        config = CONFIGURATIONS.get(config_name, {}).copy()
        
        config.update(kwargs)
        
        self.plot_width = config['plot_width']      #  (cm)
        self.margin_left = config['margin_left']    #  (cm)
        self.margin_right = config['margin_right']  #  (cm)
        self.margin_bottom = config['margin_bottom'] #  (cm)
        self.margin_top = config['margin_top']      #  (cm)
        self.ftsize = config['ftsize']              # 
        

        if isinstance(config['subplot_ratio'], (int, float)):
            self.subplot_ratio = [config['subplot_ratio']] * nrow
        else:
            self.subplot_ratio = config['subplot_ratio']
        

        if isinstance(config['space_height'], (int, float)):
            self.space_height = [config['space_height']] * (nrow-1) if nrow > 1 else []
        else:
            if len(config['space_height']) != max(0, nrow-1):
                raise ValueError(f"space_height  {max(0, nrow-1)} ， {len(config['space_height'])}")
            self.space_height = config['space_height']
        self.total_space_height = sum(self.space_height)
        

        if isinstance(config['space_width'], (int, float)):
            self.space_width = [config['space_width']] * (ncol-1) if ncol > 1 else []
        else:
            if len(config['space_width']) != max(0, ncol-1):
                raise ValueError(f"space_width  {max(0, ncol-1)} ， {len(config['space_width'])}")
            self.space_width = config['space_width']
        self.total_space_width = sum(self.space_width)
        

        self.nrow = nrow
        self.ncol = ncol
        

        self.base_subplot_width = (self.plot_width 
                                 - self.margin_left - self.margin_right
                                 - self.total_space_width
                                 ) / ncol
        

        self.row_heights = [
            self.base_subplot_width * ratio 
            for ratio in self.subplot_ratio
        ]
        

        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)


        self.row_config = {i: ncol for i in range(nrow)}
        self.row_subplot_widths = [self.base_subplot_width] * nrow
        

        self.row_width_scales = getattr(self, 'row_width_scales', {})
        self.row_extra_margins = getattr(self, 'row_extra_margins', {})
        

        self._configure_matplotlib()
    
    def set_row_config(self, row, ncols, row_width_scale=1.0, row_extra_margin=0.0):
        if ncols > self.ncol or ncols < 1:
            raise ValueError(f" {ncols}  (1-{self.ncol})")
        if row < 0 or row >= self.nrow:
            raise IndexError(f" {row}  (0-{self.nrow-1})")
        

        self.row_config[row] = ncols
        

        self.row_width_scales[row] = row_width_scale
        self.row_extra_margins[row] = row_extra_margin
        

        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_width_scale - 2 * row_extra_margin
        

        row_space_width = sum(self.space_width[:ncols-1]) if ncols > 1 else 0
        

        if ncols > 0:
            self.row_subplot_widths[row] = (
                row_available_width - row_space_width
            ) / ncols
        else:
            self.row_subplot_widths[row] = 0
        

        self.row_heights[row] = self.row_subplot_widths[row] * self.subplot_ratio[row]
        

        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)
        
    def _configure_matplotlib(self):
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
        inch = 2.54
        return tuple(i/inch for i in tupl)

    def get_fig(self, **kwargs):
        figsize = self.__cm2inch(self.plot_width, self.plot_height)
        return plt.figure(figsize=figsize, facecolor='w', **kwargs)

    def get_axes(self, fig, i=0, j=0, **kwargs):

        ncols_this_row = self.row_config[i]
        
        if j >= ncols_this_row:
            return None
        

        height_above = self.margin_top
        for k in range(i):
            height_above += self.row_heights[k]
            if k < len(self.space_height):
                height_above += self.space_height[k]
        

        row_scale = self.row_width_scales.get(i, 1.0)
        extra_margin = self.row_extra_margins.get(i, 0.0)
        

        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_scale - 2 * extra_margin
        

        total_extra_space = content_width - row_available_width
        margin_width = self.margin_left + total_extra_space / 2 + extra_margin
        

        for col_idx in range(j):
            margin_width += self.row_subplot_widths[i]
            if col_idx < len(self.space_width):
                margin_width += self.space_width[col_idx]
        

        current_row_height = self.row_heights[i]
        

        y_start = 1.0 - (height_above + current_row_height) / self.plot_height
        
        rect = (
            margin_width / self.plot_width,
            y_start,
            self.row_subplot_widths[i] / self.plot_width,
            current_row_height / self.plot_height
        )
        
        return fig.add_axes(rect, **kwargs)
    
    def get_simple(self):
        fig = self.get_fig()
        ax = self.get_axes(fig)
        return fig, ax
    
    def get_multi(self):
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
    
    def __init__(self, config_name='manuscript_single', nrow=1, ncol=1, **kwargs):


        config = CONFIGURATIONS.get(config_name, {}).copy()
        

        config.update(kwargs)
        

        self.plot_width = config['plot_width']      #  (cm)
        self.margin_left = config['margin_left']    #  (cm)
        self.margin_right = config['margin_right']  #  (cm)
        self.margin_bottom = config['margin_bottom'] #  (cm)
        self.margin_top = config['margin_top']      #  (cm)
        self.ftsize = config['ftsize']              # 
        

        if isinstance(config['subplot_ratio'], (int, float)):
            self.subplot_ratio = [config['subplot_ratio']] * nrow
        else:
            self.subplot_ratio = config['subplot_ratio']
        

        if isinstance(config['space_height'], (int, float)):
            self.space_height = [config['space_height']] * (nrow-1) if nrow > 1 else []
        else:
            if len(config['space_height']) != max(0, nrow-1):
                raise ValueError(f"space_height  {max(0, nrow-1)} ， {len(config['space_height'])}")
            self.space_height = config['space_height']
        self.total_space_height = sum(self.space_height)
        

        if isinstance(config['space_width'], (int, float)):
            global_space_width = [config['space_width']] * (ncol-1) if ncol > 1 else []
        else:
            if len(config['space_width']) != max(0, ncol-1):
                raise ValueError(f"space_width  {max(0, ncol-1)} ， {len(config['space_width'])}")
            global_space_width = config['space_width']
        

        self.row_space_widths = {}
        for i in range(nrow):
            self.row_space_widths[i] = global_space_width.copy()
        
        self.total_space_width = sum(global_space_width)
        

        self.nrow = nrow
        self.ncol = ncol
        

        self.base_subplot_width = (self.plot_width 
                                 - self.margin_left - self.margin_right
                                 - self.total_space_width
                                 ) / ncol
        

        self.row_heights = [
            self.base_subplot_width * ratio 
            for ratio in self.subplot_ratio
        ]
        

        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)


        self.row_config = {i: ncol for i in range(nrow)}
        self.row_subplot_widths = [self.base_subplot_width] * nrow
        

        self.row_width_scales = {}
        self.row_extra_margins = {}
        

        self._configure_matplotlib()
    
    def set_row_config(self, row, ncols, row_width_scale=1.0, row_extra_margin=0.0, row_space_width=None):
        if ncols > self.ncol or ncols < 1:
            raise ValueError(f" {ncols}  (1-{self.ncol})")
        if row < 0 or row >= self.nrow:
            raise IndexError(f" {row}  (0-{self.nrow-1})")
        

        self.row_config[row] = ncols
        

        self.row_width_scales[row] = row_width_scale
        self.row_extra_margins[row] = row_extra_margin
        

        if row_space_width is not None:
            if ncols <= 1:

                self.row_space_widths[row] = []
            else:
                if isinstance(row_space_width, (int, float)):

                    self.row_space_widths[row] = [row_space_width] * (ncols-1)
                else:
                    if len(row_space_width) != ncols-1:
                        raise ValueError(f" {row}  {ncols-1} ， {len(row_space_width)}")
                    self.row_space_widths[row] = row_space_width
        

        row_space_width_sum = sum(self.row_space_widths[row])
        

        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_width_scale - 2 * row_extra_margin
        

        if ncols > 0:
            self.row_subplot_widths[row] = (
                row_available_width - row_space_width_sum
            ) / ncols
        else:
            self.row_subplot_widths[row] = 0
        

        self.row_heights[row] = self.row_subplot_widths[row] * self.subplot_ratio[row]
        

        self.plot_height = (self.margin_bottom + self.margin_top
                          + sum(self.row_heights)
                          + self.total_space_height)
        
    def _configure_matplotlib(self):
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
        """"""
        inch = 2.54
        return tuple(i/inch for i in tupl)

    def get_fig(self, **kwargs):
        """ Figure """
        figsize = self.__cm2inch(self.plot_width, self.plot_height)
        return plt.figure(figsize=figsize, facecolor='w', **kwargs)

    def get_axes(self, fig, i=0, j=0, **kwargs):

        ncols_this_row = self.row_config[i]
        
        if j >= ncols_this_row:
            return None
        

        height_above = self.margin_top
        for k in range(i):
            height_above += self.row_heights[k]
            if k < len(self.space_height):
                height_above += self.space_height[k]
        

        row_scale = self.row_width_scales.get(i, 1.0)
        extra_margin = self.row_extra_margins.get(i, 0.0)
        

        content_width = self.plot_width - self.margin_left - self.margin_right
        row_available_width = content_width * row_scale - 2 * extra_margin
        

        total_extra_space = content_width - row_available_width
        margin_width = self.margin_left + total_extra_space / 2 + extra_margin
        

        for col_idx in range(j):
            margin_width += self.row_subplot_widths[i]

            if col_idx < len(self.row_space_widths[i]):
                margin_width += self.row_space_widths[i][col_idx]
        

        current_row_height = self.row_heights[i]
        

        y_start = 1.0 - (height_above + current_row_height) / self.plot_height
        
        rect = (
            margin_width / self.plot_width,
            y_start,
            self.row_subplot_widths[i] / self.plot_width,
            current_row_height / self.plot_height
        )
        
        return fig.add_axes(rect, **kwargs)
    
    def get_simple(self):
        fig = self.get_fig()
        ax = self.get_axes(fig)
        return fig, ax
    
    def get_multi(self):
        fig = self.get_fig()
        axs = []
        for i in range(self.nrow):
            row_axs = []
            for j in range(self.row_config[i]):
                ax = self.get_axes(fig, i, j)
                row_axs.append(ax)
            axs.append(row_axs)
        return fig, axs