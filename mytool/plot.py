# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月22日
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.font_manager as font_manager
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.text import OffsetFrom
import os
import numpy as np
import pandas as pd
from cycler import cycler
from mytool import tool


def plot_word_to_word_attention_matrix(attn_matrix, in_words, out_words, ax=None):
    """
    绘制word_to_word注意力权重矩阵，in_words in y-axis, out_words in x_axis
    :param attn_matrix: [i,o]
    :param in_words: [i,]
    :param out_words: [o,]
    :param ax: Axes
    :return:
        ax: Axes
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.matshow(attn_matrix)
    ax.set_xticks(range(len(out_words)))
    ax.set_yticks(range(len(in_words)))
    ax.set_xticklabels(out_words)
    ax.set_yticklabels(in_words)
    return ax


def ax_legend(ax, handles=None, labels=None, loc='best', **kwargs):
    """
    ax.legend()
    :param ax: Axes
    :param handles: 1-D plot object
    :param labels: 1-D label names
    :param loc: legend location
    :param kwargs:
        :key bbox_to_anchor: (x,y,w,h) complete with loc parameter,
                and loc defines which corner to locate
        :key bbox_transform: default Axes transform
        :key ncols: default 1
        :key fontsize: int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    :return:
    """
    ckw = {
        'handles':handles,
        'labels':labels,
        'loc':loc
    }
    ckw = tool.remove_dict_None_value(ckw)
    ckw.update(kwargs)
    ax.legend(**ckw)
    return ax
    # if handles is None or labels is None:
    #     ax.legend(loc=loc,**kwargs)
    # else:
    #     ax.legend(handles,labels,loc=loc,**kwargs)
    # return ax


def get_transform_tuple(x_trans, y_trans):
    """
    get blended transformations
    :param x_trans: in [ax.transData, ax.transAxes, subfigure.transSubfigure,
                        fig.transFigure, fig.dpi_scale_trans]
    :param y_trans: field is the same as x_trans
    :return:
        trans: blended transformations
    """
    trans = transforms.blended_transform_factory(x_trans, y_trans)
    return trans


def add_scaledtransform_to_origintransform(origin_transform,
                                               dx, dy, scale_transform):
    """
    first plot object with origin_transform,
    second move (dx,dy) in scale_transform.
    :param origin_transform: in [ax.transData, ax.transAxes, subfigure.transSubfigure,
                        fig.transFigure, fig.dpi_scale_trans]
    :param dx: numeric
    :param dy: numeric
    :param scale_transform: field is the same as prigin_transform
    :return:
        trans
    """
    trans = origin_transform + transforms.ScaledTranslation(dx, dy, scale_transform)
    return trans


def get_shadow_transform(origin_transform, fig, dxy=2):
    """
    another completion is transforms.offset_copy(origin_transform,fig,dx,dy,units='inches')
    :param origin_transform: in [ax.transData, ax.transAxes, subfigure.transSubfigure,
                        fig.transFigure, fig.dpi_scale_trans]
    :param fig: figure
    :param dxy: numeric units='points'
    :return:
        trans
    """
    trans = add_scaledtransform_to_origintransform(origin_transform,
                                                   dxy/72., -dxy/72.,
                                                   fig.dpi_scale_trans)
    return trans


def get_common_colormaps():
    """
    sequential: 顺序 binary
    diverging: 两边深 coolwarm bwr seismic
    cyclic: 中间深 twilight
    qualitative: 多颜色 Accent Dark2
    miscellaneous: 杂色 rainbow
    :return:
        colormaps: Dict[str,List[colormap]]
    """
    colormaps = {
        's': plt.cm.binary,
        'd': plt.cm.coolwarm,
        'c': plt.cm.twilight,
        'q': plt.cm.Accent,
        'm': plt.cm.rainbow,
    }
    return colormaps


def get_font_dict(family='sans-serif',
                  style='normal',
                  variant='normal',
                  weight='normal',
                  stretch='normal',
                  size=10.0,
                  math_fontfamily='dejavusans',
                  fname=None,
                  return_font_properties=False):
    """

    :param family: 'sans-serif', 'serif', 'cursive', 'fantasy', or 'monospace'.
        the full list of available fonts is in matplotlib.font_manager.get_font_names()
        Default: :rc:`font.family`.
    :param style: 'normal', 'italic' or 'oblique'. Default: :rc:`font.style`
    :param variant: 'normal' or 'small-caps'. Default: :rc:`font.variant`
    :param weight: A numeric value in the range 0-1000 or one of
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
        'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
        'extra bold', 'black'. Default: :rc:`font.weight`
    :param stretch: A numeric value in the range 0-1000 or one of
        'ultra-condensed', 'extra-condensed', 'condensed',
        'semi-condensed', 'normal', 'semi-expanded', 'expanded',
        'extra-expanded' or 'ultra-expanded'. Default: :rc:`font.stretch`
    :param size: Either a relative value of 'xx-small', 'x-small',
        'small', 'medium', 'large', 'x-large', 'xx-large' or an
        absolute font size, e.g., 10. Default: :rc:`font.size`
    :param math_fontfamily: The family of fonts used to render math text.
        Supported values are: 'dejavusans', 'dejavuserif', 'cm',
        'stix', 'stixsans' and 'custom'. Default: :rc:`mathtext.fontset`
    :param fname: custom font file path name. if fname is not None, return_font_properties must be True
    :param return_font_properties: whether to return font_manager.FontProperties
    :return:
        font_dict(funcarg: fontdict) or font_manager.FontProperties(funcarg: fontproperties)
    """
    font_dict = {'family':family,
                 'style':style,
                 'variant':variant,
                 'weight':weight,
                 'stretch':stretch,
                 'size':size,
                 'math_fontfamily':math_fontfamily}
    if fname is not None:
        if os.path.exists(fname):
            return_font_properties = True
        else:
            raise Exception('font file can not be found!')
    if return_font_properties:
        return font_manager.FontProperties(family, style, variant, weight,
                                           stretch, size, fname=fname, math_fontfamily=math_fontfamily)
    return font_dict



def set_Chinese_font_environment():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_bbox_dict(boxstyle='round,pad=0.3',
                  fc='lightblue',
                  ec='black',
                  ls='-',
                  lw=None):
    """
    bbox config dict for bbox of plt.text or plt.annotate.
    bbox definition is in mpl.patches.FancyBboxPatch.
    :param boxstyle: circle, ellipse, [pad=0.3]
        larrow, rarrow, darrow, square [pad=0.3]
        round, round4, [pad=0.3,rounding_size=None]
        sawtooth, roundtooth, [pad=0.3,tooth_size=None]
        definition is in mpl.patches.BoxStyle definition
    :param fc: facecolor
    :param ec: edgecolor
    :param ls: linestyle '-' '--' '-'. ':'
    :param lw: linewidth float
    :return:
        bbox_dict
    """
    bbox_dict = {'boxstyle':boxstyle,
                'fc':fc,
                'ec':ec,
                'ls':ls,
                'lw':lw}
    return bbox_dict


def get_arrowprops_dict(arrowstyle='fancy',
                        connectionstyle='arc3,rad=0.3',
                        relpos=(0.,0.),
                        fc='lightblue',
                        ec='black',
                        ls='-',
                        lw=None
                        ):
    """
    arrowprops config dict for arrowprops of plt.annotate.
    definition is in mpl.patches.FancyArrowPatch.
    :param arrowstyle: '-' '->' '<-' '<->'
        '<|-' '-|>' '<|-|>'
        ']-' '-[' ']-[' '|-|'
        ']->' '<-['
        'simple' 'fancy' 'wedge'
        definition is in mpl.patches.ArrowStyle
        if 'simple' 'fancy' 'wedge' connectionstyle must be in [arc3,angle3]
    :param connectionstyle:
        'arc3': rad=0.0
        'arc': angleA=0, angleB=0, armA=None, armB=None, rad=0.0 起始结束点交点
        'angle': angleA=90, angleB=0, rad=0.0 起始结束点交点
        'angle3': angleA=90, angleB=0 起始结束点交点
        'bar': armA=0.0, armB=0.0, fraction=0.3, angle=None
        definition is in mpl.patches.ConnectionStyle
    :param relpos: the starting point is set to the center of the text extent.
        (0, 0) means lower-left corner and (1, 1) means top-right.
        起始点位置，左下角(0,0)，右上角(1,1)
    :param fc: facecolor
    :param ec: edgecolor
    :param ls: linestyle '-' '--' '-'. ':'
    :param lw: linewidth float
    :return:
        arrowprops_dict
    """
    arrowprops_dict = {
        'arrowstyle':arrowstyle,
        'connectionstyle':connectionstyle,
        'relpos':relpos,
        'fc':fc,
        'ec':ec,
        'ls':ls,
        'lw':lw
    }
    return arrowprops_dict


def ax_tick_params(ax,
                   axis='both',
                   which='major',
                   direction='out',
                   length=None,
                   width=None,
                   pad=None,
                   labelsize=None,
                   labelrotation=None,
                   grid_color=None,
                   grid_alpha=None,
                   grid_linewidth=None,
                   grid_linestyle=None,
                   **kwargs):
    """
    export the args of ax.tick_params
    :param ax: Axes
    :param axis: 'x', 'y', 'both'
    :param which: 'major', 'minor', 'both'
    :param direction: 'in', 'out', 'inout'
    :param length: float. Tick length in points.
    :param width: float. Tick width in points.
    :param pad: float. Distance in points between tick and label.
    :param labelsize: float or str. Tick label font size in points or as a string (e.g., 'large').
    :param labelrotation: float. Tick label rotation
    :param grid_color: Gridline color.
    :param grid_alpha: float. Transparency of gridlines: 0 (transparent) to 1 (opaque).
    :param grid_linewidth: float. Width of gridlines in points.
    :param grid_linestyle: str. Any valid Line2D line style spec.
    :param kwargs:
    :return:
        ax: Axes
    """
    ckw = dict(axis=axis,
               which=which,
               direction=direction,
               length=length,
               width=width,
               pad=pad,
               labelsize=labelsize,
               labelrotation=labelrotation,
               grid_color=grid_color,
               grid_alpha=grid_alpha,
               grid_linewidth=grid_linewidth,
               grid_linestyle=grid_linestyle)
    ckw = tool.remove_dict_None_value(ckw)
    ckw.update(kwargs)
    ax.tick_params(**ckw)
    return ax


def get_offset_coords_from_text_in_bbox(text_obj, ref_coord, unit='points'):
    """
    apply annotate when coords relative to another text_obj
    :param text_obj: annotate or text obj
    :param ref_coord: (0, 0) means lower-left corner and (1, 1) means top-right.
    :param unit: 'points, 'pixels'
    :return:
        offset_from
    """
    offset_from = OffsetFrom(text_obj.get_bbox_patch(),ref_coord,unit=unit)
    return offset_from


def get_marker_str_list():
    """
    :return:
        marker_list: List[str]
    """
    return mpl.markers.MarkerStyle('*').markers.keys()


def get_filled_marker_str_list():
    """
    :return:
        filled_marker_list: List[str]
    """
    return mpl.markers.MarkerStyle('*').filled_markers


def get_default_color_list():
    """
    axes.prop_cycle default color
    :return:
        color_list: List[str]
    """
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_lines_with_compare_data(ax, x, y_2D_np=None, labels=None,
                                 xlabel=None, ylabel=None, title=None, **kwargs):
    """
    When comparing the experimental data of different algorithms that have a common x-axis,
    it is simple.
    :param ax: Axes
    :param x:
        1-D [xlen,] xaxis data
    or  2-D [xlen,nline+1] pd.DataFrame the first column is x, and others are y_2D_np
    :param y_2D_np: 2-D [xlen, nline] yaxis data
    :param labels: 1-D [nline,] data label
    :param xlabel: str
    :param ylabel: str
    :param title: str
    :param kwargs: for plot method
    :return:
        ax: Axes
    """
    if isinstance(x,pd.DataFrame):
        y_2D_np = x.iloc[:, 1:].to_numpy()
        labels = x.columns[1:]
        x = x.iloc[:,0].to_list()

    assert len(x) == y_2D_np.shape[0]
    nline = y_2D_np.shape[1]
    assert nline == len(labels)

    #set style
    if nline > 10: #默认nline最大为20
        marker = get_filled_marker_str_list()[:10]
        color = get_default_color_list()[:10]
        ax.set_prop_cycle(cycler(marker=marker, color=color) * cycler(fillstyle=['full','none']))
    else:
        marker = get_filled_marker_str_list()[:nline]
        color = get_default_color_list()[:nline]
        ax.set_prop_cycle(cycler(marker=marker, color=color))

    #plot
    ax.plot(x, y_2D_np, label=labels, ls='--', lw=3, markersize=8, **kwargs)
    ax.grid(ls=':', lw=2, color='0.8')

    #label
    if xlabel is not None: ax.set_xlabel(xlabel,fontdict=get_font_dict(size=15))
    if ylabel is not None: ax.set_ylabel(ylabel,fontdict=get_font_dict(size=15))
    if title is not None: ax.set_title(title,fontdict=get_font_dict(size=20))

    #ticker
    ax = ax_tick_params(ax, length=4, width=3)

    #set spines
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set(lw=2)
    return ax


def plot_bars_with_compare_data(ax: plt.Axes, x, y_2D_np=None, labels=None,
                                xlabel=None, ylabel=None, title=None,
                                tick_span=None, group_gap_percent=0.2, bar_gap_percent=0, **kwargs):
    """
    When comparing the experimental data of different algorithms that have a common x-axis,
    Args:
        ax: Axes
        x: 1-D [xlen,] xaxis data
            or  2-D [xlen,nline+1] pd.DataFrame the first column is x, and others are y_2D_np
        y_2D_np: 2-D [xlen, nline] yaxis data
        labels: 1-D [nline,] data label
        xlabel: str
        ylabel: str
        title: str
        tick_span: int/float
        group_gap_percent: float [0,1]
        bar_gap_percent: float [0,1]
        **kwargs: for bar method
    Returns:
        ax: Axes
    """
    if isinstance(x,pd.DataFrame):
        y_2D_np = x.iloc[:, 1:].to_numpy()
        labels = x.columns[1:]
        x = x.iloc[:,0].to_list()

    assert len(x) == y_2D_np.shape[0]
    nline = y_2D_np.shape[1]
    assert nline == len(labels)

    # get tick_span
    if tick_span is None:
        if isinstance(x[0], str):
            tick_span = 1
        elif isinstance(x[0], int):
            tick_span = int(np.mean(np.diff(x)))
        elif isinstance(x[0], float):
            tick_span = np.mean(np.diff(x))
        else:
            raise Exception('can not compute tick_span. x must be int/float/str')
    # compute bar width and bar_span and base_x
    ticks = np.array(x) #np.arange(len(x))
    group_num = y_2D_np.shape[1]
    # get group_gap
    group_gap = tick_span * group_gap_percent
    group_width = tick_span - group_gap
    bar_span = group_width / group_num
    # get bar_gap
    bar_gap = bar_span * bar_gap_percent
    bar_width = bar_span - bar_gap
    base_x = ticks - (group_width - bar_span) / 2
    for ind, y in enumerate(y_2D_np.T):
        ax.bar(base_x + ind * bar_span, y, width=bar_width, label=labels[ind], alpha=0.8, **kwargs)
    # grid
    ax.grid(ls=':', lw=2, color='0.8')

    # label
    if xlabel is not None: ax.set_xlabel(xlabel,fontdict=get_font_dict(size=15))
    if ylabel is not None: ax.set_ylabel(ylabel,fontdict=get_font_dict(size=15))
    if title is not None: ax.set_title(title,fontdict=get_font_dict(size=20))

    # ticker
    ax.set_xticks(x)
    ax = ax_tick_params(ax, length=4, width=3)

    #set spines
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set(lw=2)
    return ax


def plot_LossMetricTimeLr_with_df(df,
                                  loss_check='loss',
                                  metric_check='metric',
                                  time_check='time',
                                  lr_check='lr'):
    """
    plot data which contains loss/metric/time/lr for pd.DataFrame
    :param df: pd.DataFrame
    :param loss_check: loss_key will be plotted in ax1
    :param metric_check: metric_key will be plotted in ax2
    :param time_check: time_key (2-column) will be plotted in ax3
    :param lr_check: lr_key (1-column) will be plotted in ax4
    :return:
        fig
    """
    column_names = df.columns
    loss_data = df.iloc[:,column_names.str.contains(loss_check)]
    metric_data = df.iloc[:,column_names.str.contains(metric_check)]
    time_data = df.iloc[:,column_names.str.contains(time_check)]
    lr_data = df.iloc[:,column_names.str.contains(lr_check)]

    fig = plt.figure(figsize=(18,20))
    plt.subplots_adjust(hspace=0.35, wspace=0.35)

    #ax1 loss
    ax1_loss = fig.add_subplot(221)
    loss_data.reset_index(inplace=True)
    ax1_loss = plot_lines_with_compare_data(ax1_loss,loss_data,
                                            xlabel='Epoch',ylabel='Loss', title='Loss/Epoch', markevery=10)
    ax1_loss.legend()

    #ax3 metric
    ax2_metric = fig.add_subplot(222)
    metric_data.reset_index(inplace=True)
    ax2_metric = plot_lines_with_compare_data(ax2_metric, metric_data,
                                              xlabel='Epoch',ylabel='Metric', title='Metric/Epoch', markevery=10)
    ax2_metric.legend()

    #ax3 time val_time
    ax3_time = fig.add_subplot(223)
    time = time_data['time'].to_frame()
    time.reset_index(inplace=True)
    ax3_time = plot_lines_with_compare_data(ax3_time, time,
                                            xlabel='Epoch',ylabel='Time(Train)', title='Time/Epoch', c='r', markevery=10)
    ax3_time.annotate('Mean:'+str(time.mean(0)[1]),
                      xy=(0,time.mean(0)[1]), xycoords=(ax3_time.transAxes,ax3_time.transData),
                      xytext=(-70,60), textcoords='offset points',
                      ha='left',va='top',
                      arrowprops=get_arrowprops_dict(fc='r'),
                      fontproperties = get_font_dict(size=10, weight='bold', return_font_properties=True))
    ax3_time.legend(['time(left)'],loc='upper left')
    val_time = time_data['val_time'].to_frame()
    val_time.reset_index(inplace=True)
    ax3_2_time = ax3_time.twinx()
    ax3_2_time = plot_lines_with_compare_data(ax3_2_time, val_time,
                                              xlabel='Epoch',ylabel='Val_Time(Validation)',
                                              c='b', markevery=10)
    ax3_2_time.annotate('Mean:' + str(val_time.mean(0)[1]),
                      xy=(1, val_time.mean(0)[1]), xycoords=(ax3_2_time.transAxes, ax3_2_time.transData),
                      xytext=(70, -60), textcoords='offset points',
                      ha='right', va='bottom',
                      arrowprops=get_arrowprops_dict(fc='b'),
                      fontproperties=get_font_dict(size=10, weight='bold', return_font_properties=True))
    ax3_2_time.legend(['val_time(right)'],loc='upper right')

    #ax4 lr
    ax4_lr = fig.add_subplot(224)
    lr_data.reset_index(inplace=True)
    ax4_lr = plot_lines_with_compare_data(ax4_lr, lr_data,
                                          xlabel='Epoch',ylabel='LearningRate', title='LearningRate/Epoch', markevery=10)
    ax4_lr.legend()

    return fig



if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 5),layout='constrained')
    ###
    ax1 = fig.add_subplot(131)
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y = np.sin(x)
    line, = ax1.plot(x,y,label=r'$\sin(x)$')
    line_shadow, = ax1.plot(x,y,transform=get_shadow_transform(ax1.transData,fig,5),
                          zorder = 0.5*line.get_zorder(),color='0.8')
    ax1.set_title('Title',fontproperties=get_font_dict(weight=0,return_font_properties=True))
    text1 = ax1.annotate('(0,0)',xy=(0,0),xycoords='data',
                xytext=(30,30),textcoords='offset points',
                va='center',ha='center',
                bbox=get_bbox_dict(fc='red'),
                arrowprops=get_arrowprops_dict(relpos=(0,0))
                )
    # text1 = plt.text(30,30,'(0,0)',
    #                  transform=mpl.transforms.IdentityTransform()+mpl.transforms.ScaledTranslation(0,0,ax1.transData),
    #                  bbox=get_bbox_dict(),
    #                  ha='center',va='center'
    #                  )
    ax1.annotate("",xy=(0,0),xycoords=ax1.transData,
                 xytext=(0,0),textcoords=get_offset_coords_from_text_in_bbox(text1,(1,0)),
                 ha='center',va='center',
                 arrowprops=get_arrowprops_dict('<->','arc3,rad=-0.3'))
    ax1.grid()
    ax1.legend()

    ###
    ax2 = fig.add_subplot(132)
    xdates = tool.get_date_range_start_end_pd('2023/10/1','2023/10/24','1D')
    yvalue = xdates.dayofweek
    ax2.plot(xdates,yvalue,label='date_test')
    ax2.legend()
    ax_tick_params(ax2,labelrotation=45)

    ###
    ax3 = fig.add_subplot(133)
    nline = 3
    x = np.arange(6)
    y1ndarray = np.cumsum(np.random.rand(x.shape[0],nline),axis=1)
    label = [str(i+1) for i in range(nline)]
    # plot_lines_with_compare_data(ax3,x,y1ndarray,label,'epoch','loss', 'sources')
    df = pd.DataFrame(np.concatenate([x[:,np.newaxis],y1ndarray],axis=1),columns=['x']+label)
    plot_bars_with_compare_data(ax3,df)
    ax3.legend()

    plt.show()