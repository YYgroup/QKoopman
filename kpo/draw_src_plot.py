from importlib import reload  # 导入 reload 函数
import draw_src_figconfig
reload (draw_src_figconfig )
import matplotlib.pyplot as plt
import os
import numpy as np
from draw_src_figconfig import PlotConfig2,PlotConfig3
from scipy.stats import norm
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.animation import FuncAnimation

def getv(key, v):
    return v[key,:,:]

def draw_contour(v1,v3,case,ind,error_type='Relative MSE',result='result_5_16'):
    '''
    Args:
        v1 (np.array): (t, h, h) GT
        v3 (np.array): (t, h, h) QKM
    '''
    config = PlotConfig2(nrow=4,ncol=7,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.4,       # 左侧边距 (cm)
                        margin_right= 0.3,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 0.3,        # 顶部边距 (cm)
                        space_width= 0.14,       # 子图水平间距 (cm)
                        space_height= [0.342,0.342,0.7],      # 子图垂直间距 (cm)
                        subplot_ratio= [0.9,0.9,0.9,0.121],     # 子图高宽比 (height/width)
                        ftsize=8,             # 基础字体大小
                    )
    config.set_row_config(row=3,ncols=1)
    fig,axes = config.get_multi()

    # 设置 坐标轴范围 ticks,labels
    extent = [0, 128, 0, 128]  # 设置 xmin, xmax, ymin, ymax
    ticks = [0,64,128]
    if case == 0:
        labels_x = ['0', '0.5', '1']
        labels_y = ['0', '0.5', '1']
    elif case == 1:
        labels_x = ['0', '$2\pi$', '$4\pi$']
        labels_y = ['0', '$2\pi$', '$4\pi$']
    elif case == 2:
        labels_x = ['-1', '0', '1']
        labels_y = ['-1', '0', '1']
    else:
        print("case error: please set case = 0,1,2")

    print(f'v1.min()={v1.min()},v1.max()={v1.max()}')
    if case == 0:
        vmin = -12.
        vmax = 12.
        tau = np.abs(v1[0]).max()
    elif case == 1:
        vmin = -10.
        vmax = 10.
        tau = np.abs(v1[0]).max()
        # print(f'k=61: t={61*0.1/tau:.2f}tau')
        # print(f'k=63: t={63*0.1/tau:.2f}tau')
        # print(f'k=65: t={65*0.1/tau:.2f}tau')
        # print(f'k=67: t={67*0.1/tau:.2f}tau')
        # print(f'k=69: t={69*0.1/tau:.2f}tau')
        # print(f'k=70: t={70*0.1/tau:.2f}tau')
    elif case == 2:
        vmin = 0.1
        vmax = 1.
        tau = 1/0.029

        # print(f'k=61: t={61*0.1/tau:.2f}tau')
        # print(f'k=63: t={63*0.1/tau:.2f}tau')
        # print(f'k=65: t={65*0.1/tau:.2f}tau')
        # print(f'k=67: t={67*0.1/tau:.2f}tau')
        # print(f'k=69: t={69*0.1/tau:.2f}tau')
        # print(f'k=70: t={70*0.1/tau:.2f}tau')

    # 获取时间步
    t_list = [0,10,20,30,40,50,60]

    for k in range(7) : # 7列
        # 绘制 GT 图像
        im = axes[0][k].imshow(
            getv(t_list[k], v1),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )

        axes[0][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[0][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[0][k].set_xticklabels(['','',''])
        axes[0][k].set_yticklabels(['','',''])

        # 绘制 QKM 图像
        axes[1][k].imshow(
            getv(t_list[k], v3),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )
        axes[1][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[1][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[1][k].set_xticklabels(['','',''])
        axes[1][k].set_yticklabels(['','',''])

        # 绘制 Res 图像
        axes[2][k].imshow(
            getv(t_list[k], v3-v1),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )
        axes[2][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[2][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[2][k].set_xticklabels(['','',''])
        axes[2][k].set_yticklabels(['','',''])
        if case==2:
            axes[0][k].set_title(fr"${{t}}^{{*}}={t_list[k]*10/tau:.1f}$",fontsize=config.ftsize,pad=2.0)
        elif case==1:
            axes[0][k].set_title(fr"${{t}}^{{*}}={t_list[k]*0.1/tau:.1f}$",fontsize=config.ftsize,pad=2.0)
        elif case==0:
            axes[0][k].set_title(fr"${{t}}^{{*}}={t_list[k]*0.1/tau:.2f}$",fontsize=config.ftsize,pad=2.0)

    axes[0][0].set_yticklabels(labels_y,fontsize=config.ftsize)
    axes[1][0].set_yticklabels(labels_y,fontsize=config.ftsize)
    axes[2][0].set_yticklabels(labels_y,fontsize=config.ftsize)

    if case==0:
        axes[0][0].set_ylabel('$y$',labelpad=2.,fontsize=config.ftsize)
        axes[1][0].set_ylabel('$y$',labelpad=2.,fontsize=config.ftsize)
        axes[2][0].set_ylabel('$y$',labelpad=2.,fontsize=config.ftsize)
    elif case==1:
        axes[0][0].set_ylabel('$y$',labelpad=2.,fontsize=config.ftsize)
        axes[1][0].set_ylabel('$y$',labelpad=2.,fontsize=config.ftsize)
        axes[2][0].set_ylabel('$y$',labelpad=2.,fontsize=config.ftsize)
    elif case==2:
        axes[0][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)
        axes[1][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)
        axes[2][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)

    for j in range(7):
        axes[2][j].set_xticklabels(labels_x,fontsize=config.ftsize)
        axes[2][j].set_xlabel('$x$',labelpad=0,fontsize=config.ftsize)

    # color bar
    subwidth = axes[0][-1].get_position().x0 - axes[0][-2].get_position().x0 - axes[0][-2].get_position().width
    pos_up = axes[0][-1].get_position()
    pos_down = axes[2][-1].get_position()
    ax_bar = fig.add_axes([pos_down.x0+pos_up.width+subwidth*0.6,
                  pos_down.y0+pos_up.height/4,
                  subwidth*0.5,
                  pos_up.y0-pos_down.y0+pos_up.height/2,])
    cbar = fig.colorbar(im,cax=ax_bar,orientation='vertical')
    cbar.ax.tick_params(
        direction='in',  # 刻度线朝外
        length=2.,         # 刻度线长度（默认是 4，增大后会更凸出）
        width=0.7,          # 刻度线宽度
        colors='black',   # 刻度线颜色
        pad=1.5,          # 刻度标签与刻度线的距离
        top=True,
        bottom=False
    )

    if case == 0:
        bar_label='$\omega$'
        bar_ticks=[-10,-5,0,5,10]
        bar_tls=[f'{x:d}' for x in bar_ticks]
    elif case == 1:
        bar_label='$\omega$'
        bar_ticks=[-8,-4,0,4,8]
        bar_tls=[f'{x:d}' for x in bar_ticks]
    elif case == 2:
        bar_label='$Y_A$'
        bar_ticks=[0.1,0.3,0.5,0.7,0.9]
        bar_tls=[f'{x:.1f}' for x in bar_ticks]

    cbar.set_ticks(bar_ticks)
    cbar.set_ticklabels(bar_tls,fontsize=config.ftsize)


    cbar.set_label('') # 移除默认的 label（如果有）
    # 手动添加 label
    cbar.ax.text(
        0.5,                    # x 位置（0=左，1=右，0.5=居中）
        1.02,                   # y 位置
        bar_label,              # 标签文本
        ha='center',            # 水平居中
        va='bottom',            # 垂直对齐（底部）
        fontsize=config.ftsize,      # 字体大小
        transform=cbar.ax.transAxes  # 使用相对坐标（0-1）
    )

    # 绘图error
    ax = axes[3][0]
    pos = ax.get_position()
    pos_right = axes[2][-1].get_position()
    ax.remove()
    ax = fig.add_axes([axes[2][0].get_position().x0,
                  pos.y0,
                  pos_right.x0+pos_right.width-axes[2][0].get_position().x0,
                  pos.height,])

    # mae = ( np.abs(v3-v1) ).mean(axis=(-2,-1))
    mse = ( (v3-v1)**2 ).mean(axis=(-2,-1))
    # rmse = np.sqrt(mse)
    error_dict={
        # 'MAE': mae,
        # 'MSE': mse,
        # 'RMSE': rmse,
        'Relative MSE': mse / (v1**2).mean(axis=(-2,-1)),
        # 'Relative MAE': mae / ( np.abs(v1) ).mean(axis=(-2,-1)),
    }
    error = error_dict[error_type]

    rmse = np.load(f'RMSE_case={case:d}_{result:s}.npy')
    rmse_low=np.zeros(rmse.shape[1])
    rmse_high=np.zeros(rmse.shape[1])
    for i in range(rmse.shape[1]):
        data = rmse[:,i]
        rmse_low[i] = np.percentile(data, 10)  # 10%分位数
        rmse_high[i] = np.percentile(data, 90)  # 90%分位数

    # ax.fill_between(np.arange(rmse.shape[1]),rmse_low,rmse_high,
    #                 alpha=0.4,edgecolor='none',color='#e8d5d5')
    # ax.plot(error,linestyle='-',color='#c79494',linewidth=1.5)
    ax.fill_between(np.arange(rmse.shape[1]),rmse_low,rmse_high,
                    alpha=0.25,edgecolor='none',color='#A8B5C2')
    ax.plot(error,linestyle='-',color='#A6cced',linewidth=1.5)

    print(fr'[k= 0] RMSE={error[0]*100:.1f}\%')
    print(fr'[k=15] RMSE={error[14]*100:.1f}\%')
    print(fr'[k=30] RMSE={error[29]*100:.1f}\%')
    print(fr'[k=45] RMSE={error[44]*100:.1f}\%')
    print(fr'[k=60] RMSE={error[59]*100:.1f}\%')

    if case==0:
        ax.set_ylim(0, 0.05)
        ax.set_yticks([0,0.02,0.04])
        ax.set_yticklabels([0,2,4],fontsize=config.ftsize)
    elif case==1:
        ax.set_ylim(0, 1.4)
        ax.set_yticks([0,0.6,1.2])
        ax.set_yticklabels([0,60,120],fontsize=config.ftsize)
    elif case==2:
        # ax.set_ylim(0, 0.12)
        # ax.set_yticks([0,0.06,0.12])
        # ax.set_yticklabels([0,6,12],fontsize=config.ftsize)
        ax.set_ylim(0, 0.07)
        ax.set_yticks([0,0.03,0.06])
        ax.set_yticklabels([0,3,6],fontsize=config.ftsize)

    ax.grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    ax.set_axisbelow(True)  # 网格线和坐标轴置于底层
    ax.set_xlim([-1,61])
    xticks=[0,10,20,30,40,50,60]
    ax.set_xticks(xticks)
    if case==2:
        ax.set_xticklabels([fr'{i*10:.0f}' for i in xticks])
    elif case==1:
        ax.set_xticklabels([fr'{i*0.1:.0f}' for i in xticks])
    elif case==0:
        ax.set_xticklabels([fr'{i*0.1:.0f}' for i in xticks])

    ax.tick_params(axis='y', which='major',labelsize=config.ftsize,
            top=False, right=False, length=3, pad=1)
    ax.tick_params(axis='x', which='major',labelsize=config.ftsize,
            top=False, right=False, length=3, pad=3)
    ax.tick_params(which='minor',top=False, right=False, length=1.5)
    ax.set_xlabel(r'$t$',labelpad=1.,fontsize=config.ftsize)

    pos0 = axes[0][0].get_position()
    pos1 = axes[1][0].get_position()
    pos2 = axes[2][0].get_position()
    pos3 = ax.get_position()

    if case==0:
        fig.text(pos0.x0-0.0625,pos0.y0+pos0.height/2, 'GT', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.0625,pos1.y0+pos1.height/2, 'QKM', ha='center', va='center', rotation=90,  fontsize=config.ftsize)
        fig.text(pos0.x0-0.0625,pos2.y0+pos2.height/2, r'$\epsilon$', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.0375,pos3.y0+pos3.height/2, r'$\mathrm{relative}\; L_2\text{-}\epsilon$ (\%)', ha='center', va='center', rotation=90,  fontsize=config.ftsize)

        fig.text(pos0.x0-0.0725,pos0.y0+pos0.height*1.0005, r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.0725,pos1.y0+pos1.height*1.0005, r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.0725,pos2.y0+pos2.height*1.0005, r'(c)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.0725,pos3.y0+pos3.height*1.0015, r'(d)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    elif case==2:
        fig.text(pos0.x0-0.048,pos0.y0+pos0.height/2, 'GT', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.048,pos1.y0+pos1.height/2, 'QKM', ha='center', va='center', rotation=90,  fontsize=config.ftsize)
        fig.text(pos0.x0-0.048,pos2.y0+pos2.height/2, r'$\epsilon$', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.028,pos3.y0+pos3.height/2,r'$\mathrm{relative}\; L_2\text{-}\epsilon$ (\%)', ha='center', va='center', rotation=90,  fontsize=config.ftsize)

        fig.text(pos0.x0-0.058,pos0.y0+pos0.height*1.0005, r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.058,pos1.y0+pos1.height*1.0005, r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.058,pos2.y0+pos2.height*1.0005, r'(c)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.058,pos3.y0+pos3.height*1.0015, r'(d)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    elif case==1:
        fig.text(pos0.x0-0.048,pos0.y0+pos0.height/2, 'GT', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.048,pos1.y0+pos1.height/2, 'QKM', ha='center', va='center', rotation=90,  fontsize=config.ftsize)
        fig.text(pos0.x0-0.048,pos2.y0+pos2.height/2, r'$\epsilon$', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.043,pos3.y0+pos3.height/2,r'$\mathrm{relative}\; L_2\text{-}\epsilon$ (\%)', ha='center', va='center', rotation=90,  fontsize=config.ftsize)

        fig.text(pos0.x0-0.068,pos0.y0+pos0.height*1.0005, r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.068,pos1.y0+pos1.height*1.0005, r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.068,pos2.y0+pos2.height*1.0005, r'(c)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
        fig.text(pos0.x0-0.068,pos3.y0+pos3.height*1.0015, r'(d)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    for i in range(3):
        for j in range(7):
            axes[i][j].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=2., width=0.7,pad=1)
            axes[i][j].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=2., width=0.7,pad=3)
            axes[i][j].tick_params(which='minor',top=False, right=False, length=1.,width=0.7,)

    # fig.savefig(fr'case={case:d}_ind={ind:d}_contour.png', transparent=True, orientation='portrait',
                # bbox_inches='tight',dpi=600)

    fig.savefig(fr'case={case:d}_ind={ind:d}_contour.pdf', transparent=True, orientation='portrait',
                bbox_inches='tight')

# ========================================================

def draw_statistic_case1(dictS1p,dictS3p,p1_list,p2_list,
                ux1,uy1,ux3,uy3,bins,
                k1,e1,k3,e3,
                case,ind):
    '''
        p1_list \subset p2_list
        q1_list \subset q2_list
        ux (np.array): (t,h,h)
        uy (np.array): (t,h,h)
        k (np.array): (k,)
        e (np.array): (t,k)
        case,ind (int)
    '''
    config = PlotConfig2(nrow=2,ncol=2,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.4,       # 左侧边距 (cm)
                        margin_right= 0.3,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 0.3,        # 顶部边距 (cm)
                        space_width= 1.5,       # 子图水平间距 (cm)
                        space_height= 1.5,      # 子图垂直间距 (cm)
                        subplot_ratio= 0.9,     # 子图高宽比 (height/width)
                        ftsize=8,             # 基础字体大小
                    )
    fig,axes = config.get_multi()

    dictS1p_t,dictS3p_t={},{}
    for p in p2_list:
        dictS1p_t[f's{p:d}'] = dictS1p[f's{p:d}'][10] # k=10
        dictS3p_t[f's{p:d}'] = dictS3p[f's{p:d}'][10] # k=10
    # 速度结构函数
    id1,id2 = 0,18

    x = dictS1p['r'][id1:id2]
    axes[1][0].plot(dictS1p['r'][0:5], 0.012*dictS1p['r'][0:5]**(2), 'k--', linewidth=1.0, label="_nolegend_")
    colors=['#d16d5b','#b7282e','#e64532',]
    markers=['o','^','s']
    linestyles=['-','--','-.']
    markerfacecolors=['#DDE9F5','#AFD9FD','#bbcfe8']
    markeredgecolors=['#7AA7D3','#92bbd7','#659bca']
    for i,p in enumerate(p1_list):
        axes[1][0].plot(x, dictS1p_t[f's{p:d}'][id1:id2],label=f'$p={p:d}$ GT',linestyle='none',marker=markers[i],
                        markerfacecolor=markerfacecolors[i],markeredgecolor=markeredgecolors[i],markeredgewidth=1,markersize=4)
        axes[1][0].plot(x, dictS3p_t[f's{p:d}'][id1:id2],label=f'$p={p:d}$ QKM',linewidth=1.5,linestyle=linestyles[i],color=colors[i])


    axes[1][0].legend(ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 0.0), loc='lower right', fontsize=config.ftsize-1)


    axes[1][0].set_xscale('log')
    axes[1][0].set_yscale('log')
    axes[1][0].set_xlim(0.85,22)
    axes[1][0].set_ylim(1e-6,1.15e1)
    axes[1][0].set_xlabel('$r$',labelpad=0,fontsize=config.ftsize)
    axes[1][0].set_ylabel(r'$S_p(r)$',labelpad=2.5,fontsize=config.ftsize)

    axes[1][0].grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    axes[1][0].set_axisbelow(True)  # 网格线和坐标轴置于底层
    axes[1][0].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[1][0].tick_params(axis='y', which='major',labelsize=config.ftsize,
                        top=False, right=False, length=3, pad=1)
    axes[1][0].tick_params(which='minor',top=False, right=False, length=1.5)

    # 速度结构函数标度律
    id1,id2 = 1,7
    r = dictS1p['r'][id1:id2]
    sn1 = np.zeros(len(p2_list))
    sn3 = np.zeros(len(p2_list))
    for i in range(len(p2_list)):
        # sn1[i] = np.gradient(np.log(dictS1p_t[f's{p2_list[i]:d}'][id1:id2]), np.log(r)).mean()
        # sn3[i] = np.gradient(np.log(dictS3p_t[f's{p2_list[i]:d}'][id1:id2]), np.log(r)).mean()
        sn1[i] = np.gradient(np.log(np.abs(dictS1p_t[f's{p2_list[i]:d}'][id1:id2])), np.log(r)).mean()
        sn3[i] = np.gradient(np.log(np.abs(dictS3p_t[f's{p2_list[i]:d}'][id1:id2])), np.log(r)).mean()

    p2_list = np.array(p2_list)
    axes[1][1].plot(p2_list,p2_list,'k--', linewidth=1.0,label=r'$p$')
    line_gt = axes[1][1].plot(p2_list,sn1,linestyle='none',label='GT',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)
    line_qkm = axes[1][1].plot(p2_list,sn3, label='QKM',linewidth=1.5,linestyle="-",color='#ec5d65')

    axes[1][1].legend([line_gt[0],line_qkm[0]],['GT', 'QKM'],ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
        handletextpad=0.5, bbox_to_anchor=(0., 1.), loc='upper left', fontsize=config.ftsize-1)


    axes[1][1].set_xlabel('$p$',labelpad=0,fontsize=config.ftsize)
    axes[1][1].set_ylabel(r'$\xi_p$',labelpad=2.5,fontsize=config.ftsize)
    axes[1][1].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[1][1].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    axes[1][1].tick_params(which='minor',top=False, right=False, length=1.5)
    axes[1][1].grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    axes[1][1].set_axisbelow(True)  # 网格线和坐标轴置于底层


    # 速度pdf
    # 时间平均
    ux = ux1[10] # k=10
    uy = uy1[10] # k=10
    # ===== GT =====
    # 去均值,合并数据
    ux_prime = ux - np.mean(ux)
    uy_prime = uy - np.mean(uy)
    u_combined = np.concatenate([ux_prime, uy_prime])
    # 计算PDF
    hist_combined, bin_edges = np.histogram(u_combined, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 高斯分布理论值
    sigma_combined = np.std(u_combined)
    gaussian_combined = norm.pdf(bin_centers, loc=0, scale=sigma_combined)
    axes[0][1].bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='GT',color='#DDE9F5')
    axes[0][1].plot(bin_centers, gaussian_combined,linestyle='none',label='GT Gaussian fit',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)

    # ===== QKM =====
    ux = ux3[10] # k=10
    uy = uy3[10] # k=10
    # 去均值,合并数据
    ux_prime = ux - np.mean(ux)
    uy_prime = uy - np.mean(uy)
    u_combined = np.concatenate([ux_prime, uy_prime])
    # 计算PDF
    hist_combined, bin_edges = np.histogram(u_combined, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 高斯分布理论值
    sigma_combined = np.std(u_combined)
    gaussian_combined = norm.pdf(bin_centers, loc=0, scale=sigma_combined)
    axes[0][1].bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='QKM',color='#EDD0C6')
    axes[0][1].plot(bin_centers, gaussian_combined, '-', linewidth=1.5, label='QKM Gaussian fit',alpha=0.8,color='#ec5d65')

    axes[0][1].legend(ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(0., 1.0), loc='upper left', fontsize=config.ftsize-1)

    axes[0][1].set_ylim([0,1.05])
    axes[0][1].set_xlim([-1.5,1.5])
    axes[0][1].set_xlabel('$u$',labelpad=0,fontsize=config.ftsize)
    axes[0][1].set_ylabel('$P_{u}$',labelpad=2.5,fontsize=config.ftsize)
    axes[0][1].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[0][1].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    axes[0][1].tick_params(which='minor',top=False, right=False, length=1.5)


    # 时均能谱
    e1 = e1[10] # k=10
    e3 = e3[10] # k=10

    k01=np.linspace(2.85,4.45,40) # ind3
    # k01=np.linspace(1.85,3.45,40) # ind1
    k02=np.linspace(6,40,100)

    # ind3
    axes[0][0].plot(k01,10**(8.28)*k01**(-5/3),label='$\kappa^{-5/3}$',linewidth=1.0,linestyle='--',color='black')
    # ind1
    # axes[0][0].plot(k01,10**(8.15)*k01**(-5/3),label='$\kappa^{-5/3}$',linewidth=1.0,linestyle='--',color='black')

    axes[0][0].plot(k02,10**(9.9)*k02**(-4.2),label='$\kappa^{-4.2}$',linewidth=1.0,linestyle='--',color='black')


    line_gt = axes[0][0].plot(k1[k1<= (128/3)],e1[k1<= (128/3)],linestyle='none',label='GT',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)
    line_qkm = axes[0][0].plot(k3[k3<= (128/3)],e3[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle="-",color='#ec5d65')

    axes[0][0].legend([line_gt[0],line_qkm[0]],['GT', 'QKM'],ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(0., 0.), loc='lower left', fontsize=config.ftsize-1)

    axes[0][0].set_yscale('log')
    axes[0][0].set_xscale('log')
    axes[0][0].set_xlabel(r'$\kappa$',labelpad=0,fontsize=config.ftsize)
    axes[0][0].set_ylabel('$E(\kappa)$',labelpad=2.5,fontsize=config.ftsize)
    axes[0][0].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[0][0].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    axes[0][0].tick_params(which='minor',top=False, right=False, length=1.5)
    axes[0][0].set_xlim(0.901, 100)
    axes[0][0].set_ylim(1e1, 1e9)
    axes[0][0].grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    axes[0][0].set_axisbelow(True)  # 网格线和坐标轴置于底层

    pos00=axes[0][0].get_position()
    pos01=axes[0][1].get_position()
    pos02=axes[1][1].get_position()

    pos12=axes[1][0].get_position()
    fig.text(pos00.x0-0.0525,pos00.y0+pos00.height*1., r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos01.x0-0.045,pos01.y0+pos01.height*1., r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos02.x0-0.045,pos02.y0+pos02.height*1., r'(d)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos12.x0-0.045,pos12.y0+pos12.height*1., r'(c)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    axes[1][0].text(0.28,0.705, r'$r^2$', transform=axes[1][0].transAxes,
                    ha='center', va='center' ,fontsize=config.ftsize-1)
    axes[1][1].text(0.825,0.9, r'$p^1$', transform=axes[1][1].transAxes,
                    ha='center', va='center' ,fontsize=config.ftsize-1)
    # ind3
    axes[0][0].text(0.345,0.831, r'$\kappa^{-\frac{5}{3}}$', transform=axes[0][0].transAxes,
                    ha='center', va='center' ,fontsize=config.ftsize-1)
    # ind1
    # axes[0][0].text(0.275,0.84, r'$\kappa^{-\frac{5}{3}}$', transform=axes[0][0].transAxes,
    #                 ha='center', va='center' ,fontsize=config.ftsize-1)
    axes[0][0].text(0.675,0.52, r'$\kappa^{-4.2}$', transform=axes[0][0].transAxes,
                    ha='center', va='center',fontsize=config.ftsize-1)

    # fig.savefig(fr'case={case:d}_ind={ind:d}_statistic.png', transparent=True, orientation='portrait',
                # bbox_inches='tight',dpi=600)

    fig.savefig(fr'case={case:d}_ind={ind:d}_statistic.pdf', transparent=True, orientation='portrait',
                bbox_inches='tight')

# ========================================================

def draw_statistic_case2(k1,e1,k3,e3,
                     bins,v1,v3,case,ind):
    '''
        v (np.array): (t,h,h)
        k (np.array): (k,)
        e (np.array): (t,k)
        case,ind (int)
    '''
    config = PlotConfig2(nrow=1,ncol=2,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.4,       # 左侧边距 (cm)
                        margin_right= 0.3,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 0.3,        # 顶部边距 (cm)
                        space_width= 1.5,       # 子图水平间距 (cm)
                        space_height= 1.5,      # 子图垂直间距 (cm)
                        subplot_ratio= 0.9,     # 子图高宽比 (height/width)
                        ftsize=8,             # 基础字体大小
                    )
    fig,axes = config.get_multi()
    # 时均能谱
    tau = 1/0.029
    e1_10 = e1[10] # k=10
    e3_10 = e3[10] # k=10
    e1_30 = e1[30] # k=30
    e3_30 = e3[30] # k=30
    e1_50 = e1[50] # k=50
    e3_50 = e3[50] # k=50
    colors=['#d16d5b','#b7282e','#e64532',]
    markers=['o','^','s']
    linestyles=['-','--','-.']
    markerfacecolors=['#DDE9F5','#AFD9FD','#bbcfe8']
    markeredgecolors=['#7AA7D3','#92bbd7','#659bca']

    line_gt_10 = axes[0][0].plot(k1[k1<= (128/3)],e1_10[k1<= (128/3)],linestyle='none',label='GT',marker=markers[0],
                        markerfacecolor=markerfacecolors[0],markeredgecolor=markeredgecolors[0],markeredgewidth=1,markersize=4)
    line_gt_30 = axes[0][0].plot(k1[k1<= (128/3)],e1_30[k1<= (128/3)],linestyle='none',label='GT',marker=markers[1],
                        markerfacecolor=markerfacecolors[1],markeredgecolor=markeredgecolors[1],markeredgewidth=1,markersize=4)
    line_gt_50 = axes[0][0].plot(k1[k1<= (128/3)],e1_50[k1<= (128/3)],linestyle='none',label='GT',marker=markers[2],
                        markerfacecolor=markerfacecolors[2],markeredgecolor=markeredgecolors[2],markeredgewidth=1,markersize=4)

    line_qkm_10 = axes[0][0].plot(k3[k3<= (128/3)],e3_10[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle=linestyles[0],color=colors[0])
    line_qkm_30 = axes[0][0].plot(k3[k3<= (128/3)],e3_30[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle=linestyles[1],color=colors[1])
    line_qkm_50 = axes[0][0].plot(k3[k3<= (128/3)],e3_50[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle=linestyles[2],color=colors[2])

    axes[0][0].legend([line_gt_10[0],line_qkm_10[0],
                       line_gt_30[0],line_qkm_30[0],
                       line_gt_50[0],line_qkm_50[0],],
                      [fr'${{t}}^{{*}}={10*10/tau:<4.1f}$ GT', fr'${{t}}^{{*}}={10*10/tau:<4.1f}$ QKM',
                       fr'${{t}}^{{*}}={30*10/tau:<4.1f}$ GT', fr'${{t}}^{{*}}={30*10/tau:<4.1f}$ QKM',
                       fr'${{t}}^{{*}}={50*10/tau:<4.1f}$ GT', fr'${{t}}^{{*}}={50*10/tau:<4.1f}$ QKM',],
                      ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=config.ftsize-1)

    axes[0][0].set_yscale('log')
    axes[0][0].set_xscale('log')
    axes[0][0].set_xlim([0.901, 100])
    axes[0][0].set_ylim([1e-2,1e8])
    axes[0][0].set_xlabel(r'$\kappa$',labelpad=0,fontsize=config.ftsize)
    axes[0][0].set_ylabel('$E_{Y}(\kappa)$',labelpad=2.5,fontsize=config.ftsize)
    axes[0][0].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[0][0].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    axes[0][0].tick_params(which='minor',top=False, right=False, length=1.5)
    axes[0][0].grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    axes[0][0].set_axisbelow(True)  # 网格线和坐标轴置于底层

    # 时间平均
    v1=v1[10] # k=10
    v3=v3[10] # k=10


    # ===== GT =====
    hist_combined, bin_edges = np.histogram(v1, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axes[0][1].bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='GT',color='#DDE9F5')
    kde = gaussian_kde(v1.flatten(),bw_method='silverman')
    x = np.linspace(v1.min(),v1.max(),50)
    axes[0][1].plot(x,  kde(x),
                    linestyle='none',label='GT KDE',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)


    # ===== QKM =====
    hist_combined, bin_edges = np.histogram(v3, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axes[0][1].bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='QKM',color='#EDD0C6')
    kde = gaussian_kde(v3.flatten(),bw_method='silverman')
    x = np.linspace(v3.min(),v3.max(),50)
    axes[0][1].plot(x,  kde(x),
                    '-', linewidth=1.5, label='QKM KDE',alpha=0.8,color='#ec5d65')


    axes[0][1].legend(ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=config.ftsize-1)
    axes[0][1].set_xlim([0.18,0.94])
    axes[0][1].set_ylim([0,3.4])
    ticks=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    axes[0][1].set_xticks(ticks)
    axes[0][1].set_xticklabels([f'{i:.1f}' for i in ticks])
    axes[0][1].set_xlabel('$Y_A$',labelpad=0,fontsize=config.ftsize)
    axes[0][1].set_ylabel('$P_{Y}$',labelpad=2.5,fontsize=config.ftsize)
    axes[0][1].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[0][1].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    axes[0][1].tick_params(which='minor',top=False, right=False, length=1.5)


    pos00=axes[0][0].get_position()
    pos01=axes[0][1].get_position()
    fig.text(pos00.x0-0.0525,pos00.y0+pos00.height*1., r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos01.x0-0.045,pos01.y0+pos01.height*1., r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    # fig.savefig(fr'case={case:d}_ind={ind:d}_statistic.png', transparent=True, orientation='portrait',
                # bbox_inches='tight',dpi=600)

    fig.savefig(fr'case={case:d}_ind={ind:d}_statistic.pdf', transparent=True, orientation='portrait',
                bbox_inches='tight')

# ========================================================

def draw_statistic_case0(k1,e1,k3,e3,
                     bins,v1,v3,case,ind):
    '''
        v (np.array): (t,h,h)
        k (np.array): (k,)
        e (np.array): (t,k)
        case,ind (int)
    '''
    config = PlotConfig2(nrow=1,ncol=2,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.4,       # 左侧边距 (cm)
                        margin_right= 0.3,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 0.3,        # 顶部边距 (cm)
                        space_width= 1.5,       # 子图水平间距 (cm)
                        space_height= 1.5,      # 子图垂直间距 (cm)
                        subplot_ratio= 0.9,     # 子图高宽比 (height/width)
                        ftsize=8,             # 基础字体大小
                    )
    fig,axes = config.get_multi()

    # 时均能谱
    tau = np.abs(v1[0]).max()
    e1_10 = e1[10] # k=10
    e3_10 = e3[10] # k=10
    e1_30 = e1[30] # k=30
    e3_30 = e3[30] # k=30
    e1_50 = e1[50] # k=50
    e3_50 = e3[50] # k=50
    colors=['#d16d5b','#b7282e','#e64532',]
    markers=['o','^','s']
    linestyles=['-','--','-.']
    markerfacecolors=['#DDE9F5','#AFD9FD','#bbcfe8']
    markeredgecolors=['#7AA7D3','#92bbd7','#659bca']

    line_gt_10 = axes[0][0].plot(k1[k1<= (128/3)],e1_10[k1<= (128/3)],linestyle='none',label='GT',marker=markers[0],
                        markerfacecolor=markerfacecolors[0],markeredgecolor=markeredgecolors[0],markeredgewidth=1,markersize=4)
    line_gt_30 = axes[0][0].plot(k1[k1<= (128/3)],e1_30[k1<= (128/3)],linestyle='none',label='GT',marker=markers[1],
                        markerfacecolor=markerfacecolors[1],markeredgecolor=markeredgecolors[1],markeredgewidth=1,markersize=4)
    line_gt_50 = axes[0][0].plot(k1[k1<= (128/3)],e1_50[k1<= (128/3)],linestyle='none',label='GT',marker=markers[2],
                        markerfacecolor=markerfacecolors[2],markeredgecolor=markeredgecolors[2],markeredgewidth=1,markersize=4)

    line_qkm_10 = axes[0][0].plot(k3[k3<= (128/3)],e3_10[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle=linestyles[0],color=colors[0])
    line_qkm_30 = axes[0][0].plot(k3[k3<= (128/3)],e3_30[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle=linestyles[1],color=colors[1])
    line_qkm_50 = axes[0][0].plot(k3[k3<= (128/3)],e3_50[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle=linestyles[2],color=colors[2])

    axes[0][0].legend([line_gt_10[0],line_qkm_10[0],
                       line_gt_30[0],line_qkm_30[0],
                       line_gt_50[0],line_qkm_50[0],],
                      [fr'${{t}}^{{*}}={10*0.1/tau:<4.2f}$ GT', fr'${{t}}^{{*}}={10*0.1/tau:<4.2f}$ QKM',
                       fr'${{t}}^{{*}}={30*0.1/tau:<4.2f}$ GT', fr'${{t}}^{{*}}={30*0.1/tau:<4.2f}$ QKM',
                       fr'${{t}}^{{*}}={50*0.1/tau:<4.2f}$ GT', fr'${{t}}^{{*}}={50*0.1/tau:<4.2f}$ QKM',],
                      ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=config.ftsize-1)

    axes[0][0].set_yscale('log')
    axes[0][0].set_xscale('log')
    axes[0][0].set_xlim([0.901, 100])
    axes[0][0].set_ylim([1e-0,1e10])
    axes[0][0].set_xlabel(r'$\kappa$',labelpad=0,fontsize=config.ftsize)
    axes[0][0].set_ylabel('$E(\kappa)$',labelpad=2.5,fontsize=config.ftsize)
    axes[0][0].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[0][0].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    axes[0][0].tick_params(which='minor',top=False, right=False, length=1.5)
    axes[0][0].grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    axes[0][0].set_axisbelow(True)  # 网格线和坐标轴置于底层

    # 时间平均
    v1=v1[10] # k=10
    v3=v3[10] # k=10


    # ===== GT =====
    hist_combined, bin_edges = np.histogram(v1, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axes[0][1].bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='GT',color='#DDE9F5')
    kde = gaussian_kde(v1.flatten(),bw_method='silverman')
    x = np.linspace(v1.min(),v1.max(),50)
    axes[0][1].plot(x,  kde(x),
                    linestyle='none',label='GT KDE',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)


    # ===== QKM =====
    hist_combined, bin_edges = np.histogram(v3, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axes[0][1].bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='QKM',color='#EDD0C6')
    kde = gaussian_kde(v3.flatten(),bw_method='silverman')
    x = np.linspace(v3.min(),v3.max(),50)
    axes[0][1].plot(x,  kde(x),
                    '-', linewidth=1.5, label='QKM KDE',alpha=0.8,color='#ec5d65')


    axes[0][1].legend(ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=config.ftsize-1)
    axes[0][1].set_xlim([-14,14])
    axes[0][1].set_ylim([0,0.18])
    ticks=[-12,-6,0,6,12]
    axes[0][1].set_xticks(ticks)
    axes[0][1].set_xticklabels([f'{i:d}' for i in ticks])
    ticks=[0,0.05,0.1,0.15]
    axes[0][1].set_yticks(ticks)
    axes[0][1].set_yticklabels([f'{i:.2f}' for i in ticks])
    axes[0][1].set_xlabel('$\omega$',labelpad=0,fontsize=config.ftsize)
    axes[0][1].set_ylabel('$P_{\omega}$',labelpad=2.5,fontsize=config.ftsize)
    axes[0][1].tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    axes[0][1].tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    axes[0][1].tick_params(which='minor',top=False, right=False, length=1.5)


    pos00=axes[0][0].get_position()
    pos01=axes[0][1].get_position()
    fig.text(pos00.x0-0.0525,pos00.y0+pos00.height*1., r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos01.x0-0.045,pos01.y0+pos01.height*1., r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    # fig.savefig(fr'case={case:d}_ind={ind:d}_statistic.png', transparent=True, orientation='portrait',
                # bbox_inches='tight',dpi=600)

    fig.savefig(fr'case={case:d}_ind={ind:d}_statistic.pdf', transparent=True, orientation='portrait',
                bbox_inches='tight')

# ========================================================

def draw_ex_case2(v1_ex,v3_ex,k1,e1,k3,e3,bins,case,ind):
    '''
    Args:
        v1_ex (np.array): (t, h, h) GT
        v3_ex (np.array): (t, h, h) QKM
    '''
    config = PlotConfig3(nrow=3,ncol=5,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.4,       # 左侧边距 (cm)
                        margin_right= 0.3,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 0.3,        # 顶部边距 (cm)
                        space_width= 0.175,       # 子图水平间距 (cm)
                        space_height= [0.435,0.8125],      # 子图垂直间距 (cm)
                        subplot_ratio= [0.9,0.9,0.75],     # 子图高宽比 (height/width)
                        ftsize=8,             # 基础字体大小
                    )
    config.set_row_config(row=2,ncols=2,row_space_width=2)
    fig,axes = config.get_multi()

    # 设置 坐标轴范围 ticks,labels
    extent = [0, 128, 0, 128]  # 设置 xmin, xmax, ymin, ymax
    ticks = [0,64,128]

    labels_x = ['-1', '0', '1']
    labels_y = ['-1', '0', '1']

    print(f'v1_ex.min()={v1_ex.min()},v1_ex.max()={v1_ex.max()}')
    vmin = 0.1
    vmax = 1.
    tau = 1/0.029

    # 获取时间步
    t_list = [0,2,4,6,8] # k=61,63,65,67,69

    for k in range(5) :
        # 绘制 GT 图像
        im = axes[0][k].imshow(
            getv(t_list[k], v1_ex),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )

        axes[0][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[0][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[0][k].set_xticklabels(['','',''])
        axes[0][k].set_yticklabels(['','',''])

        # 绘制 QKM 图像
        axes[1][k].imshow(
            getv(t_list[k], v3_ex),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )
        axes[1][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[1][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[1][k].set_xticklabels(['','',''])
        axes[1][k].set_yticklabels(['','',''])

        axes[0][k].set_title(fr"${{t}}^{{*}}={(61+t_list[k])*10/tau:.1f}$",fontsize=config.ftsize,pad=2.0)

    axes[0][0].set_yticklabels(labels_y,fontsize=config.ftsize)
    axes[1][0].set_yticklabels(labels_y,fontsize=config.ftsize)

    axes[0][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)
    axes[1][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)
    axes[2][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)

    for j in range(5):
        axes[1][j].set_xticklabels(labels_x,fontsize=config.ftsize)
        axes[1][j].set_xlabel('$x$',labelpad=0,fontsize=config.ftsize)

    # color bar
    subwidth = axes[0][-1].get_position().x0 - axes[0][-2].get_position().x0 - axes[0][-2].get_position().width
    pos_up = axes[0][-1].get_position()
    pos_down = axes[1][-1].get_position()
    ax_bar = fig.add_axes([pos_down.x0+pos_up.width+subwidth*0.6,
                  pos_down.y0+pos_up.height/4,
                  subwidth*0.5,
                  pos_up.y0-pos_down.y0+pos_up.height/2,])
    cbar = fig.colorbar(im,cax=ax_bar,orientation='vertical')
    cbar.ax.tick_params(
        direction='in',  # 刻度线朝外
        length=2.,         # 刻度线长度（默认是 4，增大后会更凸出）
        width=0.7,          # 刻度线宽度
        colors='black',   # 刻度线颜色
        pad=1.5,          # 刻度标签与刻度线的距离
        top=True,
        bottom=False
    )


    bar_label='$Y_A$'
    bar_ticks=[0.1,0.3,0.5,0.7,0.9]
    bar_tls=[f'{x:.1f}' for x in bar_ticks]

    cbar.set_ticks(bar_ticks)
    cbar.set_ticklabels(bar_tls,fontsize=config.ftsize)

    cbar.set_label('') # 移除默认的 label（如果有）
    # 手动添加 label
    cbar.ax.text(
        0.5,                    # x 位置（0=左，1=右，0.5=居中）
        1.02,                   # y 位置
        bar_label,              # 标签文本
        ha='center',            # 水平居中
        va='bottom',            # 垂直对齐（底部）
        fontsize=config.ftsize,      # 字体大小
        transform=cbar.ax.transAxes  # 使用相对坐标（0-1）
    )

    posleft = axes[1][0].get_position()
    ax = axes[2][0]
    pos = ax.get_position()
    ax.remove()
    ax = fig.add_axes([posleft.x0,pos.y0,pos.width,pos.height])


    # 时间平均
    v1_ex=v1_ex[4] # k=65
    v3_ex=v3_ex[4] # k=65

    # ===== GT =====
    hist_combined, bin_edges = np.histogram(v1_ex, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='GT',color='#DDE9F5')
    kde = gaussian_kde(v1_ex.flatten(),bw_method='silverman')
    x = np.linspace(v1_ex.min(),v1_ex.max(),50)
    ax.plot(x,  kde(x),
                    linestyle='none',label='GT KDE',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)

    # ===== QKM =====
    hist_combined, bin_edges = np.histogram(v3_ex, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='QKM',color='#EDD0C6')
    kde = gaussian_kde(v3_ex.flatten(),bw_method='silverman')
    x = np.linspace(v3_ex.min(),v3_ex.max(),50)
    ax.plot(x,  kde(x),
                    '-', linewidth=1.5, label='QKM KDE',alpha=0.8,color='#ec5d65')

    ax.legend(ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=config.ftsize-1)
    ax.set_xlim([0.18,0.94])
    ax.set_ylim([0,4])
    ticks=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{i:.1f}' for i in ticks])
    ax.set_xlabel('$Y_A$',labelpad=0,fontsize=config.ftsize)
    ax.set_ylabel('$P_Y$',labelpad=2.5,fontsize=config.ftsize)
    ax.tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    ax.tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    ax.tick_params(which='minor',top=False, right=False, length=1.5)
    axes[2][0]=ax

    posright = axes[1][-1].get_position()
    ax = axes[2][1]
    pos = ax.get_position()
    ax.remove()
    ax = fig.add_axes([posright.x1-pos.width,
                       pos.y0,pos.width,pos.height])


    e12 = e1[4] # k=65
    e32 = e3[4] # k=65
    line_gt = ax.plot(k1[k1<= (128/3)],e12[k1<= (128/3)],linestyle='none',label='GT',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)
    line_qkm = ax.plot(k3[k3<= (128/3)],e32[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle="-",color='#ec5d65')

    ax.legend([line_gt[0],line_qkm[0]],['GT', 'QKM'],ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=config.ftsize-1)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.901, 100])
    ax.set_ylim([1e-2,1e8])
    ax.set_xlabel(r'$\kappa$',labelpad=0,fontsize=config.ftsize)
    ax.set_ylabel(r'$E_{Y}(\kappa)$',labelpad=2.5,fontsize=config.ftsize)
    ax.tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    ax.tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    ax.tick_params(which='minor',top=False, right=False, length=1.5)
    ax.grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    ax.set_axisbelow(True)  # 网格线和坐标轴置于底层
    axes[2][1]=ax

    pos0 = axes[0][0].get_position()
    pos1 = axes[1][0].get_position()

    fig.text(pos0.x0-0.048,pos0.y0+pos0.height/2, 'GT', ha='center', va='center', rotation=90, fontsize=config.ftsize)
    fig.text(pos0.x0-0.048,pos1.y0+pos1.height/2, 'QKM', ha='center', va='center', rotation=90,  fontsize=config.ftsize)

    fig.text(pos0.x0-0.058,pos0.y0+pos0.height*1.0005, r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos0.x0-0.058,pos1.y0+pos1.height*1.0005, r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    pos20=axes[2][0].get_position()
    pos21=axes[2][1].get_position()
    fig.text(pos0.x0-0.058,pos20.y1, r'(c)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos21.x0-0.065,pos20.y1, r'(d)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)


    # fig.savefig(fr'case={case:d}_ind={ind:d}_ex.png', transparent=True, orientation='portrait',
                # bbox_inches='tight',dpi=600)

    fig.savefig(fr'case={case:d}_ind={ind:d}_ex.pdf', transparent=True, orientation='portrait',
                bbox_inches='tight')

# ========================================================

def draw_ex_case1(v1,v1_ex,v3_ex,
                ux1,uy1,ux3,uy3,bins,
                k1,e1,k3,e3,case,ind):
    '''
    Args:
        v1_ex (np.array): (t, h, h) GT
        v3_ex (np.array): (t, h, h) QKM
    '''
    config = PlotConfig3(nrow=3,ncol=5,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.4,       # 左侧边距 (cm)
                        margin_right= 0.3,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 0.3,        # 顶部边距 (cm)
                        space_width= 0.175,       # 子图水平间距 (cm)
                        space_height= [0.435,0.8125],      # 子图垂直间距 (cm)
                        subplot_ratio= [0.9,0.9,0.75],     # 子图高宽比 (height/width)
                        ftsize=8,             # 基础字体大小
                    )
    config.set_row_config(row=2,ncols=2,row_space_width=2)
    fig,axes = config.get_multi()

    # 设置 坐标轴范围 ticks,labels
    extent = [0, 128, 0, 128]  # 设置 xmin, xmax, ymin, ymax
    ticks = [0,64,128]

    labels_x = ['0', '$\pi$', '$4\pi$']
    labels_y = ['0', '$\pi$', '$4\pi$']

    print(f'v1_ex.min()={v1_ex.min()},v1_ex.max()={v1_ex.max()}')
    vmin = -9.
    vmax = 9.
    tau = np.abs(v1[0]).max()

    # 获取时间步
    t_list = [0,2,4,6,8] # k=61,63,65,67,69

    for k in range(5) :
        # 绘制 GT 图像
        im = axes[0][k].imshow(
            getv(t_list[k], v1_ex),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )

        axes[0][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[0][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[0][k].set_xticklabels(['','',''])
        axes[0][k].set_yticklabels(['','',''])

        # 绘制 QKM 图像
        axes[1][k].imshow(
            getv(t_list[k], v3_ex),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )
        axes[1][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[1][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[1][k].set_xticklabels(['','',''])
        axes[1][k].set_yticklabels(['','',''])

        axes[0][k].set_title(fr"${{t}}^{{*}}={(61+t_list[k])*0.1/tau:.2f}$",fontsize=config.ftsize,pad=2.0)

    axes[0][0].set_yticklabels(labels_y,fontsize=config.ftsize)
    axes[1][0].set_yticklabels(labels_y,fontsize=config.ftsize)

    axes[0][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)
    axes[1][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)
    axes[2][0].set_ylabel('$y$',labelpad=0,fontsize=config.ftsize)

    for j in range(5):
        axes[1][j].set_xticklabels(labels_x,fontsize=config.ftsize)
        axes[1][j].set_xlabel('$x$',labelpad=0,fontsize=config.ftsize)

    # color bar
    subwidth = axes[0][-1].get_position().x0 - axes[0][-2].get_position().x0 - axes[0][-2].get_position().width
    pos_up = axes[0][-1].get_position()
    pos_down = axes[1][-1].get_position()
    ax_bar = fig.add_axes([pos_down.x0+pos_up.width+subwidth*0.6,
                  pos_down.y0+pos_up.height/4,
                  subwidth*0.5,
                  pos_up.y0-pos_down.y0+pos_up.height/2,])
    cbar = fig.colorbar(im,cax=ax_bar,orientation='vertical')
    cbar.ax.tick_params(
        direction='in',  # 刻度线朝外
        length=2.,         # 刻度线长度（默认是 4，增大后会更凸出）
        width=0.7,          # 刻度线宽度
        colors='black',   # 刻度线颜色
        pad=1.5,          # 刻度标签与刻度线的距离
        top=True,
        bottom=False
    )

    bar_label='$\omega$'
    bar_ticks=[-8,-4,0,4,8]
    bar_tls=[f'{x:d}' for x in bar_ticks]


    cbar.set_ticks(bar_ticks)
    cbar.set_ticklabels(bar_tls,fontsize=config.ftsize)

    cbar.set_label('') # 移除默认的 label（如果有）
    # 手动添加 label
    cbar.ax.text(
        0.5,                    # x 位置（0=左，1=右，0.5=居中）
        1.02,                   # y 位置
        bar_label,              # 标签文本
        ha='center',            # 水平居中
        va='bottom',            # 垂直对齐（底部）
        fontsize=config.ftsize,      # 字体大小
        transform=cbar.ax.transAxes  # 使用相对坐标（0-1）
    )

    posleft = axes[1][0].get_position()
    ax = axes[2][0]
    pos = ax.get_position()
    ax.remove()
    ax = fig.add_axes([posleft.x0,pos.y0,pos.width,pos.height])


    # 速度pdf
    # 时间平均
    ux = ux1[4] # k=65
    uy = uy1[4] # k=65
    # ===== GT =====
    # 去均值,合并数据
    ux_prime = ux - np.mean(ux)
    uy_prime = uy - np.mean(uy)
    u_combined = np.concatenate([ux_prime, uy_prime])
    # 计算PDF
    hist_combined, bin_edges = np.histogram(u_combined, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 高斯分布理论值
    sigma_combined = np.std(u_combined)
    gaussian_combined = norm.pdf(bin_centers, loc=0, scale=sigma_combined)
    ax.bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='GT',color='#DDE9F5')
    ax.plot(bin_centers, gaussian_combined,linestyle='none',label='GT Gaussian fit',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)

    # ===== QKM =====
    ux = ux3[4] # k=65
    uy = uy3[4] # k=65
    # 去均值,合并数据
    ux_prime = ux - np.mean(ux)
    uy_prime = uy - np.mean(uy)
    u_combined = np.concatenate([ux_prime, uy_prime])
    # 计算PDF
    hist_combined, bin_edges = np.histogram(u_combined, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 高斯分布理论值
    sigma_combined = np.std(u_combined)
    gaussian_combined = norm.pdf(bin_centers, loc=0, scale=sigma_combined)
    ax.bar(bin_centers, hist_combined, width=np.diff(bin_edges)[0], alpha=0.5, label='QKM',color='#EDD0C6')
    ax.plot(bin_centers, gaussian_combined, '-', linewidth=1.5, label='QKM Gaussian fit',alpha=0.8,color='#ec5d65')

    ax.legend(ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(0., 1.0), loc='upper left', fontsize=config.ftsize-1)

    ax.set_ylim([0,1.05])
    ax.set_xlim([-1.5,1.5])
    ax.set_xlabel('$u$',labelpad=0,fontsize=config.ftsize)
    ax.set_ylabel('$P_{u}$',labelpad=2.5,fontsize=config.ftsize)
    ax.tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    ax.tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    ax.tick_params(which='minor',top=False, right=False, length=1.5)
    axes[2][0]=ax




    posright = axes[1][-1].get_position()
    ax = axes[2][1]
    pos = ax.get_position()
    ax.remove()
    ax = fig.add_axes([posright.x1-pos.width,
                       pos.y0,pos.width,pos.height])


    e12 = e1[4] # k=65
    e32 = e3[4] # k=65

    k01=np.linspace(1.85,3.5,40)
    k02=np.linspace(6,40,100)


    # ind1
    # ax.plot(k01,10**(8.25)*k01**(-5/3),label='$\kappa^{-5/3}$',linewidth=1.0,linestyle='--',color='black')
    # ax.plot(k02,10**(9.97)*k02**(-4.2),label='$\kappa^{-4.2}$',linewidth=1.0,linestyle='--',color='black')

    # ind3
    ax.plot(k01,10**(8.2)*k01**(-5/3),label='$\kappa^{-5/3}$',linewidth=1.0,linestyle='--',color='black')
    ax.plot(k02,10**(9.92)*k02**(-4.2),label='$\kappa^{-4.2}$',linewidth=1.0,linestyle='--',color='black')



    line_gt = ax.plot(k1[k1<= (128/3)],e12[k1<= (128/3)],linestyle='none',label='GT',marker='o',
                        markerfacecolor='#DDE9F5',markeredgecolor='#7AA7D3',markeredgewidth=1,markersize=4)
    line_qkm = ax.plot(k3[k3<= (128/3)],e32[k3<= (128/3)],label='QKM',linewidth=1.5,linestyle="-",color='#ec5d65')

    ax.legend([line_gt[0],line_qkm[0]],['GT', 'QKM'],ncol=1, frameon=False, labelspacing=0.2, handlelength=2.0,
           handletextpad=0.5, bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=config.ftsize-1)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlim(0.901, 100)
    ax.set_ylim(1e1, 1e9)

    ax.set_xlabel(r'$\kappa$',labelpad=0,fontsize=config.ftsize)
    ax.set_ylabel('$E(\kappa)$',labelpad=2.5,fontsize=config.ftsize)
    ax.tick_params(axis='x', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=3)
    ax.tick_params(axis='y', which='major',labelsize=config.ftsize,
                    top=False, right=False, length=3, pad=1)
    ax.tick_params(which='minor',top=False, right=False, length=0)
    ax.grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    ax.set_axisbelow(True)  # 网格线和坐标轴置于底层
    ax.text(0.275,0.86, r'$\kappa^{-\frac{5}{3}}$', transform=ax.transAxes,
                    ha='center', va='center' ,fontsize=config.ftsize-1)
    ax.text(0.685,0.53, r'$\kappa^{-4.2}$', transform=ax.transAxes,
                    ha='center', va='center',fontsize=config.ftsize-1)

    axes[2][1]=ax

    pos0 = axes[0][0].get_position()
    pos1 = axes[1][0].get_position()

    fig.text(pos0.x0-0.048,pos0.y0+pos0.height/2, 'GT', ha='center', va='center', rotation=90, fontsize=config.ftsize)
    fig.text(pos0.x0-0.048,pos1.y0+pos1.height/2, 'QKM', ha='center', va='center', rotation=90,  fontsize=config.ftsize)

    fig.text(pos0.x0-0.058,pos0.y0+pos0.height*1.0005, r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos0.x0-0.058,pos1.y0+pos1.height*1.0005, r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    pos20=axes[2][0].get_position()
    pos21=axes[2][1].get_position()

    fig.text(pos0.x0-0.058,pos20.y1, r'(c)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos21.x0-0.065,pos20.y1, r'(d)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    # fig.savefig(fr'case={case:d}_ind={ind:d}_ex.png', transparent=True, orientation='portrait',
                # bbox_inches='tight',dpi=600)

    fig.savefig(fr'case={case:d}_ind={ind:d}_ex.pdf', transparent=True, orientation='portrait',
                bbox_inches='tight')

# ========================================================

def draw_3dfall_rmse(log_dict_case1,log_dict_case2,log_dict_case0):
    config = PlotConfig2(nrow=2,ncol=3,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 0.5,       # 左侧边距 (cm)
                        margin_right= 1.2,      # 右侧边距 (cm)
                        margin_bottom= 0.9,     # 底部边距 (cm)
                        margin_top= 0.5,        # 顶部边距 (cm)
                        space_width= 1.275,       # 子图水平间距 (cm)
                        space_height= 0.75,      # 子图垂直间距 (cm)
                        subplot_ratio= 0.9,     # 子图高宽比 (height/width)
                        ftsize=8,             # 基础字体大小
                    )
    config.set_row_config(1,3,row_width_scale=1.225)
    fig,axes = config.get_multi()

    ax = axes[0][0]
    pos = ax.get_position()
    pos1 = axes[1][0].get_position()
    ax.remove()
    ax = fig.add_axes([pos1.x0+(pos1.width-pos.width)*0.618,
        pos.y0,pos.width,pos.height,])

    # 主图设置
    line1 = ax.plot(log_dict_case2['train']['epoch'], np.log10(log_dict_case2['train']['loss']),
                    label='Training loss', linewidth=1.5, color='#2b6a99')
    line2 = ax.plot(log_dict_case2['val']['epoch'], np.log10(log_dict_case2['val']['loss']),
                    label='Validation loss', linewidth=1.5, color='#f16c23')

    # 坐标轴标签设置
    ax.set_xlabel('Epoch', labelpad=0, fontsize=config.ftsize)
    ax.set_ylabel('Loss', labelpad=2.5, fontsize=config.ftsize)

    # 刻度参数设置
    ax.tick_params(axis='x', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=3)
    ax.tick_params(axis='y', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=1)
    ax.tick_params(which='minor', top=False, right=False, length=1.5)

    # 坐标范围设置
    ax.set_xlim([-30, 430])
    ax.set_ylim([np.log10(0.001), np.log10(100)])

    # 主y轴刻度设置 - 使用FixedLocator确保精确控制
    main_ticks=np.array([0.001,0.01,0.1,1,10,100])
    ax.set_xticks([0,200,400])
    ax.set_xticklabels([0,200,400])

    ax.set_yticks(np.log10(main_ticks))
    ax.set_yticklabels([r'$10^{-3}$',  # 0.001
                    r'$10^{-2}$',  # 0.01
                    r'$10^{-1}$',  # 0.1
                    r'$10^{0}$',   # 1
                    r'$10^{1}$',   # 10
                    r'$10^{2}$'])  # 100
    ax.grid(True, color='#EBEBEB', linewidth=0.5, alpha=0.5, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    # 创建副y轴
    ax2 = ax.twinx()
    line3 = ax2.plot(log_dict_case2['train']['epoch'], log_dict_case2['train']['lr'],
                    color='#1b7c3d', linestyle='--', label='Learning rate', linewidth=1.5)

    # 副y轴刻度设置 - 与主y轴严格对齐
    ax2.set_ylim(0.00001, 0.00024)
    secondary_ticks = np.linspace(0.00001, 0.00024, len(main_ticks))  # 相同数量刻度
    ax2.set_yticks(secondary_ticks)

    # 刻度标签格式化
    def format_ticks(value, pos):
        """自定义刻度标签格式化函数"""
        scaled = value * 1e4
        if scaled < 1:
            return f"{scaled:.1f}"  # 小于1的值显示1位小数
        return f"{int(scaled)}" if scaled.is_integer() else f"{scaled:.1f}"

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    # 副y轴样式设置
    ax2.tick_params(axis='y', which='major', labelsize=config.ftsize,
                    top=False, right=True, length=3, pad=2,
                    color='#1b7c3d', labelcolor='#1b7c3d')
    ax2.tick_params(which='minor', top=False, right=True,
                length=1.5, color='#1b7c3d', labelcolor='#1b7c3d')

    # 添加科学计数法标注
    ax2.set_ylabel('Learning rate', color='#1b7c3d', fontsize=config.ftsize)
    ax2.text(1.0, 1.015, r'$\times 10^{-4}$', transform=ax2.transAxes,
            color='#1b7c3d', fontsize=config.ftsize, ha='right', va='bottom')

    # 合并图例
    lines = line3 + line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], ncol=1, frameon=False,
            labelspacing=0.2, handlelength=2.0, handletextpad=0.5,
            bbox_to_anchor=(1.005, 1.005), loc='upper right', fontsize=config.ftsize-1)
    axes[0][0]=ax


    ax = axes[0][1]
    pos = ax.get_position()
    pos1 = axes[1][1].get_position()
    ax.remove()
    ax = fig.add_axes([pos1.x0+(pos1.width-pos.width)*0.618,
        pos.y0,pos.width,pos.height,])

    # 主图设置
    line1 = ax.plot(log_dict_case0['train']['epoch'], np.log10(log_dict_case0['train']['loss']),
                    label='Training loss', linewidth=1.5, color='#2b6a99')
    line2 = ax.plot(log_dict_case0['val']['epoch'], np.log10(log_dict_case0['val']['loss']),
                    label='Validation loss', linewidth=1.5, color='#f16c23')

    # 坐标轴标签设置
    ax.set_xlabel('Epoch', labelpad=0, fontsize=config.ftsize)
    ax.set_ylabel('Loss', labelpad=2.5, fontsize=config.ftsize)

    # 刻度参数设置
    ax.tick_params(axis='x', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=3)
    ax.tick_params(axis='y', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=1)
    ax.tick_params(which='minor', top=False, right=False, length=1.5)

    # 坐标范围设置
    ax.set_xlim([-30, 430])
    ax.set_ylim([np.log10(1e-3), np.log10(1e1)])

    # 主y轴刻度设置 - 使用FixedLocator确保精确控制
    ax.set_xticks([0,200,400])
    ax.set_xticklabels([0,200,400])
    main_ticks=np.array([0.001,0.01,0.1,1,10])
    ax.set_yticks(np.log10(main_ticks))
    ax.set_yticklabels([r'$10^{-3}$',  # 0.001
                    r'$10^{-2}$',  # 0.01
                    r'$10^{-1}$',  # 0.1
                    r'$10^{0}$',   # 1
                    r'$10^{1}$'])  # 10
    ax.grid(True, color='#EBEBEB', linewidth=0.5, alpha=0.5, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    # 创建副y轴
    ax2 = ax.twinx()
    line3 = ax2.plot(log_dict_case0['train']['epoch'], log_dict_case0['train']['lr'],
                    color='#1b7c3d', linestyle='--', label='Learning rate', linewidth=1.5)

    # 副y轴刻度设置 - 与主y轴严格对齐
    ax2.set_ylim(0.000008, 0.00012)
    secondary_ticks = np.linspace(0.000008, 0.00012, len(main_ticks))  # 相同数量刻度
    ax2.set_yticks(secondary_ticks)

    # 刻度标签格式化
    def format_ticks(value, pos):
        """自定义刻度标签格式化函数"""
        scaled = value * 1e4
        if scaled < 1:
            return f"{scaled:.1f}"  # 小于1的值显示1位小数
        return f"{int(scaled)}" if scaled.is_integer() else f"{scaled:.1f}"

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    # 副y轴样式设置
    ax2.tick_params(axis='y', which='major', labelsize=config.ftsize,
                    top=False, right=True, length=3, pad=2,
                    color='#1b7c3d', labelcolor='#1b7c3d')
    ax2.tick_params(which='minor', top=False, right=True,
                length=1.5, color='#1b7c3d', labelcolor='#1b7c3d')

    # 添加科学计数法标注
    ax2.set_ylabel('Learning rate', color='#1b7c3d', fontsize=config.ftsize)
    ax2.text(1.0, 1.015, r'$\times 10^{-4}$', transform=ax2.transAxes,
            color='#1b7c3d', fontsize=config.ftsize, ha='right', va='bottom')

    # 合并图例
    lines = line3 + line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], ncol=1, frameon=False,
            labelspacing=0.2, handlelength=2.0, handletextpad=0.5,
            bbox_to_anchor=(1.005, 1.005), loc='upper right', fontsize=config.ftsize-1)
    axes[0][1]=ax


    ax = axes[0][2]
    pos = ax.get_position()
    pos1 = axes[1][2].get_position()
    ax.remove()
    ax = fig.add_axes([pos1.x0+(pos1.width-pos.width)*0.618,
        pos.y0,pos.width,pos.height,])

    # 主图设置
    line1 = ax.plot(log_dict_case1['train']['epoch'], log_dict_case1['train']['loss'],
                    label='Training loss', linewidth=1.5, color='#2b6a99')
    line2 = ax.plot(log_dict_case1['val']['epoch'], log_dict_case1['val']['loss'],
                    label='Validation loss', linewidth=1.5, color='#f16c23')

    # 坐标轴标签设置
    ax.set_xlabel('Epoch', labelpad=0, fontsize=config.ftsize)
    ax.set_ylabel('Loss', labelpad=2.5, fontsize=config.ftsize)

    # 刻度参数设置
    ax.tick_params(axis='x', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=3)
    ax.tick_params(axis='y', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=1)
    ax.tick_params(which='minor', top=False, right=False, length=1.5)

    # 坐标范围设置
    ax.set_xlim([-5, 105])
    ax.set_ylim([0, 1.8])

    # 主y轴刻度设置 - 使用FixedLocator确保精确控制
    main_ticks = np.arange(0, 1.81, 0.3)  # 从0到1.8，步长0.3
    ax.set_yticks(main_ticks)
    ax.grid(True, color='#EBEBEB', linewidth=0.5, alpha=0.5, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    # 创建副y轴
    ax2 = ax.twinx()
    line3 = ax2.plot(log_dict_case1['train']['epoch'], log_dict_case1['train']['lr'],
                    color='#1b7c3d', linestyle='--', label='Learning rate', linewidth=1.5)

    # 副y轴刻度设置 - 与主y轴严格对齐
    ax2.set_ylim(0.0001, 0.001)
    secondary_ticks = np.linspace(0.0001, 0.001, len(main_ticks))  # 相同数量刻度
    ax2.set_yticks(secondary_ticks)

    # 刻度标签格式化
    def format_ticks(value,pos):
        """自定义刻度标签格式化函数"""
        scaled = value * 1e3
        return f"{scaled:.1f}"

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    # 副y轴样式设置
    ax2.tick_params(axis='y', which='major', labelsize=config.ftsize,
                    top=False, right=True, length=3, pad=2, color='#1b7c3d',labelcolor='#1b7c3d')
    ax2.tick_params(which='minor', top=False, right=True, length=1.5, color='#1b7c3d',labelcolor='#1b7c3d')

    # 添加科学计数法标注
    ax2.set_ylabel('Learning rate', color='#1b7c3d', fontsize=config.ftsize)
    ax2.text(1.0, 1.02, r'$\times 10^{-3}$', transform=ax2.transAxes,
            color='#1b7c3d', fontsize=config.ftsize, ha='right', va='bottom')

    # 合并图例
    lines = line3 + line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], ncol=1, frameon=False,
            labelspacing=0.2, handlelength=2.0, handletextpad=0.5,
            bbox_to_anchor=(1.005, 1.005), loc='upper right', fontsize=config.ftsize-1)
    axes[0][2]=ax



    # validation loss
    data_case0 = [  [4.9684e-2,5.2165e-2,4.1594e-2], # h=3 : C=[8,12,16]
                    [2.5877e-2,1.7646e-2,1.2970e-2], # h=4 : C=[8,12,16]
                    [1.8136e-2,1.3040e-2,9.0219e-3]] # h=5 : C=[8,12,16]

    data_case1 = [  [2.7643e-1,2.6232e-1,2.6818e-1], # h=3 : C=[8,12,16]
                    [2.2284e-1,2.1468e-1,2.1502e-1], # h=4 : C=[8,12,16]
                    [2.4778e-1,2.5133e-1,2.4203e-1]] # h=5 : C=[8,12,16]

    data_case2 = [  [2.7998e-2,3.7154e-2,2.1343e-2], # h=3 : C=[8,12,16]
                    [2.0713e-2,7.1385e-3,5.3179e-3], # h=4 : C=[8,12,16]
                    [6.4290e-3,7.7214e-3,7.5480e-3]] # h=5 : C=[8,12,16]

    # RMSE

    # data_case0 = [  [7.7132e-2,7.9347e-2,5.2774e-2], # h=3 : C=[8,12,16]
    #                 [4.0213e-2,2.8675e-2,2.3437e-2], # h=4 : C=[8,12,16]
    #                 [2.6958e-2,2.0278e-2,1.4864e-2]] # h=5 : C=[8,12,16]

    # data_case1 = [  [5.4986e-1,5.2357e-1,5.3587e-1], # h=3 : C=[8,12,16]
    #                 [4.3891e-1,4.1911e-1,4.2931e-1], # h=4 : C=[8,12,16]
    #                 [4.8327e-1,4.9744e-1,4.7736e-1]] # h=5 : C=[8,12,16]

    # data_case2 = [  [4.8927e-2,4.4494e-2,3.7748e-2], # h=3 : C=[8,12,16]
    #                 [3.8587e-2,1.3950e-2,1.0344e-2], # h=4 : C=[8,12,16]
    #                 [1.2502e-2,1.2902e-2,1.5178e-2]] # h=5 : C=[8,12,16]

    ax = axes[1][0]
    pos = ax.get_position()
    ax.remove()
    ax = fig.add_axes(pos,projection='3d')

    Z = np.array(data_case2).T # 注意转置
    zmin=4.e-3
    zmax=4.e-2

    # 设置柱状图参数
    dx = 0.618**2
    dy = dx*4
    dz = Z.ravel()-zmin # 柱子高度
    xpos, ypos = np.meshgrid(np.array([3,4,5]), np.array([8,12,16]))
    xpos = xpos.ravel() - dx/2  # 居中显示
    ypos = ypos.ravel() - dy/2
    zpos = np.ones_like(dz)*zmin    # 底部从zmin开始

    hex_colors=['#b96570','#d37b6d','#e0a981',
                '#ecd09c','#d4daa1','#a3c8a4',
                '#79b4a0','#6888a5','#706d94',]
    colors = [mcolors.to_rgba(color) for color in hex_colors]


    # normalized_values = (Z - Z.min()) / (Z.max() - Z.min())
    # colors = plt.cm.RdBu_r(normalized_values).reshape(-1,4)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
                color=colors,
                edgecolor='gray',
                linewidth=0.5,
                alpha=0.9,
                shade=False)

    # 坐标轴设置
    r = 1
    ax.set_xlim([3-dx*r, 5+dx*r])
    ax.set_ylim([8-dy*r, 16+dy*r])
    ax.set_zlim([zmin,zmax])

    xticks=[3,4,5]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{i:d}' for i in xticks])
    yticks=[8,12,16]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{i:d}' for i in yticks])
    ax.set_zlim([zmin,zmax])

    # zticks=[0.01,0.02,0.03,0.04,0.05] # RMSE
    zticks=np.linspace(zmin,zmax,5) # validation loss
    ax.set_zticks(zticks)
    def format_ticks(value, pos):
        return f"{value * 1e2:.1f}"
    ax.zaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    ax.text2D(1.,0.74,r'$\times 10^{-2}$',
              transform=ax.transAxes,color='k',fontsize=config.ftsize,
              ha='right',va='bottom',rotation=-22.5)
    ax.set_xlabel("$h$", fontsize=config.ftsize,labelpad=-5.)
    ax.set_ylabel("$c$", fontsize=config.ftsize,labelpad=-5.)
    ax.text2D(1.16,0.5,"Validation loss",
            transform=ax.transAxes,color='k',fontsize=config.ftsize,
            ha='center',va='center',rotation=90)
    ax.tick_params(axis='x', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(axis='y', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(axis='z', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(which='minor', top=False, right=False, length=1.5)

    ax.view_init(elev=28, azim=135)
    ax.invert_xaxis()
    ax.set_box_aspect((1, 1, 0.75))   # 优化比例防止变形
    ax.xaxis.pane.set_alpha(0.5)
    ax.yaxis.pane.set_alpha(0.5)
    ax.zaxis.pane.set_alpha(0.5)
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    ax.yaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    ax.zaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    # ax.grid(False)
    axes[1][0]=ax


    ax = axes[1][1]
    pos = ax.get_position()
    ax.remove()
    ax = fig.add_axes(pos,projection='3d')

    Z = np.array(data_case0).T # 注意转置
    zmin=6.e-3
    zmax=6.e-2

    # 设置柱状图参数
    dx = 0.618**2
    dy = dx*4
    dz = Z.ravel()-zmin # 柱子高度
    xpos, ypos = np.meshgrid(np.array([3,4,5]), np.array([8,12,16]))
    xpos = xpos.ravel() - dx/2  # 居中显示
    ypos = ypos.ravel() - dy/2
    zpos = np.ones_like(dz)*zmin    # 底部从zmin开始

    hex_colors=['#b96570','#d37b6d','#e0a981',
                '#ecd09c','#d4daa1','#a3c8a4',
                '#79b4a0','#6888a5','#706d94',]
    colors = [mcolors.to_rgba(color) for color in hex_colors]


    # normalized_values = (Z - Z.min()) / (Z.max() - Z.min())
    # colors = plt.cm.RdBu_r(normalized_values).reshape(-1,4)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
                color=colors,
                edgecolor='gray',
                linewidth=0.5,
                alpha=0.9,
                shade=False)

    # 坐标轴设置
    r = 1
    ax.set_xlim([3-dx*r, 5+dx*r])
    ax.set_ylim([8-dy*r, 16+dy*r])
    ax.set_zlim([zmin,zmax])

    xticks=[3,4,5]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{i:d}' for i in xticks])
    yticks=[8,12,16]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{i:d}' for i in yticks])
    ax.set_zlim([zmin,zmax])

    # zticks=[0.01,0.02,0.03,0.04,0.05] # RMSE
    zticks=np.linspace(zmin,zmax,5) # validation loss
    ax.set_zticks(zticks)
    def format_ticks(value, pos):
        return f"{value * 1e2:.1f}"
    ax.zaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    ax.text2D(1.,0.74,r'$\times 10^{-2}$',
              transform=ax.transAxes,color='k',fontsize=config.ftsize,
              ha='right',va='bottom',rotation=-22.5)
    ax.set_xlabel("$h$", fontsize=config.ftsize,labelpad=-5.)
    ax.set_ylabel("$c$", fontsize=config.ftsize,labelpad=-5.)
    ax.text2D(1.16,0.5,"Validation loss",
            transform=ax.transAxes,color='k',fontsize=config.ftsize,
            ha='center',va='center',rotation=90)
    ax.tick_params(axis='x', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(axis='y', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(axis='z', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(which='minor', top=False, right=False, length=1.5)

    ax.view_init(elev=28, azim=135)
    ax.invert_xaxis()
    ax.set_box_aspect((1, 1, 0.75))   # 优化比例防止变形
    ax.xaxis.pane.set_alpha(0.5)
    ax.yaxis.pane.set_alpha(0.5)
    ax.zaxis.pane.set_alpha(0.5)
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    ax.yaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    ax.zaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    # ax.grid(False)

    axes[1][1]=ax


    ax = axes[1][2]
    pos = ax.get_position()
    ax.remove()
    ax = fig.add_axes(pos,projection='3d')

    Z = np.array(data_case1).T # 注意转置
    zmin=3.75e-1
    zmax=6e-1

    zmin=2.e-1
    zmax=2.8e-1


    # 设置柱状图参数
    dx = 0.618**2
    dy = dx*4
    dz = Z.ravel()-zmin # 柱子高度
    xpos, ypos = np.meshgrid(np.array([3,4,5]), np.array([8,12,16]))
    xpos = xpos.ravel() - dx/2  # 居中显示
    ypos = ypos.ravel() - dy/2
    zpos = np.ones_like(dz)*zmin    # 底部从zmin开始

    hex_colors=['#b96570','#d37b6d','#e0a981',
                '#ecd09c','#d4daa1','#a3c8a4',
                '#79b4a0','#6888a5','#706d94',]
    colors = [mcolors.to_rgba(color) for color in hex_colors]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
                color=colors,
                edgecolor='gray',
                linewidth=0.5,
                alpha=0.9,
                shade=False)

    # 坐标轴设置
    r = 1
    ax.set_xlim([3-dx*r, 5+dx*r])
    ax.set_ylim([8-dy*r, 16+dy*r])
    ax.set_zlim([zmin,zmax])

    xticks=[3,4,5]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{i:d}' for i in xticks])
    yticks=[8,12,16]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{i:d}' for i in yticks])
    ax.set_zlim([zmin,zmax])
    # zticks=[0.4,0.45,0.5,0.55] # RMSE
    zticks=[0.2,0.22,0.24,0.26,0.28,0.3] # validation loss
    ax.set_zticks(zticks)
    ax.set_zticklabels([f'{i*10:.1f}' for i in zticks])

    ax.text2D(1.,0.74,r'$\times 10^{-1}$',
              transform=ax.transAxes,color='k',fontsize=config.ftsize,
              ha='right',va='bottom',rotation=-22.5)

    ax.set_xlabel("$h$", fontsize=config.ftsize,labelpad=-5.)
    ax.set_ylabel("$c$", fontsize=config.ftsize,labelpad=-5.)
    ax.text2D(1.16,0.5,"Validation loss",
            transform=ax.transAxes,color='k',fontsize=config.ftsize,
            ha='center',va='center',rotation=90)
    ax.tick_params(axis='x', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(axis='y', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(axis='z', which='major', labelsize=config.ftsize,
                top=False, right=False, length=3, pad=-2.5)
    ax.tick_params(which='minor', top=False, right=False, length=1.5)

    ax.view_init(elev=28, azim=135)
    ax.invert_xaxis()
    ax.set_box_aspect((1, 1, 0.75))   # 优化比例防止变形
    ax.xaxis.pane.set_alpha(0.5)
    ax.yaxis.pane.set_alpha(0.5)
    ax.zaxis.pane.set_alpha(0.5)
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    ax.yaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    ax.zaxis._axinfo["grid"].update({"color": "#EBEBEB", "linewidth": 0.5,'linestyle':'-','alpha':0.5})
    # ax.grid(False)
    axes[1][2]=ax

    pos0=axes[0][0].get_position()
    pos1=axes[0][1].get_position()
    pos2=axes[1][0].get_position()
    pos3=axes[1][1].get_position()
    pos4=axes[0][2].get_position()
    pos5=axes[1][2].get_position()

    fig.text(pos0.x0-0.058,pos0.y0+pos0.height*1.0005, r'(a)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos1.x0-0.058,pos1.y0+pos1.height*1.0005, r'(b)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos4.x0-0.058,pos4.y0+pos4.height*1.0005, r'(c)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    fig.text(pos0.x0-0.058,pos2.y0+pos1.height*1.1, r'(d)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos1.x0-0.058,pos3.y0+pos1.height*1.1, r'(e)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)
    fig.text(pos4.x0-0.058,pos5.y0+pos1.height*1.1, r'(f)', ha='center', va='center', family='Times New Roman',fontsize=config.ftsize)

    # fig.savefig(fr'3drmse.png', transparent=True, orientation='portrait',
                # bbox_inches='tight',dpi=600)

    fig.savefig(fr'3drmse.pdf', transparent=True, orientation='portrait',
                bbox_inches='tight')

# ========================================================

def draw_dataset(vort,case,idx):
    '''
    Args:
        vort (np.array): (b, t, h, h) GT
        case (int): 0,1,2
        idx (list[tuple[int, int]]): e.g. [(0,0),(0,1)]
    '''
    folder_path = f"./case={case:d}_schematic"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建")
    else:
        print(f"文件夹 '{folder_path}' 已存在")
    print(f"默认 pad_inches 值: {mpl.rcParams['savefig.pad_inches']}")
    extent = [0, 128, 0, 128]  # 设置 xmin, xmax, ymin, ymax
    for i in range(len(idx)):
        config = PlotConfig3(nrow=1,ncol=1,
                            plot_width= 5.1,       # 总画布宽度 (cm)
                            margin_left= 0.1,       # 左侧边距 (cm)
                            margin_right= 0.1,      # 右侧边距 (cm)
                            margin_bottom= 0.1,     # 底部边距 (cm)
                            margin_top= 0.1,        # 顶部边距 (cm)
                            space_width= 0.1,       # 子图水平间距 (cm)
                            space_height= 0.1,      # 子图垂直间距 (cm)
                            subplot_ratio= 1.0,     # 子图高宽比 (height/width)
                            ftsize=8,             # 基础字体大小
                        )
        fig,ax = config.get_simple()
        value = vort[idx[i]]
        ax.imshow(
                value,
                cmap='RdBu_r',
                animated=False,
                vmin=value.min(),
                vmax=value.max(),
                extent=extent  # 设置坐标轴范围
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.savefig(folder_path+'/'+f'idx_{idx[i][0]:d}_{idx[i][1]:d}',
                    transparent=True, orientation='portrait',
                    bbox_inches='tight',pad_inches=0.1,dpi=600)

def draw_dataset_qkm(v1,v3,case,ind,idx):
    '''
    Args:
        v1 (np.array): (t, h, h) GT
        v3 (np.array): (t, h, h) QKM
        case (int): 0,1,2
        ind (int)
        idx (list[int]): e.g. [0,1]
    '''
    folder_path = f"./case={case:d}_schematic"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建")
    else:
        print(f"文件夹 '{folder_path}' 已存在")
    print(f"默认 pad_inches 值: {mpl.rcParams['savefig.pad_inches']}")
    extent = [0, 128, 0, 128]  # 设置 xmin, xmax, ymin, ymax
    for i in range(len(idx)):
        config = PlotConfig3(nrow=1,ncol=1,
                            plot_width= 5.1,       # 总画布宽度 (cm)
                            margin_left= 0.1,       # 左侧边距 (cm)
                            margin_right= 0.1,      # 右侧边距 (cm)
                            margin_bottom= 0.1,     # 底部边距 (cm)
                            margin_top= 0.1,        # 顶部边距 (cm)
                            space_width= 0.1,       # 子图水平间距 (cm)
                            space_height= 0.1,      # 子图垂直间距 (cm)
                            subplot_ratio= 1.0,     # 子图高宽比 (height/width)
                            ftsize=8,             # 基础字体大小
                        )
        fig,ax = config.get_simple()
        ax.imshow(
                v3[idx[i]],
                cmap='RdBu_r',
                animated=False,
                vmin=v1[idx[i]].min(),
                vmax=v1[idx[i]].max(),
                extent=extent  # 设置坐标轴范围
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        fig.savefig(folder_path+'/'+f'qkm_idx_{ind:d}_{idx[i]:d}',
                    transparent=True, orientation='portrait',
                    bbox_inches='tight',pad_inches=0.1,dpi=600)

# ========================================================

def draw_contour_ppt(v1,v3,case,ind,error_type='Relative MSE',result='result_5_16'):
    '''
    Args:
        v1 (np.array): (t, h, h) GT
        v3 (np.array): (t, h, h) QKM
    '''
    config = PlotConfig2(nrow=3,ncol=5,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.9,       # 左侧边距 (cm)
                        margin_right= 1.1,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 1.0,        # 顶部边距 (cm)
                        space_width= 0.3,       # 子图水平间距 (cm)
                        space_height= [0.5,1.0],      # 子图垂直间距 (cm)
                        subplot_ratio= [0.9,0.9,0.121],     # 子图高宽比 (height/width)
                        ftsize=12,             # 基础字体大小
                    )
    config.set_row_config(row=2,ncols=1)
    fig,axes = config.get_multi()

    # 设置 坐标轴范围 ticks,labels
    extent = [0, 128, 0, 128]  # 设置 xmin, xmax, ymin, ymax
    ticks = [0,64,128]
    if case == 0:
        labels_x = ['0', '0.5', '1']
        labels_y = ['0', '0.25', '0.5']
    elif case == 1:
        labels_x = ['0', '$2\pi$', '$4\pi$']
        labels_y = ['0', '$2\pi$', '$4\pi$']
    elif case == 2:
        labels_x = ['-1', '0', '1']
        labels_y = ['-1', '0', '1']
    else:
        print("case error: please set case = 0,1,2")

    print(f'v1.min()={v1.min()},v1.max()={v1.max()}')
    if case == 0:
        vmin = -12.
        vmax = 12.
        tau = np.abs(v1[0]).max()
    elif case == 1:
        vmin = -10.
        vmax = 10.
        tau = np.abs(v1[0]).max()
    elif case == 2:
        vmin = 0.1
        vmax = 1.
        tau = 1/0.029

    # 获取时间步
    t_list = [0,15,30,45,60]

    for k in range(5) : # 5列
        # 绘制 GT 图像
        im = axes[0][k].imshow(
            getv(t_list[k], v1),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )

        axes[0][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[0][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[0][k].set_xticklabels(['','',''])
        axes[0][k].set_yticklabels(['','',''])

        # 绘制 QKM 图像
        axes[1][k].imshow(
            getv(t_list[k], v3),
            cmap='RdBu_r',
            animated=False,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # 设置坐标轴范围
        )
        axes[1][k].set_xticks(ticks)  # 设置 x 轴 ticklabel
        axes[1][k].set_yticks(ticks)  # 设置 y 轴 ticklabel
        axes[1][k].set_xticklabels(['','',''])
        axes[1][k].set_yticklabels(['','',''])

        if case==2:
            axes[0][k].set_title(fr"$t={t_list[k]*10/tau:.2f}\tau$",fontsize=config.ftsize,pad=2.0)
        elif case==1:
            axes[0][k].set_title(fr"$t={t_list[k]*0.1/tau:.2f}\tau$",fontsize=config.ftsize,pad=2.0)
        elif case==0:
            axes[0][k].set_title(fr"$t={t_list[k]*0.1/tau:.2f}\tau$",fontsize=config.ftsize,pad=2.0)

    axes[0][0].set_yticklabels(labels_y,fontsize=config.ftsize)
    axes[1][0].set_yticklabels(labels_y,fontsize=config.ftsize)

    if case==0:
        axes[0][0].set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
        axes[1][0].set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
    elif case==1:
        axes[0][0].set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
        axes[1][0].set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
    elif case==2:
        axes[0][0].set_ylabel('$y$',labelpad=1.5,fontsize=config.ftsize)
        axes[1][0].set_ylabel('$y$',labelpad=1.5,fontsize=config.ftsize)

    for j in range(5):
        axes[1][j].set_xticklabels(labels_x,fontsize=config.ftsize)
        axes[1][j].set_xlabel('$x$',labelpad=0,fontsize=config.ftsize)

    # color bar
    subwidth = axes[0][-1].get_position().x0 - axes[0][-2].get_position().x0 - axes[0][-2].get_position().width
    pos_up = axes[0][-1].get_position()
    pos_down = axes[1][-1].get_position()
    ax_bar = fig.add_axes([pos_down.x0+pos_up.width+subwidth*0.6,
                  pos_down.y0+pos_up.height/4,
                  subwidth*0.4,
                  pos_up.y0-pos_down.y0+pos_up.height/2,])
    cbar = fig.colorbar(im,cax=ax_bar,orientation='vertical')
    cbar.ax.tick_params(
        direction='in',  # 刻度线朝外
        length=2.,         # 刻度线长度（默认是 4，增大后会更凸出）
        width=0.7,          # 刻度线宽度
        colors='black',   # 刻度线颜色
        pad=1.5,          # 刻度标签与刻度线的距离
        top=True,
        bottom=False
    )

    if case == 0:
        bar_label='$\omega$'
        bar_ticks=[-10,-5,0,5,10]
        bar_tls=[f'{x:d}' for x in bar_ticks]
    elif case == 1:
        bar_label='$\omega$'
        bar_ticks=[-8,-4,0,4,8]
        bar_tls=[f'{x:d}' for x in bar_ticks]
    elif case == 2:
        bar_label='$Y_A$'
        bar_ticks=[0.1,0.3,0.5,0.7,0.9]
        bar_tls=[f'{x:.1f}' for x in bar_ticks]

    cbar.set_ticks(bar_ticks)
    cbar.set_ticklabels(bar_tls,fontsize=config.ftsize-1)


    cbar.set_label('') # 移除默认的 label（如果有）
    # 手动添加 label
    cbar.ax.text(
        0.5,                    # x 位置（0=左，1=右，0.5=居中）
        1.02,                   # y 位置
        bar_label,              # 标签文本
        ha='center',            # 水平居中
        va='bottom',            # 垂直对齐（底部）
        fontsize=config.ftsize,      # 字体大小
        transform=cbar.ax.transAxes  # 使用相对坐标（0-1）
    )

    # 绘图error
    ax = axes[2][0]
    pos = ax.get_position()
    pos_right = axes[1][-1].get_position()
    ax.remove()
    ax = fig.add_axes([axes[1][0].get_position().x0,
                  pos.y0,
                  pos_right.x0+pos_right.width-axes[1][0].get_position().x0,
                  pos.height,])

    # mae = ( np.abs(v3-v1) ).mean(axis=(-2,-1))
    mse = ( (v3-v1)**2 ).mean(axis=(-2,-1))
    # rmse = np.sqrt(mse)
    error_dict={
        # 'MAE': mae,
        # 'MSE': mse,
        # 'RMSE': rmse,
        'Relative MSE': mse / (v1**2).mean(axis=(-2,-1)),
        # 'Relative MAE': mae / ( np.abs(v1) ).mean(axis=(-2,-1)),
    }
    error = error_dict[error_type]

    rmse = np.load(f'RMSE_case={case:d}_{result:s}.npy')
    rmse_low=np.zeros(rmse.shape[1])
    rmse_high=np.zeros(rmse.shape[1])
    for i in range(rmse.shape[1]):
        data = rmse[:,i]
        rmse_low[i] = np.percentile(data, 10)  # 10%分位数
        rmse_high[i] = np.percentile(data, 90)  # 90%分位数

    # ax.fill_between(np.arange(rmse.shape[1]),rmse_low,rmse_high,
    #                 alpha=0.4,edgecolor='none',color='#e8d5d5')
    # ax.plot(error,linestyle='-',color='#c79494',linewidth=1.5)
    ax.fill_between(np.arange(rmse.shape[1]),rmse_low,rmse_high,
                    alpha=0.25,edgecolor='none',color='#A8B5C2')
    ax.plot(error,linestyle='-',color='#A6cced',linewidth=1.5)

    if case==0:
        ax.set_ylim(0, 0.05)
        ax.set_yticks([0,0.02,0.04])
        ax.set_yticklabels([0,2,4],fontsize=config.ftsize)
    elif case==1:
        ax.set_ylim(0, 1.4)
        ax.set_yticks([0,0.6,1.2])
        ax.set_yticklabels([0,60,120],fontsize=config.ftsize)
    elif case==2:
        ax.set_ylim(0, 0.07)
        ax.set_yticks([0,0.03,0.06])
        ax.set_yticklabels([0,3,6],fontsize=config.ftsize)

    ax.grid(True,color='#EBEBEB',linewidth=0.5,alpha=0.5,linestyle='-',zorder=0)
    ax.set_axisbelow(True)  # 网格线和坐标轴置于底层
    ax.set_xlim([-1,61])
    xticks=[0,10,20,30,40,50,60]
    ax.set_xticks(xticks)
    if case==2:
        ax.set_xticklabels([fr'{i*10/tau:.1f}$\tau$' for i in xticks])
    else:
        ax.set_xticklabels([fr'{i*0.1/tau:.1f}$\tau$' for i in xticks])

    ax.tick_params(axis='y', which='major',labelsize=config.ftsize-1,
            top=False, right=False, length=3, pad=3.)
    ax.tick_params(axis='x', which='major',labelsize=config.ftsize-1,
            top=False, right=False, length=3, pad=4.5)
    ax.tick_params(which='minor',top=False, right=False, length=1.5)
    ax.set_xlabel('$t$',labelpad=0,fontsize=config.ftsize)

    pos0 = axes[0][0].get_position()
    pos1 = axes[1][0].get_position()
    pos3 = ax.get_position()

    if case==0:
        fig.text(pos0.x0-0.0925,pos0.y0+pos0.height/2, 'GT', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.0925,pos1.y0+pos1.height/2, 'QKM', ha='center', va='center', rotation=90,  fontsize=config.ftsize)
        fig.text(pos0.x0-0.0925,pos3.y0+pos3.height/2, 'RMSE (\%)', ha='center', va='center', rotation=90,  fontsize=config.ftsize)

    else:
        fig.text(pos0.x0-0.07,pos0.y0+pos0.height/2, 'GT', ha='center', va='center', rotation=90, fontsize=config.ftsize)
        fig.text(pos0.x0-0.07,pos1.y0+pos1.height/2, 'QKM', ha='center', va='center', rotation=90,  fontsize=config.ftsize)
        fig.text(pos0.x0-0.07,pos3.y0+pos3.height/2, 'RMSE (\%)', ha='center', va='center', rotation=90,  fontsize=config.ftsize)


    for i in range(2):
        for j in range(5):
            axes[i][j].tick_params(axis='y', which='major',labelsize=config.ftsize-1,
                    top=False, right=False, length=2., width=0.7,pad=1.5)
            axes[i][j].tick_params(axis='x', which='major',labelsize=config.ftsize-1,
                    top=False, right=False, length=2., width=0.7,pad=3.5)
            axes[i][j].tick_params(which='minor',top=False, right=False, length=1.,width=0.7,)

    fig.savefig(fr'case={case:d}_ind={ind:d}_ppt_contour.png', transparent=True, orientation='portrait',
                bbox_inches='tight',dpi=600) # bbox_inches='tight',

# ========================================================

def draw_video_ppt(v1,v3,case,ind):
    '''
    Args:
        v1 (np.array): (t, h, h) GT
        v3 (np.array): (t, h, h) QKM
    '''
    config = PlotConfig3(nrow=1,ncol=2,
                        plot_width= 16.5,       # 总画布宽度 (cm)
                        margin_left= 1.9,       # 左侧边距 (cm)
                        margin_right= 1.1,      # 右侧边距 (cm)
                        margin_bottom= 1.2,     # 底部边距 (cm)
                        margin_top= 1.5,        # 顶部边距 (cm)
                        space_width= 1.5,       # 子图水平间距 (cm)
                        space_height= 0.5,      # 子图垂直间距 (cm)
                        subplot_ratio= 0.9,     # 子图高宽比 (height/width)
                        ftsize=12,             # 基础字体大小
                    )
    fig,[(ax1,ax2)] = config.get_multi()

    # 设置 坐标轴范围 ticks,labels
    extent = [0, 128, 0, 128]  # 设置 xmin, xmax, ymin, ymax
    ticks = [0,64,128]
    if case == 0:
        labels_x = ['0', '0.5', '1']
        labels_y = ['0', '0.25', '0.5']
    elif case == 1:
        labels_x = ['0', '$2\pi$', '$4\pi$']
        labels_y = ['0', '$2\pi$', '$4\pi$']
    elif case == 2:
        labels_x = ['-1', '0', '1']
        labels_y = ['-1', '0', '1']
    else:
        print("case error: please set case = 0,1,2")

    for ax in [ax1,ax2]:
        ax.tick_params(axis='y', which='major',labelsize=config.ftsize-1,
                top=False, right=False, length=3, pad=3.)
        ax.tick_params(axis='x', which='major',labelsize=config.ftsize-1,
                top=False, right=False, length=3, pad=4.5)
        ax.tick_params(which='minor',top=False, right=False, length=1.5)
        ax.set_xticks(ticks)  # 设置 x 轴 ticklabel
        ax.set_yticks(ticks)  # 设置 y 轴 ticklabel
        ax.set_xticklabels(labels_x,fontsize=config.ftsize)
        ax.set_yticklabels(labels_y,fontsize=config.ftsize)
        ax.set_xlabel('$x$',labelpad=0,fontsize=config.ftsize)
        if case==0:
            ax.set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
            ax.set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
        elif case==1:
            ax.set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
            ax.set_ylabel('$y$',labelpad=3.5,fontsize=config.ftsize)
        elif case==2:
            ax.set_ylabel('$y$',labelpad=1.5,fontsize=config.ftsize)
            ax.set_ylabel('$y$',labelpad=1.5,fontsize=config.ftsize)

    images = list(np.arange(v1.shape[0]))
    getv = lambda key, v: v[key, :, :]
    # 初始化 ax1 的图像和文本
    im1 = ax1.imshow(getv(images[0], v1), cmap='RdBu_r', animated=True, extent=extent,
                     vmin=np.min([getv(img, v1) for img in images]),
                     vmax=np.max([getv(img, v1) for img in images]))
    frame_text1 = ax1.text(0.0, 1.15, f"frame: 1/{len(images)}", transform=ax1.transAxes, ha='center')
    ax1.set_title("GT")
    # 初始化 ax2 的图像
    im2 = ax2.imshow(getv(images[0], v3), cmap='RdBu_r', animated=True, extent=extent,
                     vmin=np.min([getv(img, v1) for img in images]),
                     vmax=np.max([getv(img, v1) for img in images]))
    ax2.set_title("QKM")  # 设置 ax2 的标题
    # 更新函数
    def update(frame):
        # 更新 ax1 的图像和文本
        im1.set_array(getv(images[frame], v1))
        frame_text1.set_text(f"frame: {frame+1}/{len(images)}")
        # 更新 ax2 的图像
        im2.set_array(getv(images[frame], v3))
        return [im1, frame_text1, im2]  # 返回所有更新的对象

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(images), blit=False)  # 设置 blit=False
    ani.save(
        fr'case={case:d}_ind={ind:d}_ppt_video.mp4',
        writer='ffmpeg',
        fps=25,bitrate=5000,
        extra_args=['-preset', 'slow', '-crf', '18'],  # FFmpeg优化参数
    )