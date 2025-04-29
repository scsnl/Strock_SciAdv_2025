import numpy as np
from scipy.stats import pearsonr, ttest_ind, ttest_1samp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.patheffects as PathEffects
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.legend import Legend
from itertools import combinations, product
from .prompt import print_title

scale_font = 10/7.25
plt.rc('font', size=8*scale_font)
plt.rc('axes', titlesize=8*scale_font)
plt.rc('axes', labelsize=8*scale_font)
plt.rc('xtick', labelsize=8*scale_font)
plt.rc('ytick', labelsize=8*scale_font)
plt.rc('legend', fontsize=7.5*scale_font)
plt.rc('figure', dpi=1200)

class TextHandler(HandlerBase):
    def create_artists(self, legend, text ,xdescent, ydescent,
                        width, height, fontsize, trans):
        tx = Text(width/2.,height/2, text, fontsize=fontsize,
                  ha="center", va="center", fontweight="bold")
        return [tx]

Legend.update_default_handler_map({str : TextHandler()})

phi = (1+np.sqrt(5))/2
def letter(letter, plot = lambda f, gs: f.add_subplot(gs), delta = 0.09, k=9/7.25, printletter = False): # fontsize = 9, printsize = 7.25, k = fonsize/printsize
    def wrapper(f, gs, *args, **kwargs):
        if printletter:
            print_title(letter)
        ax = plot(f, gs, *args, **kwargs)
        w,h = f.get_figwidth(), f.get_figheight()
        r = w/h
        coord = f.transFigure.inverted().transform(ax.transAxes.transform((0,1)))
        t = ax.text(coord[0]-delta*w/10, coord[1]+r*delta/phi*w/10, letter, fontsize=k*w, fontweight = 'bold', transform = f.transFigure, ha = 'left', va = 'top')
        return ax
    return wrapper

def get_figsize(ws, wspace, hs, hspace, zoom, left = 2, right = 4, top = 1.5, bottom = 2):
    _ws = np.zeros(2*len(ws)+1)+wspace
    _ws[1::2] = ws
    _ws[[0]] /= left
    _ws[[-1]] /= right
    _sum_ws = np.sum(_ws) 
    _ws /= _sum_ws
    _hs = np.zeros(2*len(hs)+1)+hspace
    _hs[1::2] = hs
    _hs[[0]] /= top
    _hs[[-1]] /= bottom
    _hs /= _sum_ws
    figsize = zoom, zoom*np.sum(_hs)/np.sum(_ws)
    return figsize, _ws, _hs

def get_ax(f, gs, title = '', xlabel = '', ylabel = '', xlim = None, ylim = None, xticks = None, yticks = None, axshow = 'half'):
    ax = f.add_subplot(gs)
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)
    if not xticks is None:
        ax.set_xticks(xticks)
    if not yticks is None:
        ax.set_yticks(yticks)
    if axshow == 'half':
        ax.spines[['right', 'top']].set_visible(False)
    elif axshow == 'none':
        ax.axis('off')
    return ax

def plot(f, gs, x, y, xlabel = '', ylabel = '', xlim = None, ylim = None, loc = 'lower right', title = '', correlate = False, xticks = None, yticks = None):
    ax = get_ax(f, gs, title, xlabel, ylabel, xlim, ylim, xticks, yticks)
    ax.plot(x, y, color = '0.0', marker = '.', clip_on = False)
    if correlate:
        a,b = np.polyfit(x, y, 1)
        r,p = pearsonr(x, y)
        _xlabel = xlabel.replace('\n', ' ')
        _ylabel = ylabel.replace('\n', ' ')
        print(f'Correlation {_xlabel} vs {_ylabel}: (r = {r:.3f}, p = {p:.2e})')
        ax.plot(x, a*x+b, color = 'C3', linestyle = '--')
        theta = ax.transData.transform_angles(np.array([180*np.arctan(a)/np.pi]), np.array([[np.mean(x), a*np.mean(x)+b]]))[0]
        if loc == 'lower right':
            ax.text(0.95, 0.1, f'$r = {r:.2f}$', color = 'C3', va = 'bottom', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        elif loc == 'lower left':
            ax.text(0.05, 0.1, f'$r = {r:.2f}$', color = 'C3', va = 'bottom', ha = 'left', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        elif loc == 'upper right':
            ax.text(0.95, 0.9, f'$r = {r:.2f}$', color = 'C3', va = 'top', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        elif loc == 'upper left':
            ax.text(0.05, 0.9, f'$r = {r:.2f}$', color = 'C3', va = 'top', ha = 'left', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        if p <= 0.05:
            if p<=0.001:
                s = '***'
            elif p<=0.01:
                s = '**'
            else:
                s = '*'
            f.draw_without_rendering()
            ax.text(np.mean(x), a*np.mean(x)+b, s, va = 'bottom', ha = 'center', size = 'large', color = 'C3', rotation = theta, rotation_mode='anchor', path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
    return ax

# Clean below

def plot_bar(f, gs, bars, steps, c, ylabel = '', sharey = None, best_step = None, ylim = None, label = ''):
    ax = f.add_subplot(gs, sharey = sharey)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(f'{ylabel}')
    ax.set_xlabel('iteration')
    for i, bar in enumerate(bars):
        ax.fill_between(steps, bar[0], bar[2], color = c[i], alpha = 0.3, zorder = 2*i, label = label[i])
        ax.plot(steps, bar[1], color = c[i], zorder = 2*i+1)
    if not best_step is None:
        ax.axvline(best_step, 0, 1, color = '0.0', linestyle = '--', zorder = 2*len(bars))
        ax.text(best_step, 1.1, f'Best iteration = {best_step}', color = '0.0', transform = transforms.blended_transform_factory(ax.transData, ax.transAxes), ha = 'center', va = 'center')
    if not ylim is None:
        ax.set_ylim(ylim)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol = 3, loc = 'lower center', bbox_to_anchor = (0.5, 1.2))
    return ax

def violinplot(f, gs, ys, names, c = None, xlabel = '', ylabel = '', ylim = None, order = 1, sort = False, pd = 'less', printstats = [], title = '', showsign = True):
    n = len(names)
    if pd == 'all':
        pd = [(i,j) for i in range(n) for j in range(i+1, n)]
    if pd == 'less':
        if order == 1:
            pd = [(i+1,i) for i in range(n-1)]
        else:
            pd = [(i+1,i) for i in range(n-1)][::-1]
    ts, ps = np.zeros((n,n)), np.zeros((n,n))
    ms, stds = np.zeros(n), np.zeros(n)
    _ylabel = ylabel.replace('\n', ' ')
    for i, y1 in enumerate(ys):
        _namei = names[i].replace('\n', ' ')
        ms[i], stds[i] = np.mean(y1), np.std(y1)
        print(f'{_ylabel} {_namei}: (M = {ms[i]:.3f}, SD = {stds[i]:.3f})')
        for j, y2 in enumerate(ys):
            ts[i,j], ps[i,j] = ttest_ind(y1, y2, equal_var = False, nan_policy = 'propagate')
    idx = np.isnan(ts)
    
    if np.sum(idx)>0:
        ps[idx] = 1.0
        idx = np.where(idx)
        s = ', '.join([f'({names[i]},{names[j]})' for i,j in zip(idx[0], idx[1]) if i < j])
        print(f'Comparing two constants: {s}')
    if sort:
        if order == 1:
            idx = np.argsort(np.sum(ts > 0, axis = 1))
        else:
            idx = np.argsort(np.sum(ts < 0, axis = 1))
    else:
        idx = np.arange(n)
        
    cmap = cm.get_cmap("viridis")
    norm  = Normalize(vmin=0, vmax=n-1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    if not ylim is None:
        ax.set_ylim(ylim)
    for k, i in enumerate(idx):
        y = ys[i]
        q1, median, q3 = np.percentile(y, [25, 50, 75])
        #mean = np.mean(y)
        parts = ax.violinplot(y, [k], widths = 0.5, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(c[i])
            pc.set_edgecolor(c[i])
            pc.set_clip_on(False)
        ax.vlines(k, np.clip(q1 - (q3 - q1) * 1.5, np.min(y), q1), np.clip(q3 + (q3 - q1) * 1.5, q3, np.max(y)), lw = 1, color = c[i], zorder = 1, clip_on = False)
        ax.vlines(k, q1, q3, color = c[i], lw = 10, zorder = 2, clip_on = False)
        ax.vlines(k, median-0.01*(ylim[1]-ylim[0]), median+0.01*(ylim[1]-ylim[0]), color = '1.0', lw = 10, zorder = 2, clip_on = False)
        _,count = np.unique(y, return_counts = True)
        if np.all(count == np.array([3,2,11,5])):
            print(ylabel, names[idx[k]], np.unique(y, return_counts = True))
        #ax.vlines(k, mean-0.01*(ylim[1]-ylim[0]), mean+0.01*(ylim[1]-ylim[0]), color = 'red', lw = 10, zorder = 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0,n))
    ax.set_xticklabels([names[i] for i in idx])
    y_min, y_max = np.min([np.min(y) for y in ys]), np.max([np.max(y) for y in ys])
    y_range = y_max-y_min
    _ylabel = ylabel.replace('\n', ' ')
    _y_max = -np.inf
    signshown = False
    for k, sig in enumerate(pd):
        if type(sig) is tuple:
            if len(sig) == 2:
                i,j = sig
                t = ts[idx[i],idx[j]], ts[idx[i],idx[j]]
                p = ps[idx[i],idx[j]], ps[idx[i],idx[j]]
                sign = t[0]>0
                _namei = names[idx[i]].replace('\n', ' ')
                _namej = names[idx[j]].replace('\n', ' ')
                print(f'{_ylabel} {_namei} vs {_namej}: (t = {t[0]:.3f}, p = {p[0]:.2e})')
                y_min, y_max = min(np.min(ys[i]),np.min(ys[j])), max(np.max(ys[i]),np.max(ys[j]))
                i,j = np.min(sig)-0.125, np.max(sig)+0.125
            else:
                assert len(sig) > 2
                _t = [ts[idx[i], idx[j]] for i,j in combinations(sig, 2)]
                _p = [ps[idx[i], idx[j]] for i,j in combinations(sig, 2)]
                t = np.min(np.abs(_t)), np.max(np.abs(_t))
                p = np.min(_p), np.max(_p)
                sign = _t[np.argmin(np.abs(_t))]>0
                i, j = np.min(sig)-0.125, np.max(sig)+0.125
                _names = ' vs '.join([names[idx[i]].replace('\n', ' ') for i in sig])
                y_min, y_max = np.min([[np.min(ys[idx[i]]),np.min(ys[idx[j]])] for i,j in combinations(sig, 2)]), np.max([[np.max(ys[idx[i]]),np.max(ys[idx[j]])] for i,j in combinations(sig, 2)])
                print(f'{_ylabel} {_names}: (|t| ∈ [{t[0]:.3f}, {t[1]:.3f}], p ∈ [{p[0]:.2e}, {p[1]:.2e}])')
        elif type(sig) == list:
            _t = [ts[idx[i], idx[j]] for i,j in sig]
            _p = [ps[idx[i], idx[j]] for i,j in sig]
            t = np.min(np.abs(_t)), np.max(np.abs(_t))
            p = np.min(_p), np.max(_p)
            sign = _t[np.argmin(np.abs(_t))]>0
            i, j = np.mean([i for i,j in sig]), np.mean([j for i,j in sig])
            y_min, y_max = np.min([[np.min(ys[idx[i]]),np.min(ys[idx[j]])] for i,j in sig]), np.max([[np.max(ys[idx[i]]),np.max(ys[idx[j]])] for i,j in sig])
            _names = [name.replace('\n', ' ') for name in names]
            _names = ' and '.join([f'({_names[idx[i]]} vs {_names[idx[j]]})' for i,j in sig])
            print(f'{_ylabel} {_names}: (|t| ∈ [{t[0]:.3f}, {t[1]:.3f}], p ∈ [{p[0]:.2e}, {p[1]:.2e}])')
        if y_max + 0.1*(ylim[1]-ylim[0]) > _y_max + (0.25 if signshown else 0.15)*(ylim[1]-ylim[0]):
            _y_max = y_max + 0.1*(ylim[1]-ylim[0])
        else:
            _y_max = _y_max + (0.25 if signshown else 0.15)*(ylim[1]-ylim[0])
        ax.hlines(_y_max, (i-0.1), (j+0.1), color = '0.0', clip_on = False)
        if p[1] <= 0.001:
            s = '***'
        elif p[1] <= 0.01:
            s = '**'
        elif p[1] <= 0.05:
            s = '*'
        elif p[0] > 0.05:
            s = 'n.s.'
        else:
            s = 'mixed'
        if p[1] <= 0.05 and showsign:
            ax.text((i+j)/2, _y_max, f'{">" if sign else "<"}\n{s}', va = 'bottom', ha = 'center')
            signshown = True
        else:
            ax.text((i+j)/2, _y_max, s, va = 'bottom', ha = 'center')
            signshown = False
    for i,j in printstats:
        t = ts[idx[i],idx[j]], ts[idx[i],idx[j]]
        p = ps[idx[i],idx[j]], ps[idx[i],idx[j]]
        _namei = names[idx[i]].replace('\n', ' ')
        _namej = names[idx[j]].replace('\n', ' ')
        print(f'{_ylabel} {_namei} vs {_namej}: (t = {t[0]:.3f}, p = {p[0]:.2e})')
    return ax

def plot_value_by_iteration(f, gs, value, es, steps, ylabel, clabel, cticks = None, xlim = None, ylim = None, title = '', sharey = None, threshold = None, chance = None, showbar = True, xlabel = 'iteration'):
    cmap = cm.get_cmap("viridis")
    norm  = Normalize(vmin=np.min(es), vmax=np.max(es))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax = f.add_subplot(gs, sharey = sharey)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    for i, e in enumerate(es):
        ax.plot(steps, value[i], label = f'{e:.1f}', c = sm.to_rgba(e), zorder = i, marker = '.', clip_on = False)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'{ylabel}')
    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)
    #ax.fill_between(np.arange(1,steps[-1]+1), 0, 1, where = (np.arange(0,steps[-1])//190)%2 == 1, color = '0.9', zorder = -1, transform = trans)
    if not threshold is None:
        ax.axhline(threshold, 0, 1, color = 'C3', linestyle = '--', zorder = len(es))
        ax.text(steps[0], threshold, 'required level', va = 'bottom', ha = 'left', color = 'C3')
    if not chance is None:
        ax.axhline(chance, 0, 1, color = '0.5', linestyle = '--', zorder = len(es))
        ax.text(steps[-1], chance, 'chance level', va = 'bottom', ha = 'right', color = '0.5')
    if showbar:
        cbar = plt.colorbar(sm, cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0]), label = clabel, rasterized = True)
        if not cticks is None:
            cbar.ax.set_yticks(cticks)
    return ax

def plot_value_by_gain(f, gs, value, es, steps, ylabel, clabel, cticks = None, xlim = None, ylim = None, title = '', sharey = None, threshold = None, chance = None, showbar = True):
    cmap = cm.get_cmap("plasma")
    norm  = Normalize(vmin=np.min(steps), vmax=np.max(steps))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax = f.add_subplot(gs, sharey = sharey)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    for i, step in enumerate(steps):
        ax.plot(es, value[:,i], label = f'{step:.1f}', c = sm.to_rgba(step), zorder = i, marker = '.', clip_on = False)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xlabel('gain $G$')
    ax.set_ylabel(f'{ylabel}')
    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)
    #ax.fill_between(np.arange(1,steps[-1]+1), 0, 1, where = (np.arange(0,steps[-1])//190)%2 == 1, color = '0.9', zorder = -1, transform = trans)
    if not threshold is None:
        ax.axhline(threshold, 0, 1, color = 'C3', linestyle = '--', zorder = len(es))
        ax.text(steps[0], threshold, 'required level', va = 'bottom', ha = 'left', color = 'C3')
    if not chance is None:
        ax.axhline(chance, 0, 1, color = '0.5', linestyle = '--', zorder = len(es))
        ax.text(steps[-1], chance, 'chance level', va = 'bottom', ha = 'right', color = '0.5')
    if showbar:
        cbar = plt.colorbar(sm, cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0]), label = clabel)
        if not cticks is None:
            cbar.ax.set_yticks(cticks)
    return ax

def plot_r_by_iteration(f, gs, rs, ps, steps, legend = False, title = ''):
    ax = f.add_subplot(gs)
    ax.set_title(title, fontweight = 'bold')
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(steps, -1, 1, where = ps <= 0.05, color = '0.9', label = '$p < 0.05$', transform = trans)
    ax.fill_between(steps, -1, 1, where = ps <= 0.01, color = '0.85', label = '$p < 0.01$', transform = trans)
    ax.fill_between(steps, -1, 1, where = ps <= 0.001, color = '0.8', label = '$p < 0.001$', transform = trans)
    ax.axhline(0, 0, 1, color = '0.7')
    ax.plot(steps, rs, color = '0.0', marker = '.')
    ax.set_xlabel('iteration')
    ax.set_ylabel(f'correlation $r$')
    if legend:
        ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.1), ncol = 3, columnspacing = 1.0, fancybox = False)
    return ax

def violinplot_by_gain(f,gs,es,value,xlabel='',ylabel='', title = '', sharey = None, ylim = None):
    cmap = cm.get_cmap("viridis")
    norm  = Normalize(vmin=np.min(es), vmax=np.max(es))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax = f.add_subplot(gs, sharey = sharey)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    w = (es[1]-es[0])/2
    q1, m, q3 = np.percentile(value, [25, 50, 75], axis = 1)
    t = np.empty((len(es), len(es)))
    p = np.empty((len(es), len(es)))
    for i, e in enumerate(es):
        for j, e in enumerate(es):
            t[i,j], p[i,j] = ttest_ind(value[i].flatten(), value[j].flatten())
    for i, e in enumerate(es):
        parts = ax.violinplot(value[i], [e], widths = w, showmeans=False, showmedians=False, showextrema=False)
        ax.vlines(e, np.clip(q1[i] - (q3[i] - q1[i]) * 1.5, np.min(value[i]), q1[i]), np.clip(q3[i] + (q3[i] - q1[i]) * 1.5, q3[i], np.max(value[i])), lw = 1, color = sm.to_rgba(e), zorder = 1)
        ax.vlines(e, q1[i], q3[i], color = sm.to_rgba(e), lw = 3, zorder = 2)
        ax.scatter(e, m[i], color = '1.0', zorder = 3, s = 10, marker = '_', edgecolors='none')
        for pc in parts['bodies']:
            pc.set_facecolor(sm.to_rgba(e))
            pc.set_edgecolor(sm.to_rgba(e))
    e_m = (es[0]+es[-1])/2
    a,b = np.polyfit(np.tile(es, (value.shape[1],1)).T.flatten(), value.flatten(), 1)
    r,p = pearsonr(np.tile(es, (value.shape[1],1)).T.flatten(), value.flatten())
    _title = title.replace("\n", " ")
    _ylabel = ylabel.replace("\n", " ")
    print(f'Correlation gain vs {_ylabel} ({_title}): {r:.2f}, {p:.2e}')
    theta = ax.transData.transform_angles(np.array([180*np.arctan(a)/np.pi]), np.array([[e_m, a*e_m+b]]))[0]
    ax.plot(es, a*es+b, color = 'C3', linestyle = '--')
    ax.text(0.95, 0.1, f'r = {r:.2f}', color = 'C3', va = 'bottom', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
    if p <= 0.05:
        if p<=0.001:
            s = '***'
        elif p<=0.01:
            s = '**'
        else:
            s = '*'
        ax.text(e_m, a*e_m+b, s, va = 'bottom', ha = 'center', color = 'C3', rotation = theta, rotation_mode='anchor', path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
    ax.set_xlabel('gain $G$')
    ax.set_ylabel(ylabel)
    if not ylim is None:
        ax.set_ylim(ylim)
    return ax

def plot_by_layer(f, gs, x, y, xlabel = '', ylabel = '', xlim = None, ylim = None, loc = 'lower right', title = [''], correlate = False, xticks = None, yticks = None):
    ax = f.add_subplot(gs)
    ax.axis('off')
    _gs = gs.subgridspec(1,len(y), wspace = 0.5)
    for i in range(len(y)):
        if i == 0:
            plot(f, _gs[0,i], x, y[i], xlabel, ylabel, xlim, ylim, loc, title[i], correlate, xticks, yticks)
        else:
            plot(f, _gs[0,i], x, y[i], xlabel, '', xlim, ylim, loc, title[i], correlate, xticks, yticks)
    return ax

def multiplot_by_layer(f, gs, x, y, es, xlabel = '', ylabel = '', xlim = None, ylim = None, loc = 'lower right', title = '', xticklabels = None, showbar = False, clabel = '', cticks = None):
    cmap = cm.get_cmap("viridis")
    norm  = Normalize(vmin=np.min(es), vmax=np.max(es))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not ylim is None:
        ax.set_ylim(ylim)
    if not xlim is None:
        ax.set_xlim(xlim)
    if not xticklabels is None:
        ax.set_xticks(np.arange(x.shape[0]))
        ax.set_xticklabels(xticklabels)
    _xlabel = xlabel.replace('\n', ' ')
    _ylabel = ylabel.replace('\n', ' ')
    for i in range(y.shape[0]):
        ax.plot(x, y[i], color = sm.to_rgba(es[i]), marker = '.')
        try:
            a,b = np.polyfit(x, y[i], 1)
            r,p = pearsonr(x, y[i])
            print(f'Correlation {_xlabel} vs {_ylabel} ({clabel}={es[i]}): (r = {r:.3f}, p={p:.2e})')
        except:
            print(f'Correlation {_xlabel} vs {_ylabel} ({clabel}={es[i]}): Error')
    if showbar:
        cbar = plt.colorbar(sm, cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0]), label = clabel)
        if not cticks is None:
            cbar.ax.set_yticks(cticks)
    return ax

def plot_bar_by_gain(f, gs, x, y, xlabel = '', ylabel = '', xlim = None, ylim = None, loc = 'lower right', title = '', correlate = False, xticks = None):
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not ylim is None:
        ax.set_ylim(ylim)
    if not xlim is None:
        ax.set_xlim(xlim)
    if not xticks is None:
        ax.set_xticks(xticks)
    ax.fill_between(x, y[0], y[2], color = '0.0', alpha = 0.1)
    ax.plot(x, y[1], color = '0.0', marker = '.')
    if correlate:
        _xlabel = xlabel.replace('\n', ' ')
        _ylabel = ylabel.replace('\n', ' ')
        ext = f' ({title})' if len(title)>0 else ''
        try:
            a,b = np.polyfit(x, y[1], 1)
            r,p = pearsonr(x, y[1])
            print(f'Correlation {_xlabel} vs {_ylabel}{ext}: (r = {r:.3f}, p = {p:.2e})')
            ax.plot(x, a*x+b, color = 'C3', linestyle = '--')
            theta = ax.transData.transform_angles(np.array([180*np.arctan(a)/np.pi]), np.array([[np.mean(x), a*np.mean(x)+b]]))[0]
            if loc == 'lower right':
                ax.text(0.95, 0.1, f'$r = {r:.2f}$', color = 'C3', va = 'bottom', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
            elif loc == 'lower left':
                ax.text(0.05, 0.1, f'$r = {r:.2f}$', color = 'C3', va = 'bottom', ha = 'left', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
            elif loc == 'upper right':
                ax.text(0.95, 0.9, f'$r = {r:.2f}$', color = 'C3', va = 'top', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
            elif loc == 'upper left':
                ax.text(0.05, 0.9, f'$r = {r:.2f}$', color = 'C3', va = 'top', ha = 'left', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
            if p <= 0.05:
                if p<=0.001:
                    s = '***'
                elif p<=0.01:
                    s = '**'
                else:
                    s = '*'
                ax.text(np.mean(x), a*np.mean(x)+b, s, va = 'bottom', ha = 'center', color = 'C3', rotation = theta, rotation_mode='anchor', path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        except:
            print(f'Correlation {_xlabel} vs {_ylabel}{ext}: Error')
    return ax

def barplot_by_layer(f, gs, x, y, xlabel = 'layer', ylabel = '', xlim = None, ylim = None, title = '', c = None, label = None):
    _x = np.arange(len(x))
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not ylim is None:
        ax.set_ylim(ylim)
    if not xlim is None:
        ax.set_xlim(xlim)
    ax.set_xticks(_x)
    ax.set_xticklabels(x)
    for i in range(len(y)):
        _label = None if label is None else label[i]
        ax.fill_between(_x, y[i][0], y[i][2], color = c[i], alpha = 0.3, label = _label)
        ax.plot(_x, y[i][1], color = c[i], marker = '.')
    if not label is None:
        ax.legend(ncol = 2, loc = 'lower center', bbox_to_anchor = (0.5, 1.2))
    return ax

def scatter(f, gs, x, y, xlabel = '', ylabel = '', xlim = None, ylim = None, loc = 'lower right', title = '', correlate = False):
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not ylim is None:
        ax.set_ylim(ylim)
    if not xlim is None:
        ax.set_xlim(xlim)
    ax.scatter(x, y, color = '0.0', marker = '.')
    if correlate:
        a,b = np.polyfit(x, y, 1)
        r,p = pearsonr(x, y)
        _xlabel = xlabel.replace('\n', ' ')
        _ylabel = ylabel.replace('\n', ' ')
        print(f'Correlation {_xlabel} vs {_ylabel}: (r = {r:.3f}, p = {p:.2e})')
        _x = np.array([np.min(x), np.max(x)])
        ax.plot(_x, a*_x+b, color = 'C3', linestyle = '--')
        theta = ax.transData.transform_angles(np.array([180*np.arctan(a)/np.pi]), np.array([[np.mean(x), a*np.mean(x)+b]]))[0]
        if loc == 'lower right':
            ax.text(0.95, 0.1, f'$r = {r:.2f}$', color = 'C3', va = 'bottom', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        elif loc == 'lower left':
            ax.text(0.05, 0.1, f'$r = {r:.2f}$', color = 'C3', va = 'bottom', ha = 'left', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        elif loc == 'upper right':
            ax.text(0.95, 0.9, f'$r = {r:.2f}$', color = 'C3', va = 'top', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        elif loc == 'upper left':
            ax.text(0.05, 0.9, f'$r = {r:.2f}$', color = 'C3', va = 'top', ha = 'left', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
        if p <= 0.05:
            if p<=0.001:
                s = '***'
            elif p<=0.01:
                s = '**'
            else:
                s = '*'
            f.draw_without_rendering()
            ax.text(np.mean(x), a*np.mean(x)+b, s, va = 'bottom', ha = 'center', size = 'large', color = 'C3', rotation = theta, rotation_mode='anchor', path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
    return ax

def hist(f, gs, x, x0, c = '0.7', xlabel = '', ylabel = 'count', title = ''):
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontweight = 'bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.hist(x, bins = 20, color = c)
    ax.axvline(x0, color = 'C3', linestyle = '--')
    ax.text(x0, 1.1, f'{xlabel} = {x0:.2f}', color = 'C3', transform = transforms.blended_transform_factory(ax.transData, ax.transAxes), ha = 'center', va = 'center')
    return ax

def imshow_by_gain(f, gs, x, y, xlabel = '', ylabel = '', ticks = None, clabel = ''):
    cmap = cm.get_cmap("Oranges")
    norm  = Normalize(vmin=np.min(x), vmax=np.max(x))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    _gs = gs.subgridspec(1,len(y))
    for i in range(len(y)):
        ax = f.add_subplot(_gs[i])
        ax.set_title(f'$G$ = {y[i]:.1f}')
        ax.imshow(x[i].T, cmap = cmap, norm = norm, rasterized = True)
        ax.set_xlabel(xlabel)
        if not ticks is None:
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])
        if i == len(y)-1:
            _ax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
            cbar = plt.colorbar(sm, cax = _ax, label = clabel)
            _ax.ticklabel_format(scilimits=(2,2))
    ax = f.add_subplot(gs)
    ax.axis('off')
    return ax

def imshow(f, gs, img, xlabel = '', ylabel = '', clabel = '', xticks = None, yticks = None, cmap = None, norm = None, title = '', xlim = None, ylim = None, showbar = False, sm = None, aspect = 'equal'):
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax = get_ax(f, gs, title, xlabel, ylabel, xlim, ylim, xticks, yticks)
    ax.imshow(img, cmap = cmap, norm = norm, aspect = aspect)
    if showbar:
        cbar = plt.colorbar(sm, cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0]), label = clabel)

def imshow_by_layer(f, gs, img, xlabel = '', ylabel = '', xlim = None, ylim = None, title = [''], xticks = None, yticks = None, vlim = None, clabel = '', aspect = 'equal'):
    cmap = cm.get_cmap("Spectral")
    if vlim is None:
        norm  = Normalize(vmin=np.min(img), vmax=np.max(img))
    else:
        norm  = Normalize(vmin=vlim[0], vmax=vlim[1])
    _gs = gs.subgridspec(1,len(img), wspace = 0.5)
    for i in range(len(img)):
        imshow(f, _gs[i], img[i], title = title[i], cmap = cmap, norm = norm, xlabel = xlabel, ylabel = ylabel if i == 0 else '', showbar = i == len(img)-1, clabel = clabel, aspect = aspect)
    ax = f.add_subplot(gs)
    ax.axis('off')
    return ax