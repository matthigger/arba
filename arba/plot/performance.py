import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mh_pytools import file


def plot_performance(f_in, x_axis):
    folder = f_in.parent
    t2size_method_perf_tree = file.load(f_in)

    t2size_list = sorted(t2size_method_perf_tree.keys())
    method_list = sorted(t2size_method_perf_tree[t2size_list[0]].keys())

    sns.set(font_scale=1.2)
    cm = plt.get_cmap('Set1')
    color_dict = {method: cm(idx) for idx, method in enumerate(method_list)}
    color_dict.update({method.upper(): c for method, c in color_dict.items()})

    series_list = list()
    for (t2size, num_vox), d0 in t2size_method_perf_tree.items():
        for method, ss_list in d0.items():
            for (sens, spec) in ss_list:
                s = pd.Series({'Effect Size (vox)': num_vox,
                               'T-squared': t2size,
                               'Method': method.upper(),
                               'Sensitivity': sens,
                               'Specificity': spec})
                series_list.append(s)
    df = pd.concat(series_list, axis=1).T

    # https://stackoverflow.com/questions/12844529/no-numeric-types-to-aggregate-change-in-groupby-behaviour
    for col in df.columns:
        if col == 'Method':
            continue
        df[col] = df[col].astype(float)

    sns.set(font_scale=1)
    for y_ax in ['Sensitivity', 'Specificity']:
        f_out = folder / f'{y_ax}.pdf'
        with PdfPages(f_out) as pdf:
            sns.lineplot(x=x_axis, y=y_ax, data=df, hue='Method',
                         ci=95,
                         palette=color_dict)
            plt.gca().set_xscale('log')
            plt.gcf().set_size_inches(4, 4)
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            plt.close()

    f_out = folder / 'sens_spec_t2.pdf'
    with PdfPages(f_out) as pdf:
        for t2size in t2size_list:
            for method in reversed(method_list):
                perf_list = t2size_method_perf_tree[t2size][method]

                sens = [x[0] for x in perf_list]
                spec = [x[1] for x in perf_list]

                plt.scatter(spec, sens, label=method, alpha=.3,
                            color=color_dict[method])

            for method in reversed(method_list):
                perf_list = t2size_method_perf_tree[t2size][method]

                sens = [x[0] for x in perf_list]
                spec = [x[1] for x in perf_list]

                plt.scatter(np.mean(spec), np.mean(sens), marker='s',
                            color='w',
                            edgecolors=color_dict[method], linewidths=2)

            ax = plt.gca()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            t2, size = t2size
            plt.suptitle(f't2: {t2:.1E}, size: {size:.1E}')
            plt.xlabel('specificity')
            plt.ylabel('sensitivity')
            plt.xlim((0, 1.05))
            plt.ylim((-.05, 1.05))
            pdf.savefig(plt.gcf())
            plt.gcf().set_size_inches(8, 8)
            plt.close()
