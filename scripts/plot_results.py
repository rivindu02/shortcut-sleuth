"""Plot measured seed summaries and all nine shape/background groups."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.metrics)
    result = json.loads(source.read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = [name for name in ('cnn/correlated','cnn/independent','vit/correlated','vit/independent') if name in result['groups']]
    if not names:
        raise ValueError('No measured groups to plot')
    labels = [('CNN' if name.startswith('cnn') else 'ViT') + '\n' + name.split('/')[1] for name in names]
    groups = [result['groups'][name] for name in names]
    seed_counts = [g['seed_count'] for g in groups]
    note = 'Mean ± sample SD across three training seeds' if all(n == 3 for n in seed_counts) else 'Measured means; sample SD shown only when at least two seeds exist'
    stage = ('Frozen study' if result['full_matrix_complete'] else 'Development matrix') + ' · ' + result['protocol']
    colours = ['#c9ee91','#91bde8','#c4a2ef']
    style = {'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#11151b',
             'axes.facecolor':'#191e26','axes.edgecolor':'#414955','axes.labelcolor':'#e9e6e0',
             'text.color':'#e9e6e0','xtick.color':'#bec7d3','ytick.color':'#bec7d3',
             'grid.color':'#414955','svg.hashsalt':'shortcut-sleuth','text.usetex':False}
    def stats(key):
        means = np.array([g['metrics'][key]['mean'] for g in groups], dtype=float) * 100
        errors = np.array([g['metrics'][key]['sample_std'] if g['metrics'][key]['sample_std'] is not None else np.nan for g in groups], dtype=float) * 100
        return means, errors
    def save(fig, name):
        fig.savefig(output / (name+'.svg'), metadata={'Creator':'Shortcut Sleuth','Date':None})
        fig.savefig(output / (name+'.png'), dpi=160, metadata={'Creator':'Shortcut Sleuth'})
        plt.close(fig)
    with plt.rc_context(style):
        fig, axes = plt.subplots(1,3,figsize=(15.5,5.2),gridspec_kw={'width_ratios':[1.55,1,1]},layout='constrained')
        x = np.arange(len(names))
        width = .24
        for j,(key,label) in enumerate((('training_like_accuracy','Training-like'),('independent_accuracy','Independent'),('conflicting_accuracy','Conflicting'))):
            means, errors = stats(key)
            axes[0].bar(x+(j-1)*width,means,width,yerr=errors,capsize=2,color=colours[j],label=label,
                        error_kw={'ecolor':'#f1eee7','elinewidth':1})
        axes[0].set_title('Paired test accuracy',loc='left',fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncols=3,fontsize=8,frameon=False)
        for ax,key,title,ylabel in ((axes[1],'background_flip_rate','Prediction changes','Objects with a label flip (%)'),
                                    (axes[2],'true_class_probability_range','Probability sensitivity','True-class range (pp)')):
            means, errors = stats(key)
            ax.bar(x,means,.58,yerr=errors,capsize=3,color=[colours[0] if n.startswith('cnn') else colours[2] for n in names],
                   error_kw={'ecolor':'#f1eee7','elinewidth':1})
            ax.set_title(title,loc='left',fontweight='bold')
            ax.set_ylabel(ylabel)
            for i,value in enumerate(means):
                ax.text(i,min(102,value+np.nan_to_num(errors[i])+1.5),f'{value:.1f}',ha='center',fontsize=9)
        for ax in axes:
            ax.set_xticks(x,labels,fontsize=8)
            ax.set_ylim(0,107)
            ax.set_yticks([0,25,50,75,100])
            ax.grid(axis='y',alpha=.5)
            ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle(stage+' · '+note,fontsize=13,fontweight='bold')
        save(fig,'overview')
        fig, axes = plt.subplots(1,len(names),figsize=(3.5*len(names)+1,4.3),squeeze=False,layout='constrained')
        for ax,name,group in zip(axes[0],names,groups):
            matrix=np.array(group['metrics']['group_accuracy']['mean'])
            image=ax.imshow(matrix,vmin=0,vmax=1,cmap='cividis')
            ax.set_title(name.replace('/',' / '),fontsize=11)
            ax.set_xticks(range(3),['Coral','Mint','Violet'])
            ax.set_yticks(range(3),['Circle','Square','Triangle'])
            ax.set_xlabel('Background')
            for i in range(3):
                for j in range(3):
                    ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',fontsize=10,
                            color='#10151b' if matrix[i,j]>.6 else '#ffffff')
        bar=fig.colorbar(image,ax=axes[0].tolist(),shrink=.78,pad=.02)
        bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        bar.set_label('Mean accuracy')
        fig.suptitle('All nine shape/background groups · means over available training seeds',fontsize=13,fontweight='bold')
        save(fig,'groups')
    record={'matplotlib':matplotlib.__version__,'metrics_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'protocol':result['protocol'],'full_matrix_complete':result['full_matrix_complete'],
            'training_seed_counts':dict(zip(names,seed_counts)),
            'error_bars':'sample standard deviation across training seeds; not a confidence interval',
            'files':['overview.svg','overview.png','groups.svg','groups.png']}
    (output/'plots.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print('Saved measured overview and group figures')


if __name__=='__main__':
    main()
