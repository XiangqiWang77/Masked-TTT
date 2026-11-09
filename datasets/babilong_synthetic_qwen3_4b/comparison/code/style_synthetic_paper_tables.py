"""Render the user's grouped, gray/cyan, dark-red-best synthetic-data tables."""
import argparse
from pathlib import Path
import json
import sys


KEYS=('dilution_no_support','dilution_query_focus','support_recall',
      'fragmentation_joint_exercises','fragmentation_joint_cases')
NAMES=('No-support','Query focus','Support recall','Joint exercise','Joint case')
ARROWS=(r'$\downarrow$',r'$\uparrow$',r'$\uparrow$',r'$\uparrow$',r'$\uparrow$')
METHODS=('Vanilla Cartridges','Question-guided (ours)')
DARKRED=(.6148,0.,0.)


def tex_number(v,index,direction):
    values=(v['original'],v['guided']);value=values[index]
    if value is None:return '--'
    number=f'{value:.2f}'
    other=values[1-index]
    if other is None or abs(value-other)<1e-10 or direction*(value-other)>0:
        return r'\textcolor{darkred}{\textbf{'+number+'}}'
    return r'\uline{'+number+'}'


def latex(report,groups,title,label,grouped=False):
    leftcols=4 if grouped else 1
    columns='?l|r|r?l?ccc?cc?' if grouped else '?l?ccc?cc?'
    count=leftcols+5
    caption=(r'\textbf{'+title+r'} Both methods use Qwen3-4B and the same per-case synthetic-question '
        r'counts (20,558 train and 7,391 validation examples over 1,000 BABILong document--query pairs). '
        r'All values are percentages. $\downarrow$/$\uparrow$ indicate the preferred direction. '
        r'Source diagnostics use exact source-mapped cases ($n_D$); fragmentation uses only cross-chunk '
        r'support cases ($n_F$). Query focus uses every planned case. '
        r'Joint evidence exposure does not establish necessary multi-hop reasoning or answer accuracy. '
        r'The better result is \textcolor{darkred}{red}; the second is \uline{underlined}.')
    if not grouped:caption+=r' Here $n_D=993$, focus $n=1000$, and $n_F=296$.'
    lines=[r'\begin{table*}[t!]',r'\renewcommand{\arraystretch}{1.10}',
        r'\caption{'+caption+'}',r'\label{tab:'+label+'}',r'\centering',
        r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{'+columns+'}',
        r'\specialrule{1.2pt}{0pt}{0pt}']
    if grouped:
        lines += [r'Group & $n_D$ & $n_F$ & Method & \multicolumn{3}{c?}{Dilution and relevance} & \multicolumn{2}{c?}{Fragmentation} \\',
                  r'\cline{5-7}\cline{8-9}',r' & & & & No-support $\downarrow$ & Query focus $\uparrow$ & Support recall $\uparrow$ & Joint exercise $\uparrow$ & Joint case $\uparrow$ \\']
    else:
        lines += [r'Method & \multicolumn{3}{c?}{Dilution and relevance} & \multicolumn{2}{c?}{Fragmentation} \\',
                  r'\cline{2-4}\cline{5-6}',r' & No-support $\downarrow$ & Query focus $\uparrow$ & Support recall $\uparrow$ & Joint exercise $\uparrow$ & Joint case $\uparrow$ \\']
    lines.append(r'\specialrule{1.2pt}{0pt}{0pt}')
    for position,(name,key) in enumerate(groups):
        group=report['groups'][key]['metrics']
        for index,method in enumerate(METHODS):
            lines.append(r'\rowcolor{gray!10}' if index==0 else r'\rowcolor{cyan!10}')
            prefix=([name if index==0 else '',str(group[KEYS[0]]['n']),str(group[KEYS[-1]]['n'])] if grouped else [])
            numbers=[tex_number(group[metric],index,report['metric_specs'][metric]['direction']) for metric in KEYS]
            lines.append(' & '.join(prefix+[method]+numbers)+r' \\')
        if position<len(groups)-1:lines.append(r'\specialrule{.5pt}{0pt}{0pt}')
    if not grouped:
        lines.append(r'\rowcolor{lightyellow}')
        v=report['groups']['overall']['metrics']
        lines.append(r'$\Delta$ (guided$-$agnostic, pp) & '+
            ' & '.join(f'{v[key]["delta"]:+.2f}' for key in KEYS)+r' \\')
    lines.extend([r'\specialrule{1.2pt}{0pt}{0pt}',r'\end{tabular}}',r'\end{table*}',''])
    return '\n'.join(lines)


def draw(report,groups,title,grouped=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    nrows=2*len(groups)+(0 if grouped else 1)
    fig,ax=plt.subplots(figsize=(11.0,max(3.0,1.5+.255*nrows)))
    fig.subplots_adjust(left=.02,right=.98,bottom=.13,top=.83)
    ax.set_xlim(0,1);ax.set_ylim(0,nrows+2.3);ax.axis('off')
    widths=[.065,.045,.045,.23]+[.123]*5 if grouped else [.31]+[.138]*5
    xs=np.concatenate(([0],np.cumsum(widths)));xs=xs/xs[-1]
    left=4 if grouped else 1
    top=nrows+2.3;group_y=top-.42;header_y=top-1.43;body_top=nrows
    fig.suptitle(title,fontsize=13,fontweight='bold',fontfamily='DejaVu Serif',y=.98)
    fig.text(.5,.89,'Matched Qwen3-4B BABILong synthetic datasets; 27,949 examples per method',
             ha='center',fontsize=9,fontfamily='DejaVu Serif')
    def text(x,y,value,**kw):
        return ax.text(x,y,value,ha='center',va='center',fontsize=9,fontfamily='DejaVu Serif',**kw)
    if grouped:
        for i,name in enumerate(('Group',r'$n_D$',r'$n_F$','Method')):
            text((xs[i]+xs[i+1])/2,top-1.08,name,fontweight='bold')
    else:text((xs[0]+xs[1])/2,top-1.08,'Method',fontweight='bold')
    text((xs[left]+xs[left+3])/2,group_y,'Dilution and relevance',fontweight='bold')
    text((xs[left+3]+xs[-1])/2,group_y,'Fragmentation',fontweight='bold')
    for j,(name,arrow) in enumerate(zip(NAMES,ARROWS)):
        text((xs[left+j]+xs[left+j+1])/2,header_y,name+' '+arrow,fontweight='bold')
    for a,b in ((left,left+3),(left+3,len(xs)-1)):
        ax.plot([xs[a],xs[b]],[top-.85,top-.85],color='black',lw=.5)
    underlines=[];row=0
    for position,(name,key) in enumerate(groups):
        group=report['groups'][key]['metrics']
        for method_index,method in enumerate(METHODS):
            y=nrows-row-1
            color='#F2F2F2' if method_index==0 else '#E5FFFF'
            ax.add_patch(Rectangle((0,y),1,1,facecolor=color,edgecolor='none',zorder=-1))
            if grouped:
                cells=(name if method_index==0 else '',str(group[KEYS[0]]['n']),str(group[KEYS[-1]]['n']),method)
            else:cells=(method,)
            for i,value in enumerate(cells):text((xs[i]+xs[i+1])/2,y+.5,value)
            for j,metric in enumerate(KEYS):
                v=group[metric];a,b=v['original'],v['guided'];value=(a,b)[method_index]
                other=(a,b)[1-method_index];direction=report['metric_specs'][metric]['direction']
                best=value is not None and (other is None or abs(value-other)<1e-10 or direction*(value-other)>0)
                artist=text((xs[left+j]+xs[left+j+1])/2,y+.5,'—' if value is None else f'{value:.2f}',
                    color=DARKRED if best else 'black',fontweight='bold' if best else 'normal')
                if value is not None and not best:underlines.append(artist)
            row+=1
        if position<len(groups)-1:ax.plot([0,1],[nrows-row]*2,color='black',lw=.5)
    if not grouped:
        ax.add_patch(Rectangle((0,0),1,1,facecolor='#FFFFF2',edgecolor='none',zorder=-1))
        text((xs[0]+xs[1])/2,.5,'Δ guided − agnostic (pp)',fontweight='bold')
        group=report['groups']['overall']['metrics']
        for j,key in enumerate(KEYS):text((xs[j+1]+xs[j+2])/2,.5,f'{group[key]["delta"]:+.2f}')
    for x in [0,xs[left],xs[left+3],1]:ax.plot([x,x],[0,top],color='black',lw=1.)
    if grouped:
        for x in xs[1:3]:ax.plot([x,x],[0,top],color='black',lw=.5)
    for y in (0,body_top,top):ax.plot([0,1],[y,y],color='black',lw=1.2,clip_on=False)
    note=('Source-mapped cases: nD; cross-chunk subset: nF; focus uses all planned cases. '
          'Gray: vanilla; cyan: guided; red: better; underline: second.') if grouped else (
          'Source diagnostics n=993; query focus n=1000; cross-chunk diagnostics n=296. '
          'These are evidence-exposure diagnostics, not answer accuracy.')
    fig.text(.5,.045,note,ha='center',fontsize=8,fontfamily='DejaVu Serif')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for artist in underlines:
        box=artist.get_window_extent(renderer=renderer)
        points=ax.transData.inverted().transform([[box.x0,box.y0-1.1],[box.x1,box.y0-1.1]])
        ax.plot(points[:,0],points[:,1],color='black',lw=.5)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--statistics',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    report=json.loads(Path(a.statistics).read_text());root=Path(a.output)
    tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
    figures=root/'figures';figures.mkdir(parents=True,exist_ok=True)
    jobs=[('overall',[('Overall','overall')],'Synthetic data quality: dilution and fragmentation',False),
          ('by_length',[(f'{n//1024}K',f'length_{n}') for n in (8192,16384,32768,65536)],'Synthetic data quality by document length',True),
          ('by_task',[(f'qa{i}',f'task_qa{i}') for i in range(1,11)],'Synthetic data quality by task',True)]
    jobs += [(f'by_task_{n//1024}k',[(f'qa{i}',f'cell_qa{i}_{n}') for i in range(1,11)],
              f'Synthetic data quality: all tasks at {n//1024}K',True) for n in (8192,16384,32768,65536)]
    preamble=r'''% Add once in the manuscript preamble (user-requested table style).
\usepackage[table]{xcolor}
\usepackage{array,booktabs,graphicx}
\usepackage[normalem]{ulem}
\newcolumntype{?}{!{\vrule width 1pt}}
\definecolor{darkred}{rgb}{0.6148,0.,0.}
\definecolor{lightyellow}{rgb}{1.,1.,0.95}
'''
    (tables/'preamble.tex').write_text(preamble)
    for name,groups,title,grouped in jobs:
        (tables/(name+'.tex')).write_text(latex(report,groups,title,'synthetic-'+name,grouped))
    standalone=r'''\documentclass[10pt,twocolumn]{article}
\usepackage[margin=.65in]{geometry}
\input{preamble.tex}
\begin{document}
'''
    standalone+='\n'.join(r'\input{'+name+'.tex}\clearpage' for name,*_ in jobs)
    standalone+='\n'+r'\end{document}'+'\n'
    (tables/'paper_tables.tex').write_text(standalone)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    skill=Path('/home/da839/.codex/skills/scientific-visualization/scripts')
    if skill.is_dir():
        sys.path.insert(0,str(skill))
        from figure_export import export_figure
    else:
        def export_figure(fig,filename,*,formats,dpi,provenance,**kwargs):
            for ext in formats:
                fig.savefig(Path(filename).with_suffix('.'+ext),dpi=dpi,facecolor='white')
            Path(filename).with_suffix('.export.json').write_text(json.dumps({'provenance':provenance},indent=2)+'\n')
    with plt.rc_context({'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'}):
        with PdfPages(tables/'all_styled_tables.pdf') as pdf:
            for name,groups,title,grouped in jobs:
                fig=draw(report,groups,title,grouped)
                pdf.savefig(fig,facecolor='white')
                if name in {'overall','by_length'}:
                    export_figure(fig,figures/('table_'+name),formats=['png','svg','pdf'],dpi=300,
                        overwrite=True,write_manifest=True,provenance={
                            'data':'comparison.json','style':'User-provided gray/cyan rows, darkred best, underlined second, grouped headers and vertical rules',
                            'values_unchanged':True,'LaTeX_engine_used':False})
                plt.close(fig)
    print({'tables':len(jobs),'PDF':str(tables/'all_styled_tables.pdf'),'LaTeX_source':str(tables/'paper_tables.tex')})


if __name__=='__main__':main()
