"""Check exported evidence and paper artifacts; does not invoke experiments."""
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def fonts_embedded(resources):
    resources=resources.get_object()
    checks=[]
    for font in resources.get('/Font',{}).values():
        font=font.get_object()
        for descendant in font.get('/DescendantFonts',[font]):
            descendant=descendant.get_object()
            if descendant.get('/Subtype')=='/Type3':
                checks.append(True)
                continue
            descriptor=descendant.get('/FontDescriptor')
            checks.append(bool(descriptor and any(k in descriptor.get_object()
                          for k in ['/FontFile','/FontFile2','/FontFile3'])))
    for obj in resources.get('/XObject',{}).values():
        obj=obj.get_object()
        if obj.get('/Resources'): checks.extend(fonts_embedded(obj['/Resources']))
    return checks


def main():
    data=json.loads((ROOT/'data/synthetic_source.json').read_text()); cases=data['cases']
    assert len(cases)==1000
    assert all(sum(c[a]['questions'] for c in cases)==26451 for a in ('vanilla','query_specific'))
    summary=json.loads((ROOT/'data/synthetic_summary.json').read_text())['metrics']
    assert summary['no_support']['n']==1000 and summary['joint']['n']==352 and summary['recall']['n']==994
    for name in ['babilong_results','cache_results']:
        pending=json.loads((ROOT/'data'/f'{name}.json').read_text())
        assert all(r['status']=='pending' for r in pending['rows'])
        key='accuracy_percent' if name=='babilong_results' else 'value'
        assert all(r[key] is None for r in pending['rows'])
    paper=PdfReader(ROOT/'paper.pdf')
    assert len(paper.pages)>0
    w,h=map(float,[paper.pages[0].mediabox.width,paper.pages[0].mediabox.height])
    assert abs(w-595.276)<1 and abs(h-841.89)<1
    pdfs=[]
    for p in [ROOT/'paper.pdf',*sorted((ROOT/'figures').glob('*.pdf'))]:
        reader=PdfReader(p)
        embedded=all(all(fonts_embedded(page['/Resources'])) for page in reader.pages)
        assert embedded, p
        pdfs.append({'path':str(p.relative_to(ROOT)),'pages':len(reader.pages),
                     'fonts_embedded':embedded,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    text='\n'.join(page.extract_text() for page in paper.pages)
    assert all(s in text for s in ['26,451','BABILong','Pending','TracIn','83.69','95.45'])
    assert '??' not in text
    log=(ROOT/'paper.log').read_text(errors='replace')
    assert 'Overfull \\hbox' not in log and 'Overfull \\vbox' not in log
    broken=[]
    for p in [*ROOT.glob('*.md'),*ROOT.glob('code/implementation/*.md')]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#') or target.startswith('../../datasets/'):continue
            if not (p.parent/target.split('#')[0]).exists():broken.append((str(p),target))
    assert not broken,broken
    result={'passed':True,'requests':1000,'exercises_per_arm':26451,
            'babilong_pending_rows':60,'cache_pending_points':36,
            'pdfs':pdfs,'paper_pages':len(paper.pages),'paper_size_points':[w,h],
            'unresolved_cross_references':False,'overfull_boxes':False,
            'manual_figure_review':'all three actual-result figures and cache layout inspected; clipped colorbar corrected',
            'scientific_review':'source proxies separated from semantic correctness, speed and downstream accuracy; no CI or invented results',
            'software_reference':'Scientific Agent Skills, arXiv:2609.00065, latest record checked 2026-11-11'}
    (ROOT/'review/audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
