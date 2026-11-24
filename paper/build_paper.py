from pathlib import Path
import json, re, html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Preformatted, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'research/gpu_experiment/7b_measured'
s=json.loads((DATA/'summary.json').read_text())
methods=['autoregressive','ngram','heuristic','rules']
labels=['Ordinary\ngreedy','Indexed\nn-gram','One-example\nheuristic','MorphDraft']
fig,axes=plt.subplots(1,2,figsize=(11.8,4.3),gridspec_kw={'width_ratios':[1,1.35]})
plt.rcParams.update({'font.family':'DejaVu Sans'})
palette=['#9ba7b4','#537a9e','#69a5a0','#125e56']
vals=[s['methods'][m]['pooled_tokens_per_second'] for m in methods]
axes[0].bar(range(4),vals,color=palette,width=.65)
axes[0].set_xticks(range(4),labels,fontsize=9)
axes[0].set_ylim(0,245)
axes[0].set_ylabel('Pooled generated tokens / second')
axes[0].set_title('(a) Aggregate throughput',loc='left',fontweight='bold')
for i,v in enumerate(vals): axes[0].text(i,v+5,f'{v:.1f}',ha='center',fontsize=10)
names=list(s['cases'])
ratios=[s['cases'][c]['methods']['rules']['median_tokens_per_second']/s['cases'][c]['methods']['ngram']['median_tokens_per_second'] for c in names]
axes[1].barh(names,ratios,color=['#125e56' if v>=1 else '#ba745b' for v in ratios],height=.65)
axes[1].invert_yaxis(); axes[1].axvline(1,color='#263545',ls='--',lw=1)
axes[1].set_xlim(0,2.65); axes[1].set_xlabel('MorphDraft / indexed n-gram throughput')
axes[1].set_title('(b) Per-case median speedup',loc='left',fontweight='bold')
for i,v in enumerate(ratios): axes[1].text(v+.035,i,f'{v:.2f}×',va='center',fontsize=9)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y' if ax==axes[0] else 'x',alpha=.12)
    ax.set_axisbelow(True)
fig.tight_layout(w_pad=2.5)
fig.savefig(ROOT/'results.png',dpi=220,bbox_inches='tight')
fig.savefig(ROOT/'results-figure.pdf',bbox_inches='tight')
plt.close(fig)

fontdir=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('Body','DejaVuSerif.ttf'),('BodyBold','DejaVuSerif-Bold.ttf'),('Sans','DejaVuSans.ttf'),('SansBold','DejaVuSans-Bold.ttf'),('Mono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='PaperBody',fontName='Body',fontSize=9.5,leading=14,spaceAfter=7,textColor=colors.HexColor('#1f2933')))
styles.add(ParagraphStyle(name='PaperTitle',fontName='SansBold',fontSize=21,leading=26,spaceAfter=14,textColor=colors.HexColor('#104f49')))
styles.add(ParagraphStyle(name='PaperH1',fontName='SansBold',fontSize=13,leading=17,spaceBefore=13,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='PaperH2',fontName='SansBold',fontSize=10,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle(name='Cell',fontName='Sans',fontSize=7.4,leading=10))
styles.add(ParagraphStyle(name='CaptionPaper',fontName='Sans',fontSize=8,leading=11,spaceAfter=9,textColor=colors.HexColor('#465565')))
styles.add(ParagraphStyle(name='CodePaper',fontName='Mono',fontSize=7.1,leading=10,backColor=colors.HexColor('#f1f5f7'),borderPadding=7,spaceBefore=5,spaceAfter=9))

def inline(t):
    t=html.escape(t)
    t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t)
    t=re.sub(r'`([^`]+)`',r'<font name="Mono" size="8">\1</font>',t)
    t=re.sub(r'(https://[^\s]+)',r'<link href="\1" color="#176f88">\1</link>',t)
    return t

raw=(ROOT/'manuscript.md').read_text()
lines=raw.splitlines(); story=[]; i=0
while i<len(lines):
    line=lines[i]
    if not line.strip(): i+=1; continue
    if line=='![MorphDraft throughput results](results.png)':
        story.append(Image(str(ROOT/'results.png'),width=475,height=173)); i+=1; continue
    if line.startswith('```'):
        lang=line[3:]; buf=[]; i+=1
        while i<len(lines) and not lines[i].startswith('```'): buf.append(lines[i]); i+=1
        i+=1
        if lang=='math':
            mathtext='\n'.join(buf)
            f=plt.figure(figsize=(8,.52)); f.text(.02,.35,'$'+mathtext+'$',fontsize=13)
            dest=ROOT/f'equation-{i}.png'; f.savefig(dest,dpi=200,bbox_inches='tight',pad_inches=.12); plt.close(f)
            import PIL.Image
            w,h=PIL.Image.open(dest).size
            w2=min(440,w*.4)
            story.append(Image(str(dest),width=w2,height=h*w2/w)); story.append(Spacer(1,7))
        else: story.append(Preformatted('\n'.join(buf),styles['CodePaper']))
        continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in cells): rows.append([Paragraph(inline(c),styles['Cell']) for c in cells])
            i+=1
        n=len(rows[0]); widths={5:[119,64,96,100,96],4:[156,106,106,107],2:[237.5,237.5]}[n]
        tab=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2efec')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#125e56')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d5dde2')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
        story.append(tab); story.append(Spacer(1,8)); continue
    if line.startswith('# '): story.append(Paragraph(inline(line[2:]),styles['PaperTitle'])); i+=1; continue
    if line.startswith('## '): story.append(Paragraph(inline(line[3:]),styles['PaperH1'])); i+=1; continue
    if line.startswith('### '): story.append(Paragraph(inline(line[4:]),styles['PaperH2'])); i+=1; continue
    buf=[line]; i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','{{FIGURE}}')): buf.append(lines[i]); i+=1
    text=' '.join(buf)
    style=styles['CaptionPaper'] if text.startswith(('**Table','**Figure','**Draft')) else styles['PaperBody']
    story.append(Paragraph(inline(text),style))

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#ced9dc')); c.line(60,45,535,45)
    c.setFont('Sans',7); c.setFillColor(colors.HexColor('#60707b'))
    c.drawString(60,32,'MorphDraft  |  Draft manuscript  |  Exploratory evaluation')
    c.drawRightString(535,32,str(doc.page))

doc=SimpleDocTemplate(str(ROOT/'MorphDraft-paper.pdf'),pagesize=(595.28,841.89),rightMargin=60,leftMargin=60,topMargin=49,bottomMargin=59,title='MorphDraft: Transformation-Aware Drafting for Speculative Decoding with CPU Learning and GPU Verification',author='')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
import fitz
pdf=fitz.open(ROOT/'MorphDraft-paper.pdf')
for pno in [0,4,len(pdf)-1]:
    page=pdf[pno]; page.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(str(ROOT/f'preview-{pno+1}.png'))
text='\n'.join(p.get_text() for p in pdf)
assert '205.55' in text and '132.52' in text and 'References' in text
assert '{{FIGURE}}' not in text
print(json.dumps({'pages':len(pdf),'words':len(raw.split()),'pdf_bytes':(ROOT/'MorphDraft-paper.pdf').stat().st_size,'output':str(ROOT/'MorphDraft-paper.pdf')}))
