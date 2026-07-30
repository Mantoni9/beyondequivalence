import csv, re, sys, statistics
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from rdflib import Graph, RDFS, URIRef
from closure_credit import build_closure, credited_direction
from Alignment import Alignment
from evaluation_recall import RELATION_NORMALIZATION
from tracks.zenodo_loader import load_subdataset
OUT=Path(sys.argv[1])
MODELS=('llama','mistral','gemma4','gpt-oss'); OAEI=('g3-text','g5-groceries','g7-literature','mouse-human'); REL=('<','>','=')
def load_gold(ds):
    if ds=='vdi-ebay':
        s=open('goldstandard_ebay/reference_full.rdf').read(); g={}
        for c in re.findall(r'<Cell>.*?</Cell>',s,re.S):
            e1=re.search(r'entity1 rdf:resource="([^"]*)"',c).group(1); e2=re.search(r'entity2 rdf:resource="([^"]*)"',c).group(1)
            rel=re.search(r'<relation>([^<]*)</relation>',c).group(1).replace('&lt;','<').replace('&gt;','>')
            if rel in REL: g[(e1,e2)]=rel
        return g
    g={}
    for c in Alignment(str(load_subdataset(ds)[2])):
        n=RELATION_NORMALIZATION.get(c.relation.strip())
        if n: g[(c.source,c.target)]=n
    return g
def load_clos(ds):
    tp=('goldstandard_ebay/ebay_kfz_target.owl') if ds=='vdi-ebay' else str(load_subdataset(ds)[1])
    G=Graph(); G.parse(tp)
    return build_closure([(str(c),str(p)) for c,p in G.subject_objects(RDFS.subClassOf) if isinstance(c,URIRef) and isinstance(p,URIRef)])
def load_verify(m,ds):
    out={}
    for r in csv.DictReader(open(f'results/e17/e17_verify_{m}_{ds}_test.tsv'),delimiter='\t'):
        if r['rel'] in REL:
            def f(v):
                try:return float(v)
                except:return None
            out[(r['source_uri'],r['target_uri'])]=(r['rel'],f(r['p_yes']),f(r.get('p_yes_rev')))
    return out
def comp(a,b,anc): return a==b or b in anc.get(a,set()) or a in anc.get(b,set())
def ext(pc,anc,W,rankfn=lambda p,pr:p):
    bysrc={}; out={}
    for (s,t),(rel,p,pr) in pc.items():
        if rel=='<': bysrc.setdefault(s,[]).append((t,rankfn(p,pr)))
        else: out[(s,t)]=rel
    for s,lst in bysrc.items():
        br=[]
        for t,c in sorted(lst,key=lambda x:-(x[1] if x[1] is not None else -9)):
            pl=False
            for b in br:
                if all(comp(t,u,anc) for u in b): b.append(t);pl=True;break
            if not pl and len(br)<W: br.append([t]);pl=True
            if pl: out[(s,t)]='<'
    return out
def _eq(pr,g,U):
    pe=[p for p in U if pr.get(p)=='=']; ge=[p for p in U if g.get(p)=='=']
    tp=sum(1 for p in pe if g.get(p)=='='); P=tp/len(pe) if pe else 0; R=tp/len(ge) if ge else 0
    return 2*P*R/(P+R) if (P+R) else 0.0
def score(preds,g,anc,desc):
    U=set(preds)|set(g); lt=credited_direction(preds,g,anc,'<',U); gt=credited_direction(preds,g,desc,'>',U)
    return statistics.mean([lt['credited']['f1'],gt['credited']['f1'],_eq(preds,g,U)]), lt['credited']['f1'], lt['credited']['precision'], lt['credited']['recall']
G={ds:load_gold(ds) for ds in OAEI+('vdi-ebay',)}; C={ds:load_clos(ds) for ds in OAEI+('vdi-ebay',)}
V={(m,ds):load_verify(m,ds) for m in MODELS for ds in OAEI+('vdi-ebay',)}
pbar={m:statistics.mean([p for ds in OAEI for (r,p,pr) in V[(m,ds)].values() if p]) for m in MODELS}
def arm(name,v,anc,pb):
    if name=='baseline': return {k:r for k,(r,p,pr) in v.items()}
    if name=='BC': return {k:r for k,(r,p,pr) in v.items() if (p or 1)>pb}
    if name.startswith('EXT.W'): return ext({k:v[k] for k in v},anc,int(name[-1]))
    if name=='EXTbyMargin.W1': return ext({k:v[k] for k in v},anc,1,lambda p,pr:(p or 0)-(pr or 0))
    if name=='EXTbyMargin.W2': return ext({k:v[k] for k in v},anc,2,lambda p,pr:(p or 0)-(pr or 0))
    if name=='EXTthenBC':
        e=ext({k:v[k] for k in v},anc,2); return {k:r for k,r in e.items() if (v[k][1] or 1)>pb}

# TABLE 1: main arm comparison (p_yes ranked)
arms=['baseline','BC','EXT.W1','EXT.W2','EXTbyMargin.W1','EXTthenBC']
rows=[]
for m in MODELS:
    pb=pbar[m]
    for scope,dss in (('OAEI-mean',OAEI),('vdi',('vdi-ebay',))):
        base={ds:score(arm('baseline',V[(m,ds)],C[ds][0],pb),G[ds],*C[ds]) for ds in dss}
        for a in arms:
            macs=[];lts=[]
            for ds in dss:
                anc,desc=C[ds]; s=score(arm(a,V[(m,ds)],anc,pb),G[ds],anc,desc)
                macs.append(s[0]-base[ds][0]); lts.append(s[1]-base[ds][1])
            rows.append((m,scope,a,round(statistics.mean(macs),4),round(statistics.mean(lts),4)))
with open(OUT/'table1_arm_comparison_pyes.tsv','w',newline='') as f:
    w=csv.writer(f,delimiter='\t'); w.writerow(['model','scope','arm','d_credited_macro','d_credited_lt']); w.writerows(rows)

# TABLE 2: vdi absolute (baseline + best), per-relation P/R
rows=[]
for m in MODELS:
    pb=pbar[m]
    anc,desc=C['vdi-ebay']
    for a in ['baseline','EXT.W1','EXT.W2','EXT.W3','BC','EXTthenBC']:
        mac,lt,P,R=score(arm(a,V[(m,'vdi-ebay')],anc,pb) if a!='EXT.W3' else ext({k:V[(m,'vdi-ebay')][k] for k in V[(m,'vdi-ebay')]},anc,3),G['vdi-ebay'],anc,desc)
        rows.append((m,a,round(mac,4),round(lt,4),round(P,4),round(R,4)))
with open(OUT/'table2_vdi_absolute.tsv','w',newline='') as f:
    w=csv.writer(f,delimiter='\t'); w.writerow(['model','arm','credited_macro','credited_lt_f1','credited_lt_P','credited_lt_R']); w.writerows(rows)
print("wrote table1, table2  | pbar:",{k:round(v,3) for k,v in pbar.items()})
