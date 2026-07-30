import csv, re, sys, statistics
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from rdflib import Graph, RDFS, URIRef
from closure_credit import build_closure, credited_direction
from Alignment import Alignment
from evaluation_recall import RELATION_NORMALIZATION
from tracks.zenodo_loader import load_subdataset
OUT=Path(sys.argv[1]); REL=('<','>','=')
def comp(a,b,anc): return a==b or b in anc.get(a,set()) or a in anc.get(b,set())
def ext(pc,anc,W):
    bysrc={}; out={}
    for (s,t),(rel,c) in pc.items():
        if rel=='<': bysrc.setdefault(s,[]).append((t,c))
        else: out[(s,t)]=rel
    for s,lst in bysrc.items():
        br=[]
        for t,c in sorted(lst,key=lambda x:-x[1]):
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
    return statistics.mean([lt['credited']['f1'],gt['credited']['f1'],_eq(preds,g,U)]),lt['credited']['f1']

# ---- TABLE 3: GPU-free extractor by Stage-2 confidence (g3/g5/g7 matrix cells) ----
DSMAP={'g3':'g3-text','g5':'g5-groceries','g7':'g7-literature'}
CELLS={('llama','g3'):'llama_g3_seed42',('llama','g5'):'llama_g5_seed42_REUSED',('llama','g7'):'llama_g7_seed42_REUSED',
 ('mistral','g3'):'mistral_g3_seed42',('mistral','g5'):'mistral_g5_seed42',('mistral','g7'):'mistral_g7_seed42',
 ('gemma4','g3'):'gemma4_g3_seed42',('gemma4','g5'):'gemma4_g5_seed42',('gemma4','g7'):'gemma4_g7_seed42',
 ('gpt-oss','g3'):'gpt-oss_g3_seed42',('gpt-oss','g5'):'gpt-oss_g5_seed42',('gpt-oss','g7'):'gpt-oss_g7_seed42'}
def gold(ds):
    g={}
    for c in Alignment(str(load_subdataset(ds)[2])):
        n=RELATION_NORMALIZATION.get(c.relation.strip())
        if n: g[(c.source,c.target)]=n
    return g
def clos(ds):
    G=Graph(); G.parse(str(load_subdataset(ds)[1]))
    return build_closure([(str(c),str(p)) for c,p in G.subject_objects(RDFS.subClassOf) if isinstance(c,URIRef) and isinstance(p,URIRef)])
def cell(name):
    out={}
    for r in csv.DictReader(open(f'results/stage2_results_bundle/02_matrix_cells/{name}/predictions.tsv'),delimiter='\t'):
        if r.get('kept')=='True' and r.get('predicted_relation') in REL:
            try:c=float(r['confidence'])
            except:c=0.0
            out[(r['source_uri'],r['target_uri'])]=(r['predicted_relation'],c)
    return out
Gd={ds:gold(DSMAP[ds]) for ds in DSMAP}; Cl={ds:clos(DSMAP[ds]) for ds in DSMAP}
rows=[]
for m in ('llama','mistral','gemma4','gpt-oss'):
    for ds in DSMAP:
        pc=cell(CELLS[(m,ds)]); anc,desc=Cl[ds]
        base={k:r for k,(r,c) in pc.items()}
        b=score(base,Gd[ds],anc,desc)[0]
        for W in (1,2,3):
            mac=score(ext(pc,anc,W),Gd[ds],anc,desc)[0]
            rows.append((m,ds,f'W{W}',round(mac,4),round(mac-b,4),round(b,4)))
with open(OUT/'table3_extractor_gpufree_stage2conf.tsv','w',newline='') as f:
    w=csv.writer(f,delimiter='\t'); w.writerow(['model','dataset','W','credited_macro','d_vs_baseline','baseline']); w.writerows(rows)

# ---- TABLE 4: vdi decomposition (seed vs full, per-relation, ceiling) ----
def load_ref(p):
    s=open(p).read(); out={}
    for c in re.findall(r'<Cell>.*?</Cell>',s,re.S):
        e1=re.search(r'entity1 rdf:resource="([^"]*)"',c).group(1); e2=re.search(r'entity2 rdf:resource="([^"]*)"',c).group(1)
        rel=re.search(r'<relation>([^<]*)</relation>',c).group(1).replace('&lt;','<').replace('&gt;','>')
        if rel in REL: out[(e1,e2)]=rel
    return out
seed=load_ref('goldstandard_ebay/reference_seed.rdf'); full=load_ref('goldstandard_ebay/reference_full.rdf')
Gv=Graph(); Gv.parse('goldstandard_ebay/ebay_kfz_target.owl')
anc,desc=build_closure([(str(c),str(p)) for c,p in Gv.subject_objects(RDFS.subClassOf) if isinstance(c,URIRef) and isinstance(p,URIRef)])
rows=[]
for m in ('llama','mistral','gemma4','gpt-oss'):
    preds={}
    for r in csv.DictReader(open(f'results/e17/e17_verify_{m}_vdi-ebay_test.tsv'),delimiter='\t'):
        if r['rel'] in REL: preds[(r['source_uri'],r['target_uri'])]=r['rel']
    for refname,ref in (('seed',seed),('full',full)):
        mac,lt=score(preds,ref,anc,desc)
        rows.append((m,refname,'baseline',round(mac,4),round(lt,4)))
    # ceiling: perfect precision filter at current recall (2R/(1+R)) vs full, <-only
    U=set(preds)|set(full); ltd=credited_direction(preds,full,anc,'<',U)
    R=ltd['credited']['recall']; ceil=2*R/(1+R) if R else 0
    rows.append((m,'full','oracle_precision_ceiling_lt',round(ceil,4),round(ceil,4)))
with open(OUT/'table4_vdi_decomposition.tsv','w',newline='') as f:
    w=csv.writer(f,delimiter='\t'); w.writerow(['model','reference','arm','credited_macro','credited_lt_f1']); w.writerows(rows)
print("wrote table3, table4")
