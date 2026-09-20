import json, math
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    linear_sum_assignment = None
    HAS_SCIPY = False

VERSION = "0.8.1"

def _regions(v, name):
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], dict):
        v = v[0]
    if not isinstance(v, dict):
        raise ValueError(f"{name} must be RA_REGIONS")
    b = np.asarray(v["boxes"], dtype=np.float32)
    if b.size == 0:
        b = b.reshape(0, 4)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"{name}.boxes must be [N,4]")
    h, w = int(v["height"]), int(v["width"])
    src = v.get("source_indices", list(range(len(b))))
    if len(src) != len(b):
        src = list(range(len(b)))
    return b, h, w, [int(x) for x in src]

def _norm(b, h, w):
    if len(b) == 0:
        return b.astype(np.float32), np.zeros((0,2), np.float32), np.zeros((0,2), np.float32)
    n = b.astype(np.float32).copy()
    n[:, [0,2]] /= float(w)
    n[:, [1,3]] /= float(h)
    c = np.stack(((n[:,0]+n[:,2])*0.5, (n[:,1]+n[:,3])*0.5), 1)
    s = np.stack((n[:,2]-n[:,0], n[:,3]-n[:,1]), 1)
    return n, c, s

def _iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw * ih
    aa = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    bb = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    u = aa + bb - inter
    return inter/u if u > 0 else 0.0

def _cluster(c, axis, tol):
    if len(c) == 0:
        return np.zeros((0,), np.int32), []
    order = np.argsort(c[:, axis])
    groups, cur = [], [int(order[0])]
    mean = float(c[cur[0], axis])
    for x in order[1:]:
        x = int(x); v = float(c[x, axis])
        if abs(v-mean) <= tol:
            cur.append(x)
            mean = float(np.mean([c[i, axis] for i in cur]))
        else:
            groups.append(cur); cur = [x]; mean = v
    groups.append(cur)
    labels = np.full(len(c), -1, np.int32)
    for gi, g in enumerate(groups):
        for i in g:
            labels[i] = gi
    return labels, groups

def _neigh(c, k=4):
    out = np.zeros((len(c), k), np.float32)
    for i in range(len(c)):
        d = np.sort(np.linalg.norm(c-c[i], axis=1))
        d = d[d > 1e-8][:k]
        vals = list(d)
        while len(vals) < k:
            vals.append(vals[-1] if vals else 0.0)
        out[i] = vals
    return out

def _cost(bn,bc,bs,br,bcol,bnei,an,ac,ass,ar,acol,anei,wc,wi,ws,wg,wn):
    C = np.zeros((len(bn), len(an)), np.float32)
    detail = {}
    diag = math.sqrt(2.0)
    brd = max(1, int(br.max()) if len(br) else 1)
    ard = max(1, int(ar.max()) if len(ar) else 1)
    bcd = max(1, int(bcol.max()) if len(bcol) else 1)
    acd = max(1, int(acol.max()) if len(acol) else 1)
    for i in range(len(bn)):
        for j in range(len(an)):
            center = float(np.linalg.norm(bc[i]-ac[j]) / diag)
            iou = float(_iou(bn[i], an[j]))
            size = float(np.mean(np.abs(np.log(np.maximum(ass[j],1e-6) / np.maximum(bs[i],1e-6)))))
            grid = 0.5*abs(float(br[i])/brd - float(ar[j])/ard) + 0.5*abs(float(bcol[i])/bcd - float(acol[j])/acd)
            neigh = float(np.mean(np.abs(bnei[i]-anei[j])))
            total = wc*center + wi*(1.0-iou) + ws*size + wg*grid + wn*neigh
            C[i,j] = total
            detail[(i,j)] = {"center_normdiag":center,"iou":iou,"size_log_cost":size,"grid_cost":grid,"neighborhood_cost":neigh,"total_cost":float(total)}
    return C, detail

def _assign(C):
    if C.size == 0:
        return [], [], "none"
    if HAS_SCIPY:
        r, c = linear_sum_assignment(C)
        return list(map(int,r)), list(map(int,c)), "hungarian/scipy"
    pairs = sorted((float(C[i,j]), i, j) for i in range(C.shape[0]) for j in range(C.shape[1]))
    ur, uc, rr, cc = set(), set(), [], []
    for _, i, j in pairs:
        if i in ur or j in uc:
            continue
        ur.add(i); uc.add(j); rr.append(i); cc.append(j)
    return rr, cc, "greedy/fallback"

def _structure(matches, bc, ac, row_tol, col_tol, order_tol):
    mapping = {m["before_idx"]: m["after_idx"] for m in matches}
    ids = sorted(mapping)
    lr, ud, rows, cols = [], [], [], []
    for p in range(len(ids)):
        for q in range(p+1, len(ids)):
            i, k = ids[p], ids[q]
            j, l = mapping[i], mapping[k]
            bdx, adx = float(bc[k,0]-bc[i,0]), float(ac[l,0]-ac[j,0])
            bdy, ady = float(bc[k,1]-bc[i,1]), float(ac[l,1]-ac[j,1])
            if abs(bdx)>order_tol and abs(adx)>order_tol and np.sign(bdx)!=np.sign(adx): lr.append([i,k])
            if abs(bdy)>order_tol and abs(ady)>order_tol and np.sign(bdy)!=np.sign(ady): ud.append([i,k])
            if (abs(bdy)<=row_tol) != (abs(ady)<=row_tol): rows.append([i,k])
            if (abs(bdx)<=col_tol) != (abs(adx)<=col_tol): cols.append([i,k])
    return {"left_right_order_flips":lr,"above_below_order_flips":ud,"same_row_changes":rows,"same_column_changes":cols}

def _to_pil(t):
    if isinstance(t, list): t = t[0]
    a = np.clip(t[0].detach().cpu().float().numpy(), 0, 1)
    a = (a*255+0.5).astype(np.uint8)
    if a.shape[-1] == 4: a = a[...,:3]
    return Image.fromarray(a, "RGB")

def _fit(img, h=900):
    if img.height == h: return img
    w = max(1, int(round(img.width*h/img.height)))
    return img.resize((w,h), Image.Resampling.LANCZOS)

def _overlay(before_image, after_image, bb, ab, matches, missing, added, cwarn, swarn):
    b0, a0 = _to_pil(before_image), _to_pil(after_image)
    b, a = _fit(b0), _fit(a0)
    gap, head = 24, 44
    cv = Image.new("RGB", (b.width+a.width+gap, b.height+head), (22,22,22))
    cv.paste(b, (0,head)); cv.paste(a, (b.width+gap,head))
    d = ImageDraw.Draw(cv)
    d.text((10,12), "BEFORE / GEOMETRY TRUTH", fill="white")
    d.text((b.width+gap+10,12), "AFTER / AI RESULT", fill="white")
    bsx,bsy = b.width/b0.width, b.height/b0.height
    asx,asy = a.width/a0.width, a.height/a0.height
    green=(50,220,90); yellow=(255,210,60); red=(255,70,70); blue=(70,140,255)
    def box(rect,sx,sy,ox,col,label,w=3):
        x1,y1,x2,y2 = map(float,rect)
        r=(x1*sx+ox,y1*sy+head,x2*sx+ox,y2*sy+head)
        d.rectangle(r, outline=col, width=w); d.text((r[0]+2,r[1]+2),label,fill=col)
    for m in matches:
        i,j=m["before_idx"],m["after_idx"]
        drift=m["center_drift_pct_diag"]>cwarn or m["size_drift_pct_mean"]>swarn
        col=yellow if drift else green
        box(bb[i],bsx,bsy,0,col,f"B{i}")
        box(ab[j],asx,asy,b.width+gap,col,f"A{j}")
    for i in missing: box(bb[i],bsx,bsy,0,red,f"MISS B{i}",4)
    for j in added: box(ab[j],asx,asy,b.width+gap,blue,f"ADD A{j}",4)
    arr=np.asarray(cv,dtype=np.float32)/255.0
    return torch.from_numpy(arr)[None,...]

class RACompareSmartQCV08:
    RETURN_TYPES=("IMAGE","STRING","STRING")
    RETURN_NAMES=("qc_overlay","qc_report","qc_json")
    FUNCTION="compare"
    CATEGORY="RelateAnything/QC v0.8"
    OUTPUT_NODE=True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "before_image":("IMAGE",),"after_image":("IMAGE",),
            "before_regions":("RA_REGIONS",),"after_regions":("RA_REGIONS",),
            "row_tolerance":("FLOAT",{"default":0.02,"min":0.001,"max":0.2,"step":0.001}),
            "column_tolerance":("FLOAT",{"default":0.02,"min":0.001,"max":0.2,"step":0.001}),
            "weight_center":("FLOAT",{"default":4.0,"min":0.0,"max":20.0,"step":0.1}),
            "weight_iou":("FLOAT",{"default":1.5,"min":0.0,"max":20.0,"step":0.1}),
            "weight_size":("FLOAT",{"default":1.0,"min":0.0,"max":20.0,"step":0.1}),
            "weight_grid":("FLOAT",{"default":1.5,"min":0.0,"max":20.0,"step":0.1}),
            "weight_neighborhood":("FLOAT",{"default":1.0,"min":0.0,"max":20.0,"step":0.1}),
            "max_match_cost":("FLOAT",{"default":0.65,"min":0.01,"max":10.0,"step":0.01}),
            "warn_center_drift_pct":("FLOAT",{"default":1.0,"min":0.0,"max":20.0,"step":0.1}),
            "warn_size_drift_pct":("FLOAT",{"default":8.0,"min":0.0,"max":100.0,"step":0.5}),
            "fail_unmatched_fraction":("FLOAT",{"default":0.20,"min":0.0,"max":1.0,"step":0.01}),
            "order_tolerance":("FLOAT",{"default":0.01,"min":0.001,"max":0.2,"step":0.001})
        },"optional":{
            "before_relations_json":("STRING",{"forceInput":True}),
            "after_relations_json":("STRING",{"forceInput":True})
        }}

    def compare(self,before_image,after_image,before_regions,after_regions,row_tolerance,column_tolerance,weight_center,weight_iou,weight_size,weight_grid,weight_neighborhood,max_match_cost,warn_center_drift_pct,warn_size_drift_pct,fail_unmatched_fraction,order_tolerance,before_relations_json=None,after_relations_json=None):
        bb,bh,bw,bsrc=_regions(before_regions,"before_regions")
        ab,ah,aw,asrc=_regions(after_regions,"after_regions")
        bn,bc,bs=_norm(bb,bh,bw); an,ac,ass=_norm(ab,ah,aw)
        br,brg=_cluster(bc,1,row_tolerance); bcol,bcg=_cluster(bc,0,column_tolerance)
        ar,arg=_cluster(ac,1,row_tolerance); acol,acg=_cluster(ac,0,column_tolerance)
        C,det=_cost(bn,bc,bs,br,bcol,_neigh(bc),an,ac,ass,ar,acol,_neigh(ac),weight_center,weight_iou,weight_size,weight_grid,weight_neighborhood)
        rr,cc,method=_assign(C)
        matches=[]; ub=set(); ua=set(); diag=math.sqrt(2.0)
        for i,j in zip(rr,cc):
            if float(C[i,j])>max_match_cost: continue
            m=dict(det[(i,j)])
            m.update(before_idx=i,after_idx=j,before_source_idx=bsrc[i],after_source_idx=asrc[j],
                     center_drift_pct_diag=float(np.linalg.norm(ac[j]-bc[i])/diag*100.0),
                     size_drift_pct_mean=float(np.mean(np.abs(ass[j]-bs[i])/np.maximum(bs[i],1e-6))*100.0))
            matches.append(m); ub.add(i); ua.add(j)
        missing=[i for i in range(len(bb)) if i not in ub]
        added=[j for j in range(len(ab)) if j not in ua]
        frac=max(len(missing)/max(1,len(bb)),len(added)/max(1,len(ab)))
        det_status="FAIL" if frac>fail_unmatched_fraction else ("WARN" if missing or added else "PASS")
        st=_structure(matches,bc,ac,row_tolerance,column_tolerance,order_tolerance)
        cds=[m["center_drift_pct_diag"] for m in matches]; sds=[m["size_drift_pct_mean"] for m in matches]
        hard=bool(st["left_right_order_flips"] or st["above_below_order_flips"])
        gw=bool((max(cds) if cds else 0)>warn_center_drift_pct or (max(sds) if sds else 0)>warn_size_drift_pct or st["same_row_changes"] or st["same_column_changes"])
        geo_status="FAIL" if hard else ("WARN" if gw else "PASS")
        overall="FAIL" if "FAIL" in (det_status,geo_status) else ("WARN" if "WARN" in (det_status,geo_status) else "PASS")
        ov=_overlay(before_image,after_image,bb,ab,matches,missing,added,warn_center_drift_pct,warn_size_drift_pct)
        res={"version":VERSION,"overall_status":overall,"detection_status":det_status,"geometry_status":geo_status,"assignment_method":method,"scipy_available":HAS_SCIPY,"matched_count":len(matches),"missing_before_indices":missing,"added_after_indices":added,"unmatched_fraction":frac,"center_drift":{"mean_pct_diag":float(np.mean(cds)) if cds else 0.0,"max_pct_diag":max(cds) if cds else 0.0},"size_drift":{"mean_pct":float(np.mean(sds)) if sds else 0.0,"max_pct":max(sds) if sds else 0.0},"structure":st,"matches":matches}
        report="\n".join([
            f"ARCHVIZ QC v0.8 — {overall}",
            f"Detection: {det_status} | Geometry: {geo_status}",
            f"Assignment: {method}",
            f"Regions: BEFORE {len(bb)} → AFTER {len(ab)} | matched {len(matches)}",
            f"Missing {len(missing)} | Added {len(added)} | unmatched {frac*100:.1f}%",
            f"Rows {len(brg)}→{len(arg)} | Columns {len(bcg)}→{len(acg)}",
            f"Center drift mean {res['center_drift']['mean_pct_diag']:.3f}% | max {res['center_drift']['max_pct_diag']:.3f}%",
            f"Size drift mean {res['size_drift']['mean_pct']:.2f}% | max {res['size_drift']['max_pct']:.2f}%",
            f"Order flips LR {len(st['left_right_order_flips'])} | UD {len(st['above_below_order_flips'])}",
            f"Grid changes rows {len(st['same_row_changes'])} | columns {len(st['same_column_changes'])}",
            "Geometry truth is deterministic; RelateAnything semantics are advisory only."
        ])
        return {"ui":{"text":[report]},"result":(ov,report,json.dumps(res,ensure_ascii=False,indent=2))}

NODE_CLASS_MAPPINGS={"RACompareSmartQCV08":RACompareSmartQCV08}
NODE_DISPLAY_NAME_MAPPINGS={"RACompareSmartQCV08":"RA · SMART BEFORE vs AFTER QC · v0.8"}
