import json, math
import numpy as np
import torch
from PIL import Image, ImageDraw
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY=True
except Exception:
    linear_sum_assignment=None; HAS_SCIPY=False

VERSION='0.8.0'

def _regions(v,name):
    if isinstance(v,list) and len(v)==1 and isinstance(v[0],dict): v=v[0]
    if not isinstance(v,dict): raise ValueError(f'{name} must be RA_REGIONS')
    b=np.asarray(v['boxes'],dtype=np.float32)
    if b.size==0: b=b.reshape(0,4)
    h,w=int(v['height']),int(v['width'])
    s=v.get('source_indices',list(range(len(b))))
    return b,h,w,[int(x) for x in s]

def _norm(b,h,w):
    if len(b)==0: return b.astype(np.float32),np.zeros((0,2),np.float32),np.zeros((0,2),np.float32)
    n=b.astype(np.float32).copy(); n[:,[0,2]]/=w; n[:,[1,3]]/=h
    c=np.stack(((n[:,0]+n[:,2])/2,(n[:,1]+n[:,3])/2),1)
    s=np.stack((n[:,2]-n[:,0],n[:,3]-n[:,1]),1)
    return n,c,s

def _iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0.,x2-x1); ih=max(0.,y2-y1); inter=iw*ih
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]); bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    u=aa+bb-inter
    return float(inter/u) if u>0 else 0.

def _cluster(c,axis,tol):
    if len(c)==0:return np.zeros((0,),np.int32),[]
    order=np.argsort(c[:,axis]); groups=[]; cur=[int(order[0])]
    mean=float(c[cur[0],axis])
    for x in order[1:]:
        x=int(x); v=float(c[x,axis])
        if abs(v-mean)<=tol:
            cur.append(x); mean=float(np.mean([c[i,axis] for i in cur]))
        else:
            groups.append(cur); cur=[x]; mean=v
    groups.append(cur); lab=np.full(len(c),-1,np.int32)
    for gi,g in enumerate(groups):
        for i in g: lab[i]=gi
    return lab,groups

def _neigh(c,k=4):
    out=np.zeros((len(c),k),np.float32)
    for i in range(len(c)):
        d=np.sort(np.linalg.norm(c-c[i],axis=1)); d=d[d>1e-8][:k]
        vals=list(d)
        while len(vals)<k: vals.append(vals[-1] if vals else 0.)
        out[i]=vals
    return out

def _cost(bn,bc,bs,br,bcol,bnei,an,ac,ass,ar,acol,anei,wc,wi,ws,wg,wn):
    C=np.zeros((len(bn),len(an)),np.float32); detail={}; diag=math.sqrt(2.)
    brd=max(1,int(br.max()) if len(br) else 1); ard=max(1,int(ar.max()) if len(ar) else 1)
    bcd=max(1,int(bcol.max()) if len(bcol) else 1); acd=max(1,int(acol.max()) if len(acol) else 1)
    for i in range(len(bn)):
        for j in range(len(an)):
            cd=float(np.linalg.norm(bc[i]-ac[j])/diag); iou=_iou(bn[i],an[j])
            sz=float(np.mean(np.abs(np.log(np.maximum(ass[j],1e-6)/np.maximum(bs[i],1e-6)))))
            grid=.5*abs(float(br[i])/brd-float(ar[j])/ard)+.5*abs(float(bcol[i])/bcd-float(acol[j])/acd)
            nei=float(np.mean(np.abs(bnei[i]-anei[j])))
            val=wc*cd+wi*(1-iou)+ws*sz+wg*grid+wn*nei
            C[i,j]=val; detail[(i,j)]={'center_normdiag':cd,'iou':iou,'size_log_cost':sz,'grid_cost':grid,'neighborhood_cost':nei,'total_cost':float(val)}
    return C,detail

def _assign(C):
    if C.size==0:return [],[],'none'
    if HAS_SCIPY:
        r,c=linear_sum_assignment(C); return list(map(int,r)),list(map(int,c)),'hungarian/scipy'
    pairs=sorted((float(C[i,j]),i,j) for i in range(C.shape[0]) for j in range(C.shape[1])); ur=set(); uc=set(); r=[]; c=[]
    for _,i,j in pairs:
        if i in ur or j in uc: continue
        ur.add(i); uc.add(j); r.append(i); c.append(j)
    return r,c,'greedy/fallback'

"†&5¶²ÃÒÖ&5¶’ÃÒ“²GƒÖfÆöB†5¶ÂÃÒÖ5¶¢ÃÒ“²&G“ÖfÆöB†&5¶²ÃÒÖ&5¶’ÃÒ“²G“ÖfÆöB†5¶ÂÃÒÖ5¶¢ÃÒ¢–b'2†&G‚“æ÷&FW%÷FöÂæB'2†G‚“æ÷&FW%÷FöÂæBçç6–vâ†&G‚’Öçç6–vâ†G‚“¢Ç"æVæB…¶’ÆµÒ¢–b'2†&G’“æ÷&FW%÷FöÂæB'2†G’“æ÷&FW%÷FöÂæBçç6–vâ†&G’’Öçç6–vâ†G’“¢VBæVæB…¶’ÆµÒ¢–b†'2†&G’“Ã×&÷u÷FöÂ’Ò†'2†G’“Ã×&÷u÷FöÂ“¢&÷w2æVæB…¶’ÆµÒ¢–b†'2†&G‚“ÃÖ6öÅ÷FöÂ’Ò†'2†G‚“ÃÖ6öÅ÷FöÂ“¢6öÇ2æVæB…¶’ÆµÒ¢7æVæB†'2†fÆöB†çæÆ–æÆrææ÷&Ò†5¶ÅÒÖ5¶¥Ò’’ÖfÆöB†çæÆ–æÆrææ÷&Ò†&5¶µÒÖ&5¶•Ò’’’¢&WGW&â²vÆVgE÷&–v‡Eö÷&FW%öfÆ—2s¦Ç"Âv&÷fUö&VÆ÷uö÷&FW%öfÆ—2s§VBÂw6ÖU÷&÷uö6†ævW2s§&÷w2Âw6ÖUö6öÇVÖåö6†ævW2s¦6öÇ2Âw—%÷76–æuöG&–gEöÖVåöæ÷&Òs¦fÆöB†çæÖVâ‡7’’–b7VÇ6RâÂw—%÷76–æuöG&–gEöÖ…öæ÷&Òs¦fÆöB†çæÖ‚‡7’’–b7VÇ6RçÐ ¦FVb÷Fõ÷–Â‡B“ ¢–b—6–ç7Fæ6R‡BÆÆ—7B“¢C×E³Ð¢Öçæ6Æ—‡E³ÒæFWF6‚‚’æ7R‚’æfÆöB‚’æçV×’‚’ÃÃ“²Ò†£#SR²ãR’æ7G—R†ççV–çC‚¢–bç6†U²ÓÓÓÓC¦Ö²âââÃ£5Ð¢&WGW&â–ÖvRæg&öÖ'&’†Âu$t"r ¦FVböf—B†–ÖrÆ‚“ ¢–b–Öræ†V–v‡CÓÖƒ§&WGW&â–Öp¢sÖÖ‚ƒÇ&÷VæB†–Örçv–GF‚¦‚ö–Öræ†V–v‡B’“²&WGW&â–Örç&W6—¦R‚‡rÆ‚’Ä–ÖvRå&W6×Æ–æräÄä5¤õ2 ¦FVbö÷fW&Æ’†&–ÖrÆ–ÖrÆ&"Æ"ÆÖF6†W2ÆÖ—76–ærÆFFVBÆ7v&âÇ7v&â“ ¢#Õ÷Fõ÷–Â†&–Ör“²Õ÷Fõ÷–Â†–Ör“²ƒÓ“²#Õöf—B†#Ä‚“²Õöf—B†Ä‚“²vÓ#C²†VCÓC@¢7cÔ–ÖvRææWr‚u$t"rÂ†"çv–GF‚¶çv–GF‚¶vÄ‚¶†VB’Âƒ#"Ã#"Ã#"’“²7bç7FR†"ÂƒÆ†VB’“²7bç7FR†Â†"çv–GF‚¶vÆ†VB’“²CÔ–ÖvTG&räG&r†7b¢BçFW‡B‚ƒÃ"’Ât$Tdõ$RòtTôÔUE%’E%UD‚rÆf–ÆÃÒwv†—FRr“²BçFW‡B‚†"çv–GF‚¶v³Ã"’ÂteDU"ò’$U5TÅBrÆf–ÆÃÒwv†—FRr¢'7‚Æ'7“Ö"çv–GF‚ö#çv–GF‚Æ"æ†V–v‡Bö#æ†V–v‡C²7‚Æ7“Öçv–GF‚öçv–GF‚Ææ†V–v‡Böæ†V–v‡@¢FVb&÷‚‡&V7BÇ7‚Ç7’Æ÷‚Æ6öÆ÷"ÆÆ&VÂÇv–GFƒÓ2“ ¢ƒÇ“Çƒ"Ç“#ÖÖ†fÆöBÇ&V7B“²#Ò‡ƒ§7‚¶÷‚Ç“§7’¶†VBÇƒ"§7‚¶÷‚Ç“"§7’¶†VB“²Bç&V7FævÆR‡"Æ÷WFÆ–æSÖ6öÆ÷"Çv–GFƒ×v–GF‚“²BçFW‡B‚‡%³Ò³"Ç%³Ò³"’ÆÆ&VÂÆf–ÆÃÖ6öÆ÷"¢f÷"Ò–âÖF6†W3 ¢’Æ£ÖÕ²v&Vf÷&Uö–G‚uÒÆÕ²vgFW%ö–G‚uÓ²G&–gCÖÕ²v6VçFW%öG&–gE÷7EöF–ruÓæ7v&â÷"Õ²w6—¦UöG&–gE÷7EöÖVâuÓç7v&ã²6öÃÒƒ#SRÃ#Ãc’–bG&–gBVÇ6RƒSÃ##Ã“¢&÷‚†&%¶•ÒÆ'7‚Æ'7’ÃÆ6öÂÆbt'¶—Òr“²&÷‚†%¶¥ÒÆ7‚Æ7’Æ"çv–GF‚¶vÆ6öÂÆbt¶§Òr¢'ƒÒ‚†&%¶•Õ³Ò¶&%¶•Õ³%Ò’ó"’¦'7ƒ²'“Ò‚†&%¶•Õ³Ò¶&%¶•Õ³5Ò’ó"’¦'7’¶†VC²ƒÒ‚†%¶¥Õ³Ò¶%¶¥Õ³%Ò’ó"’¦7‚¶"çv–GF‚¶v²“Ò‚†%¶¥Õ³Ò¶%¶¥Õ³5Ò’ó"’¦7’¶†VC²BæÆ–æR‚†'‚Æ'’Æ‚Æ’’Æf–ÆÃÖ6öÂÇv–GFƒÓ¢f÷"’–âÖ—76–æs¢&÷‚†&%¶•ÒÆ'7‚Æ'7’ÃÂƒ#SRÃsÃs’ÆbtÔ•52'¶—ÒrÃB¢f÷"¢–âFFVC¢&÷‚†%¶¥ÒÆ7‚Æ7’Æ"çv–GF‚¶vÂƒsÃCÃ#SR’ÆbtDB¶§ÒrÃB¢'#Öçæ6'&’†7bÆGG—SÖçæfÆöC3"’ó#SRã²&WGW&âF÷&6‚æg&öÕöçV×’†'"•´æöæRÂââåÐ ¦FVb÷6VÒ†"Æ“ ¢FVb'6R‡‚“ ¢–bæ÷Bƒ§&WGW&âæöæP¢–b—6–ç7Fæ6R‡‚ÆÆ—7B“¢ƒ×…³Ò–b‚VÇ6Rrp¢–bæ÷B7G"‡‚’ç7G&—‚“§&WGW&âæöæP¢#Ö§6öâæÆöG2‡7G"‡‚’’ævWB‚w&VÆF–öç2r“²&WGW&â"–b—6–ç7Fæ6R‡"ÆÆ—7B’VÇ6RæöæP¢G'“¢'#×'6R†"“²#×'6R†¢W†6WBW†6WF–öâ2S§&WGW&â²vf–Æ&ÆRs¤fÇ6RÂw&V6öâs§7G"†R—Ð¢–b'"—2æöæR÷""—2æöæS§&WGW&â²vf–Æ&ÆRs¤fÇ6RÂw&V6öâs¢w&VÆF–öç5ö§6öâæ÷B6öææV7FVB÷"V×G’wÐ¢6–sÖÆÖ&F#¢†–çB‡"ævWB‚w7V&¦V7E÷6÷W&6Uö–G‚rÂÓ’’Ç7G"‡"ævWB‚w&VF–6FRrÂrr’’Æ–çB‡"ævWB‚vö&¦V7E÷6÷W&6Uö–G‚rÂÓ’’¢'3×·6–r‡"’f÷""–â''Ó²73×·6–r‡"’f÷""–â'Ð¢&WGW&â²vf–Æ&ÆRs¥G'VRÂvæ÷FRs¢vGf—6÷'’öæÇ’rÂw&VÖ÷fVBs¥¶Æ—7B‡‚’f÷"‚–â6÷'FVB†'2Ö72•ÒÂvFFVBs¥¶Æ—7B‡‚’f÷"‚–â6÷'FVB†72Ö'2•×Ð ¦6Æ72$6ö×&U6Ö'E5cƒ ¢$UEU$åõE•U3Ò‚t”ÔtRrÂu5E$”ärrÂu5E$”ärr“²$UEU$åôäÔU3Ò‚w5ö÷fW&Æ’rÂw5÷&W÷'BrÂw5ö§6öâr“²eTä5D”ôãÒv6ö×&Rs²4DTtõ%“Òu&VÆFTç—F†–ærõ2cã‚s²õUEUEôäôDSÕG'VP¢6Æ76ÖWF†ö@¢FVb”åUEõE•U2†6Ç2“ ¢&WGW&â²w&WV—&VBs§°¢v&Vf÷&Uö–ÖvRs¢‚t”ÔtRrÂ’ÂvgFW%ö–ÖvRs¢‚t”ÔtRrÂ’Âv&Vf÷&U÷&Vv–öç2s¢‚u$õ$Tt”ôå2rÂ’ÂvgFW%÷&Vv–öç2s¢‚u$õ$Tt”ôå2rÂ’À¢w&÷u÷FöÆW&æ6Rs¢‚tdÄôBrÇ²vFVfVÇBs£ã"ÂvÖ–âs£ãÂvÖ‚s£ã"Âw7FWs£ãÒ’Âv6öÇVÖå÷FöÆW&æ6Rs¢‚tdÄôBrÇ²vFVfVÇBs£ã"ÂvÖ–âs£ãÂvÖ‚s£ã"Âw7FWs£ãÒ’À¢wvV–v‡Eö6VçFW"s¢‚tdÄôBrÇ²vFVfVÇBs£BãÂvÖ–âs£âÂvÖ‚s£#âÂw7FWs£ãÒ’ÂwvV–v‡Eö–÷Rs¢‚tdÄôBrÇ²vFVfVÇBs£ãRÂvÖ–âs£âÂvÖ‚s£#âÂw7FWs£ãÒ’ÂwvV–v‡E÷6—¦Rs¢‚tdÄôBrÇ²vFVfVÇBs£ãÂvÖ–âs£âÂvÖ‚s£#âÂw7FWs£ãÒ’ÂwvV–v‡Eöw&–Bs¢‚tdÄôBrÇ²vFVfVÇBs£ãRÂvÖ–âs£âÂvÖ‚s£#âÂw7FWs£ãÒ’ÂwvV–v‡EöæV–v†&÷&†ööBs¢‚tdÄôBrÇ²vFVfVÇBs£ãÂvÖ–âs£âÂvÖ‚s£#âÂw7FWs£ãÒ’À¢vÖ…öÖF6…ö6÷7Bs¢‚tdÄôBrÇ²vFVfVÇBs£ãcRÂvÖ–âs£ãÂvÖ‚s£âÂw7FWs£ãÒ’Âwv&åö6VçFW%öG&–gE÷7Bs¢‚tdÄôBrÇ²vFVfVÇBs£ãÂvÖ–âs£âÂvÖ‚s£#âÂw7FWs£ãÒ’Âwv&å÷6—¦UöG&–gE÷7Bs¢‚tdÄôBrÇ²vFVfVÇBs£‚ãÂvÖ–âs£âÂvÖ‚s£âÂw7FWs£ãWÒ’Âvf–Å÷VæÖF6†VEög&7F–öâs¢‚tdÄôBrÇ²vFVfVÇBs£ã#ÂvÖ–âs£âÂvÖ‚s£âÂw7FWs£ãÒ’Âv÷&FW%÷FöÆW&æ6Rs¢‚tdÄôBrÇ²vFVfVÇBs£ãÂvÖ–âs£ãÂvÖ‚s£ã"Âw7FWs£ãÒ—ÒÀ¢v÷F–öæÂs§²v&Vf÷&U÷&VÆF–öç5ö§6öâs¢‚u5E$”ärrÇ²vf÷&6T–çWBs¥G'VWÒ’ÂvgFW%÷&VÆF–öç5ö§6öâs¢‚u5E$”ärrÇ²vf÷&6T–çWBs¥G'VWÒ—×Ð¢FVb6ö×&R‡6VÆbÆ&Vf÷&Uö–ÖvRÆgFW%ö–ÖvRÆ&Vf÷&U÷&Vv–öç2ÆgFW%÷&Vv–öç2Ç&÷u÷FöÆW&æ6RÆ6öÇVÖå÷FöÆW&æ6RÇvV–v‡Eö6VçFW"ÇvV–v‡Eö–÷RÇvV–v‡E÷6—¦RÇvV–v‡Eöw&–BÇvV–v‡EöæV–v†&÷&†ööBÆÖ…öÖF6…ö6÷7BÇv&åö6VçFW%öG&–gE÷7BÇv&å÷6—¦UöG&–gE÷7BÆf–Å÷VæÖF6†VEög&7F–öâÆ÷&FW%÷FöÆW&æ6RÆ&Vf÷&U÷&VÆF–öç5ö§6öãÔæöæRÆgFW%÷&VÆF–öç5ö§6öãÔæöæR“ ¢&"Æ&‚Æ'rÆ'7&3Õ÷&Vv–öç2†&Vf÷&U÷&Vv–öç2Âv&Vf÷&Rr“²"Æ‚ÆrÆ7&3Õ÷&Vv–öç2†gFW%÷&Vv–öç2ÂvgFW"r“²&âÆ&2Æ'3Õöæ÷&Ò†&"Æ&‚Æ'r“²âÆ2Æ73Õöæ÷&Ò†"Æ‚Ær¢'"Æ'&sÕö6ÇW7FW"†&2ÃÇ&÷u÷FöÆW&æ6R“²&6öÂÆ&6sÕö6ÇW7FW"†&2ÃÆ6öÇVÖå÷FöÆW&æ6R“²"Æ&sÕö6ÇW7FW"†2ÃÇ&÷u÷FöÆW&æ6R“²6öÂÆ6sÕö6ÇW7FW"†2ÃÆ6öÇVÖå÷FöÆW&æ6R¢2ÆFWCÕö6÷7B†&âÆ&2Æ'2Æ'"Æ&6öÂÅöæV–v‚†&2’ÆâÆ2Æ72Æ"Æ6öÂÅöæV–v‚†2’ÇvV–v‡Eö6VçFW"ÇvV–v‡Eö–÷RÇvV–v‡E÷6—¦RÇvV–v‡Eöw&–BÇvV–v‡EöæV–v†&÷&†ööB“²'"Æ62ÆÖWF†öCÕö76–vâ„2¢ÖF6†W3ÕµÓ²V#×6WB‚“²V×6WB‚“²F–sÖÖF‚ç7'Bƒ"â¢f÷"’Æ¢–â¦—‡'"Æ62“ ¢–bfÆöB„5¶’Æ¥Ò“æÖ…öÖF6…ö6÷7C¦6öçF–çVP¢ÓÖF–7B†FWE²†’Æ¢•Ò“²ÒçWFFR†&Vf÷&Uö–GƒÖ’ÆgFW%ö–GƒÖ¢Æ&Vf÷&U÷6÷W&6Uö–GƒÖ'7&5¶•ÒÆgFW%÷6÷W&6Uö–GƒÖ7&5¶¥ÒÆ&Vf÷&U÷&÷sÖ–çB†'%¶•Ò’ÆgFW%÷&÷sÖ–çB†%¶¥Ò’Æ&Vf÷&Uö6öÃÖ–çB†&6öÅ¶•Ò’ÆgFW%ö6öÃÖ–çB†6öÅ¶¥Ò’Æ6VçFW%öG&–gE÷7EöF–sÖfÆöB†çæÆ–æÆrææ÷&Ò†5¶¥ÒÖ&5¶•Ò’öF–r£’Ç6—¦UöG&–gE÷7EöÖVãÖfÆöB†çæÖVâ†çæ'2†75¶¥ÒÖ'5¶•Ò’öçæÖ†–×VÒ†'5¶•ÒÃRÓb’’£’“²ÖF6†W2æVæB†Ò“²V"æFB†’“²VæFB†¢¢Ö—76–æsÕ¶’f÷"’–â&ævR†ÆVâ†&"’’–b’æ÷B–âV%Ó²FFVCÕ¶¢f÷"¢–â&ævR†ÆVâ†"’’–b¢æ÷B–âVÓ²7CÕ÷7G'V7GW&R†ÖF6†W2Æ&2Æ2Ç&÷u÷FöÆW&æ6RÆ6öÇVÖå÷FöÆW&æ6RÆ÷&FW%÷FöÆW&æ6R¢6G3Õ¶Õ²v6VçFW%öG&–gE÷7EöF–ruÒf÷"Ò–âÖF6†W5Ó²6G3Õ¶Õ²w6—¦UöG&–gE÷7EöÖVâuÒf÷"Ò–âÖF6†W5Ó²g&3ÖÖ‚†ÆVâ†Ö—76–ær’öÖ‚ƒÆÆVâ†&"’’ÆÆVâ†FFVB’öÖ‚ƒÆÆVâ†"’’¢FWE÷7FGW3Òtd”Âr–bg&3æf–Å÷VæÖF6†VEög&7F–öâVÇ6R‚ut$âr–bÖ—76–ær÷"FFVBVÇ6Ru52r“²†&CÖ&ööÂ‡7E²vÆVgE÷&–v‡Eö÷&FW%öfÆ—2uÒ÷"7E²v&÷fUö&VÆ÷uö÷&FW%öfÆ—2uÒ“²wsÖ&ööÂ‚†Ö‚†6G2’–b6G2VÇ6R“çv&åö6VçFW%öG&–gE÷7B÷"†Ö‚‡6G2’–b6G2VÇ6R“çv&å÷6—¦UöG&–gE÷7B÷"7E²w6ÖU÷&÷uö6†ævW2uÒ÷"7E²w6ÖUö6öÇVÖåö6†ævW2uÒ“²vVõ÷7FGW3Òtd”Âr–b†&BVÇ6R‚ut$âr–bwrVÇ6Ru52r“²÷fW&ÆÃÒtd”Âr–btd”Âr–â†FWE÷7FGW2ÆvVõ÷7FGW2’VÇ6R‚ut$âr–but$âr–â†FWE÷7FGW2ÆvVõ÷7FGW2’VÇ6Ru52r¢6VÓÕ÷6VÒ†&Vf÷&U÷&VÆF–öç5ö§6öâÆgFW%÷&VÆF–öç5ö§6öâ“²÷cÕö÷fW&Æ’†&Vf÷&Uö–ÖvRÆgFW%ö–ÖvRÆ&"Æ"ÆÖF6†W2ÆÖ—76–ærÆFFVBÇv&åö6VçFW%öG&–gE÷7BÇv&å÷6—¦UöG&–gE÷7B¢&W3×²wfW'6–öâs¥dU%4”ôâÂv÷fW&ÆÅ÷7FGW2s¦÷fW&ÆÂÂvFWFV7F–öå÷7FGW2s¦FWE÷7FGW2ÂvvVöÖWG'•÷7FGW2s¦vVõ÷7FGW2Âv76–væÖVçEöÖWF†öBs¦ÖWF†öBÂw66—•öf–Æ&ÆRs¤„5õ44•’Âv&Vf÷&Rs§²v6÷VçBs¦ÆVâ†&"’Âw&÷w2s¦ÆVâ†'&r’Âv6öÇVÖç2s¦ÆVâ†&6r—ÒÂvgFW"s§²v6÷VçBs¦ÆVâ†"’Âw&÷w2s¦ÆVâ†&r’Âv6öÇVÖç2s¦ÆVâ†6r—ÒÂvÖF6†VEö6÷VçBs¦ÆVâ†ÖF6†W2’ÂvÖ—76–æuö&Vf÷&Uö–æF–6W2s¦Ö—76–ærÂvFFVEögFW%ö–æF–6W2s¦FFVBÂwVæÖF6†VEög&7F–öâs¦g&2Âv6VçFW%öG&–gBs§²vÖVå÷7EöF–rs¦fÆöB†çæÖVâ†6G2’’–b6G2VÇ6RâÂvÖ…÷7EöF–rs¦Ö‚†6G2’–b6G2VÇ6RçÒÂw6—¦UöG&–gBs§²vÖVå÷7Bs¦fÆöB†çæÖVâ‡6G2’’–b6G2VÇ6RâÂvÖ…÷7Bs¦Ö‚‡6G2’–b6G2VÇ6RçÒÂw7G'V7GW&Rs§7BÂvÖF6†W2s¦ÖF6†W2Âw6VÖçF–5÷&VÆF–öç2s§6V×Ð¢Æ–æW3Õ¶bt$4…d•¢2cã‚(	B¶÷fW&ÆÇÒrÆbtFWFV7F–öã¢¶FWE÷7FGW7ÒÂvVöÖWG'“¢¶vVõ÷7FGW7ÒrÆbt76–væÖVçC¢¶ÖWF†öGÒrÆbu&Vv–öç3¢$Tdõ$R¶ÆVâ†&"—Ò(i"eDU"¶ÆVâ†"—ÒÂÖF6†VB¶ÆVâ†ÖF6†W2—ÒrÆbtÖ—76–ær¶ÆVâ†Ö—76–ær—ÒÂFFVB¶ÆVâ†FFVB—ÒÂVæÖF6†VB¶g&2£¢ãgÒRrÆbu&÷w2¶ÆVâ†'&r—Þ(i'¶ÆVâ†&r—ÒÂ6öÇVÖç2¶ÆVâ†&6r—Þ(i'¶ÆVâ†6r—ÒrÆbt6VçFW"G&–gBÖVâ·&W5²&6VçFW%öG&–gB%Õ²&ÖVå÷7EöF–r%Ó¢ã6gÒRÂÖ‚·&W5²&6VçFW%öG&–gB%Õ²&Ö…÷7EöF–r%Ó¢ã6gÒRrÆbu6—¦RG&–gBÖVâ·&W5²'6—¦UöG&–gB%Õ²&ÖVå÷7B%Ó¢ã&gÒRÂÖ‚·&W5²'6—¦UöG&–gB%Õ²&Ö…÷7B%Ó¢ã&gÒRrÆbt÷&FW"fÆ—2Å"¶ÆVâ‡7E²&ÆVgE÷&–v‡Eö÷&FW%öfÆ—2%Ò—ÒÂTB¶ÆVâ‡7E²&&÷fUö&VÆ÷uö÷&FW%öfÆ—2%Ò—ÒrÆbtw&–B6†ævW2&÷w2¶ÆVâ‡7E²'6ÖU÷&÷uö6†ævW2%Ò—ÒÂ6öÇVÖç2¶ÆVâ‡7E²'6ÖUö6öÇVÖåö6†ævW2%Ò—ÒrÂt÷fW&Æ“¢u$TTâÖF6†VBÂ”TÄÄõrG&–gFVBÂ$TBÖ—76–ærÂ$ÅTRFFVBrÂtvVöÖWG'’G'WF‚—2FWFW&Ö–æ—7F–3²$6VÖçF–72Gf—6÷'’öæÇ’âuÐ¢&W÷'CÒuÆâræ¦ö–â†Æ–æW2“²–ÆöCÖ§6öâæGV×2‡&W2ÆVç7W&Uö66–“ÔfÇ6RÆ–æFVçCÓ"“²&WGW&â²wV’s§²wFW‡Bs¥·&W÷'E×ÒÂw&W7VÇBs¢†÷bÇ&W÷'BÇ–ÆöB—Ð ¤äôDUô4Ä55ôÔ”äu3×²u$6ö×&U6Ö'E5c‚s¥$6ö×&U6Ö'E5c‡Ð¤äôDUôD•5Ä•ôäÔUôÔ”äu3×²u$6ö×&U6Ö'E5c‚s¢u$+r4Ô%B$Tdõ$Rg2eDU"2+rcã‚wÐ