"""Reproducible official CIFAR-10 cat/dog/bird experiment. No test data enters fitting."""
from pathlib import Path
import hashlib,tarfile,pickle,json,sqlite3,time,io,random
import numpy as np
from PIL import Image
import requests
import os
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data';archive=DATA/'cifar-10-python.tar.gz'
assert hashlib.md5(archive.read_bytes()).hexdigest()=='c58f30108f718f92721af3b95e74349a','Official checksum mismatch'
labels={2:'鸟',3:'猫',5:'狗'};seen=set();items=[];folder=DATA/'images';folder.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
 for part in ['test_batch']+['data_batch_'+str(i) for i in range(1,6)]:
  obj=pickle.load(tar.extractfile('cifar-10-batches-py/'+part),encoding='bytes');arr=obj[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
  for a,y in zip(arr,obj[b'labels']):
   if y not in labels:continue
   digest=hashlib.sha256(a.tobytes()+b'(32, 32)').hexdigest()
   if digest in seen:continue
   seen.add(digest);name=digest+'.png';Image.fromarray(a).save(folder/name)
   items.append({'id':name,'label':labels[y],'split':'test' if part=='test_batch' else 'train'})
for label in labels.values():
 group=sorted([i for i in items if i['label']==label and i['split']=='train'],key=lambda x:x['id']);random.Random(42).shuffle(group)
 for i in group[:500]:i['split']='val'
conn=sqlite3.connect(DATA/'platform.db');conn.execute('INSERT OR REPLACE INTO datasets VALUES(?,?,?)',('cifar10-pets','CIFAR-10 · 猫 / 狗 / 鸟',json.dumps(items,ensure_ascii=False)));conn.commit();conn.close()
counts={s:sum(i['split']==s for i in items) for s in ['train','val','test']}
provenance={'source':'https://www.cs.toronto.edu/~kriz/cifar.html','download':'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz','md5':'c58f30108f718f92721af3b95e74349a','classes':labels,'resolution':[32,32,3],'split':counts,'method':'Official test split preserved; validation 500/class sampled only from official training set; exact pixel duplicates removed globally, test prioritized; seed 42. Near duplicates remain possible.'}
(DATA/'dataset-source.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),'utf-8');print('DATASET',counts,flush=True)
s=requests.Session();base='http://127.0.0.1:5088/api/'
name=os.environ.get('NEURAL_PLATFORM_USER');password=os.environ.get('NEURAL_PLATFORM_PASSWORD')
if not name or not password:raise RuntimeError('Set NEURAL_PLATFORM_USER and NEURAL_PLATFORM_PASSWORD before training.')
p={'name':name,'password':password,'register':False};r=s.post(base+'auth',json=p)
r.raise_for_status()
for mode,epochs in [('FCN',15),('CNN',40)]:
 config={'mode':mode,'width':32,'height':32,'channels':3,'k':3,'hidden':[256,128] if mode=='FCN' else [128],'activation':'ReLU','loss':'CrossEntropy','optimizer':'AdamW','lr':.001,'batch':128,'epochs':epochs,'decay':.0001,'dropoutOn':True,'dropout':.3,'early':True,'patience':8,'dataset':'cifar10-pets','normalize':True,'augment':True}
 r=s.post(base+'train',json=config);r.raise_for_status();last=-1
 while True:
  j=s.get(base+'state').json()['job']
  if j['epoch']!=last:print(mode,json.dumps({k:v for k,v in j.items() if k not in ['curves','config']},ensure_ascii=False),flush=True);last=j['epoch']
  if j['status'] in ['completed','error','stopped']:break
  time.sleep(2)
 if j['status']!='completed':raise RuntimeError(j)
 r=s.post(base+'evaluate',json={'model':j['model_id']});r.raise_for_status();(DATA/(mode+'-evaluation.json')).write_text(json.dumps(r.json(),ensure_ascii=False,indent=2),'utf-8');print('EVALUATION',mode,r.json(),flush=True)
print('ALL DONE',flush=True)
