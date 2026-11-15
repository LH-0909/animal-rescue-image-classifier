from pathlib import Path
import requests,json,time,os
D=Path(__file__).resolve().parent/'data';s=requests.Session();base='http://127.0.0.1:5088/api/'
name=os.environ.get('NEURAL_PLATFORM_USER');password=os.environ.get('NEURAL_PLATFORM_PASSWORD')
if not name or not password:raise RuntimeError('Set NEURAL_PLATFORM_USER and NEURAL_PLATFORM_PASSWORD before training.')
s.post(base+'auth',json={'name':name,'password':password}).raise_for_status()
r=s.post(base+'evaluate',json={'model':'d9e8bc607e'});r.raise_for_status();(D/'FCN-evaluation.json').write_text(json.dumps(r.json(),ensure_ascii=False,indent=2),'utf-8');print('FCN EVALUATION',r.json(),flush=True)
p={'mode':'CNN','width':32,'height':32,'channels':3,'k':3,'hidden':[128],'activation':'ReLU','loss':'CrossEntropy','optimizer':'AdamW','lr':.001,'batch':128,'epochs':40,'decay':.0001,'dropoutOn':True,'dropout':.3,'early':True,'patience':8,'dataset':'cifar10-pets','normalize':True,'augment':True}
r=s.post(base+'train',json=p);r.raise_for_status();last=-1
while True:
 j=s.get(base+'state').json()['job']
 if j['epoch']!=last:print(json.dumps({k:v for k,v in j.items() if k not in ['config','curves']},ensure_ascii=False),flush=True);last=j['epoch']
 if j['status'] in ['completed','error','stopped']:break
 time.sleep(2)
if j['status']!='completed':raise RuntimeError(j)
r=s.post(base+'evaluate',json={'model':j['model_id']});r.raise_for_status();(D/'CNN-evaluation.json').write_text(json.dumps(r.json(),ensure_ascii=False,indent=2),'utf-8');print('CNN EVALUATION',r.json(),flush=True)
