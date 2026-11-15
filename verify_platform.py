import tempfile,io,zipfile,time,json
from pathlib import Path
from PIL import Image
import app as m
with tempfile.TemporaryDirectory() as tmp:
 m.DATA=Path(tmp);m.init();m.job={'status':'idle','epoch':0,'curves':[]};c=m.app.test_client()
 assert c.post('/api/train',json={}).status_code==401
 assert c.post('/api/auth',json={'name':'test','password':'testpass','register':True}).status_code==200
 z=io.BytesIO()
 with zipfile.ZipFile(z,'w') as ar:
  for k in range(2):
   for i in range(8):
    im=Image.new('RGB',(12,12),(k*180+i*5,30+i*3,60));b=io.BytesIO();im.save(b,'PNG');ar.writestr(f'class{k}/{i}.png',b.getvalue())
 r=c.post('/api/datasets',data={'name':'testset','files':(io.BytesIO(z.getvalue()),'set.zip')});assert r.status_code==200,r.json
 ds=c.get('/api/state').json['datasets'][0];assert ds['count']==16 and ds['splits']=={'train':12,'val':2,'test':2}
 r=c.post('/api/preprocess',data={'file':(io.BytesIO(b'bad'),'x.png')});assert r.status_code==400
 cfg={'mode':'FCN','width':8,'height':8,'channels':3,'k':2,'hidden':[4],'activation':'ReLU','loss':'CrossEntropy','optimizer':'Adam','lr':.001,'batch':4,'epochs':2,'decay':0,'dropoutOn':False,'dropout':.2,'early':False,'patience':2,'dataset':ds['id'],'normalize':True,'augment':True}
 for mode in ['FCN','CNN']:
  cfg['mode']=mode;r=c.post('/api/train',json=cfg);assert r.status_code==200,r.json
  for _ in range(300):
   if m.job['status'] not in ['running','paused']:break
   time.sleep(.05)
  assert m.job['status']=='completed',m.job
  ident=m.job['model_id'];r=c.post('/api/evaluate',json={'model':ident});assert r.status_code==200,r.json;assert r.json['samples']==2
  ims=json.loads(m.rows('datasets')[0]['images']);test=next(i for i in ims if i['split']=='test')
  r=c.post('/api/predict',data={'model':ident,'files':(io.BytesIO((m.DATA/'images'/test['id']).read_bytes()),'image.png')});assert r.status_code==200,r.json;assert abs(sum(p[1] for p in r.json['results'][0]['probabilities'])-1)<1e-5
  assert c.get('/api/models/'+ident+'/download').status_code==200
 assert c.post('/api/datasets/'+ds['id']+'/label',json={'old':'class0','name':'new'}).status_code==400
 assert c.delete('/api/datasets/'+ds['id']).status_code==400
 assert len(m.rows('models'))==2
 print('PASS: auth, ZIP import, stratified split, bad-image validation, FCN and CNN train/save/load/predict/evaluate, model download, bound dataset protection')
