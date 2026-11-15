from flask import Flask,request,jsonify,send_file,session
from pathlib import Path
from PIL import Image,ImageOps,ImageFilter,UnidentifiedImageError
import sqlite3,json,uuid,hashlib,io,zipfile,threading,time,os,random
import numpy as np
import torch
from torch import nn
from werkzeug.security import generate_password_hash,check_password_hash

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
app=Flask(__name__); app.secret_key=os.environ.get('APP_SECRET') or ((DATA/'secret').read_text() if (DATA/'secret').exists() else os.urandom(32).hex())
if not (DATA/'secret').exists(): (DATA/'secret').write_text(app.secret_key)
app.config.update(MAX_CONTENT_LENGTH=100*1024*1024,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict')
torch.set_num_threads(2)
lock=threading.Lock(); job={'status':'idle','epoch':0,'curves':[],'message':'等待开始训练'}
def db():
 c=sqlite3.connect(DATA/'platform.db');c.row_factory=sqlite3.Row;return c
def init():
 with db() as c:
  c.executescript('CREATE TABLE IF NOT EXISTS users(name TEXT PRIMARY KEY,password TEXT); CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT,images TEXT); CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY,name TEXT,config TEXT,classes TEXT,metrics TEXT,created TEXT,dataset TEXT); CREATE TABLE IF NOT EXISTS history(id TEXT PRIMARY KEY,type TEXT,name TEXT,result TEXT,created TEXT); CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY,filename TEXT,image_id TEXT,model TEXT,predicted TEXT,probabilities TEXT,correct_label TEXT,note TEXT,status TEXT,created TEXT);')
init()
if (DATA/'last-job.json').exists():job=json.loads((DATA/'last-job.json').read_text('utf-8'))
def rows(table):
 with db() as c:return [dict(r) for r in c.execute('SELECT * FROM '+table)]
def record(kind,name,result):
 with db() as c:c.execute('INSERT INTO history VALUES(?,?,?,?,?)',(uuid.uuid4().hex[:8],kind,name,json.dumps(result,ensure_ascii=False),time.strftime('%Y-%m-%d %H:%M:%S')))
@app.before_request
def auth():
 if request.path.startswith('/api/') and request.path not in ['/api/auth','/api/session','/api/state','/api/predict','/api/evaluate','/api/preprocess'] and not request.path.startswith('/api/image/') and not session.get('user'):return jsonify(error='请先登录或注册后使用真实数据功能'),401
@app.errorhandler(Exception)
def error(e):
 from werkzeug.exceptions import HTTPException
 return jsonify(error=e.description if isinstance(e,HTTPException) else str(e)), e.code if isinstance(e,HTTPException) else 400
@app.get('/')
def index():return app.send_static_file('index.html')
@app.route('/api/auth',methods=['POST','DELETE'])
def login():
 if request.method=='DELETE':session.clear();return jsonify(ok=True)
 p=request.json;name=p.get('name','').strip();password=p.get('password','')
 if len(name)<2 or len(password)<6:raise ValueError('用户名至少 2 个字符，密码至少 6 位')
 with db() as c:
  u=c.execute('SELECT * FROM users WHERE name=?',(name,)).fetchone()
  if p.get('register'):
   if u:raise ValueError('用户名已存在')
   c.execute('INSERT INTO users VALUES(?,?)',(name,generate_password_hash(password)))
  elif not u or not check_password_hash(u['password'],password):raise ValueError('用户名或密码错误')
 session['user']=name;return jsonify(user=name)
@app.get('/api/session')
def who():return jsonify(user=session.get('user'))
@app.get('/api/state')
def state():
 logged=bool(session.get('user'));ds=rows('datasets')
 for d in ds:
  ims=json.loads(d.pop('images'));d.update(count=len(ims),class_counts={label:sum(i['label']==label for i in ims) for label in set(i['label'] for i in ims)},classes=sorted(set(i['label'] for i in ims)),splits={s:sum(i['split']==s for i in ims) for s in ['train','val','test']},preview=[i['id'] for i in ims[:5]])
 reviews=rows('reviews') if logged else []
 return jsonify(datasets=ds,models=rows('models'),history=rows('history')[::-1],reviews=reviews[::-1],job=job)
def valid_image(raw):
 if len(raw)>10*1024*1024:raise ValueError('单张图片不能超过 10 MB')
 im=Image.open(io.BytesIO(raw));im.verify();im=Image.open(io.BytesIO(raw));
 if im.format not in ['JPEG','PNG','BMP']:raise ValueError('仅支持 JPG、PNG、BMP')
 if im.width*im.height>25000000:raise ValueError('图片像素过大')
 return ImageOps.exif_transpose(im).convert('RGB')
@app.post('/api/datasets')
def upload_dataset():
 name=request.form.get('name','新建数据集').strip()[:100];items=[];seen=set();duplicates=0
 for f in request.files.getlist('files'):
  raw=f.read()
  if f.filename.lower().endswith('.zip'):
   with zipfile.ZipFile(io.BytesIO(raw)) as z:
    entries=z.infolist()
    if len(entries)>10000 or sum(e.file_size for e in entries)>500*1024*1024:raise ValueError('压缩包最多 10000 文件、解压总大小 500 MB')
    for e in entries:
     parts=Path(e.filename.replace('\\','/')).parts
     if e.is_dir() or Path(e.filename).suffix.lower() not in ['.jpg','.jpeg','.png','.bmp']:continue
     if len(parts)<2:raise ValueError('ZIP 请按 类别名/图片.jpg 组织')
     items.append((parts[-2],z.read(e)))
  else: items.append((request.form.get('label','未分类'),raw))
 if not items:raise ValueError('没有有效图片')
 stored=[];folder=DATA/'images';folder.mkdir(exist_ok=True)
 for label,raw in items:
  im=valid_image(raw);digest=hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest()
  if digest in seen:duplicates+=1;continue
  seen.add(digest);ident=digest+'.jpg';im.save(folder/ident)
  stored.append({'id':ident,'label':label,'split':'train'})
 # Stratified and deterministic. Identical decoded images deduplicated before partitioning.
 for label in set(i['label'] for i in stored):
  group=sorted([i for i in stored if i['label']==label],key=lambda x:x['id']);random.Random(42).shuffle(group)
  if len(group)<5:raise ValueError(f'类别 {label} 去重后至少需要 5 张图片，以保留独立验证和测试样本')
  n=max(1,round(len(group)*.1))
  for idx,i in enumerate(group):i['split']='test' if idx<n else 'val' if idx<2*n else 'train'
 ident=uuid.uuid4().hex[:10]
 with db() as c:c.execute('INSERT INTO datasets VALUES(?,?,?)',(ident,name,json.dumps(stored,ensure_ascii=False)))
 record('数据导入',name,{'images':len(stored),'duplicates':duplicates});return jsonify(ok=True,duplicates=duplicates)
@app.get('/api/image/<ident>')
def image(ident):return send_file(DATA/'images'/Path(ident).name)
@app.delete('/api/datasets/<ident>')
def delete_dataset(ident):
 if job['status'] in ['running','paused']:raise ValueError('训练进行中，无法删除数据集')
 if any(m['dataset']==ident for m in rows('models')):raise ValueError('已有模型绑定此数据集，请保留以保证独立测试可复现')
 with db() as c:c.execute('DELETE FROM datasets WHERE id=?',(ident,))
 return jsonify(ok=True)
@app.get('/api/datasets/<ident>/export')
def export_dataset(ident):
 d=next(x for x in rows('datasets') if x['id']==ident);return jsonify(name=d['name'],images=json.loads(d['images']),split_seed=42)
@app.post('/api/preprocess')
def preprocess():
 im=valid_image(request.files['file'].read());p=json.loads(request.form.get('config','{}'));im=process(im,p);buf=io.BytesIO();im.save(buf,format='PNG');buf.seek(0);return send_file(buf,mimetype='image/png')
def process(im,p):
 w=int(p.get('width',224));h=int(p.get('height',224))
 if not 8<=w<=512 or not 8<=h<=512:raise ValueError('图像宽高范围 8–512')
 if p.get('crop'):im=ImageOps.fit(im,(min(im.size),)*2)
 if p.get('flip'):im=ImageOps.mirror(im)
 if p.get('rotate'):im=im.rotate(int(p['rotate']),expand=True)
 if p.get('denoise'):im=im.filter(ImageFilter.MedianFilter(3))
 im=im.resize((w,h));return im.convert('L' if int(p.get('channels',3))==1 else 'RGB')
def tensor(im,p,augment=False):
 im=process(im,p)
 if augment and random.random()<.5:im=ImageOps.mirror(im)
 a=np.array(im,dtype=np.float32)/255
 if a.ndim==2:a=a[:,:,None]
 a=torch.from_numpy(a.transpose(2,0,1).copy())
 return (a-.5)/.5 if p.get('normalize',True) else a
class Network(nn.Module):
 def __init__(self,p):
  super().__init__();w=int(p['width']);h=int(p['height']);ch=int(p['channels']);dims=w*h*ch;layers=[]
  if p['mode']=='CNN':layers=[nn.Conv2d(ch,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4))];dims=512
  layers.append(nn.Flatten());act={'ReLU':nn.ReLU,'Sigmoid':nn.Sigmoid,'Tanh':nn.Tanh,'GELU':nn.GELU}[p['activation']]
  count=sum((a+1)*b for a,b in zip([dims]+p['hidden'],p['hidden']+[p['k']]))
  if count>120000000:raise ValueError('参数超过 1.2 亿，请减小图像尺寸或隐藏层')
  for n in p['hidden']:
   layers.extend([nn.Linear(dims,n),act()]);dims=n
   if p.get('dropoutOn'):layers.append(nn.Dropout(float(p['dropout'])))
  layers.append(nn.Linear(dims,p['k']));self.net=nn.Sequential(*layers)
 def forward(self,x):return self.net(x)
def dataset(ident):return json.loads(next(x for x in rows('datasets') if x['id']==ident)['images'])
def batch(items,p,classes,augment=False):return torch.stack([tensor(Image.open(DATA/'images'/i['id']),p,augment) for i in items]),torch.tensor([classes.index(i['label']) for i in items])
@app.route('/api/train',methods=['POST','PATCH'])
def train():
 global job
 if request.method=='PATCH':
  action=request.json['action']
  if action=='pause' and job['status']=='running':job['status']='paused'
  if action=='resume' and job['status']=='paused':job['status']='running'
  if action=='stop' and job['status'] in ['running','paused']:job['status']='stopping'
  return jsonify(job)
 with lock:
  if job['status'] in ['running','paused','stopping']:raise ValueError('已有训练任务')
  p=request.json;items=dataset(p['dataset']);classes=sorted(set(i['label'] for i in items))
  if p['k']!=len(classes):raise ValueError('输出类别数 K 必须与数据集类别数一致')
  if not 1<=int(p['epochs'])<=1000 or not 1<=int(p['batch'])<=256:raise ValueError('训练轮数范围 1–1000，批次大小 1–256')
  if not 0<float(p['lr'])<=1 or not 0<=float(p['dropout'])<1:raise ValueError('学习率或 Dropout 不合法')
  if len(p['hidden'])>8 or any(not 1<=n<=4096 for n in p['hidden']):raise ValueError('隐藏层最多 8 层，每层 1–4096 个神经元')
  process(Image.new('RGB',(8,8)),p);model=Network(p)
  job={'status':'running','epoch':0,'total':int(p['epochs']),'curves':[],'message':'正在初始化训练','config':p}
  threading.Thread(target=run,args=(p,items,classes,model),daemon=True).start()
 return jsonify(job)
def run(p,items,classes,model):
 try:
  random.seed(42);torch.manual_seed(42);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device);cache={};train=[i for i in items if i['split']=='train'];val=[i for i in items if i['split']=='val']
  if len(items)*int(p['width'])*int(p['height'])*int(p['channels'])*4<700000000:
   for i in items:
    if i['split']!='test':cache[i['id']]=tensor(Image.open(DATA/'images'/i['id']),p)
  def getbatch(group,aug=False):
   if cache:
    x=torch.stack([cache[i['id']] for i in group]);y=torch.tensor([classes.index(i['label']) for i in group])
    if aug:
     mask=torch.rand(len(x))<.5;x[mask]=x[mask].flip(-1)
   else:x,y=batch(group,p,classes,aug)
   return x.to(device),y.to(device)
  opt={'Adam':torch.optim.Adam,'SGD':torch.optim.SGD,'AdamW':torch.optim.AdamW}[p['optimizer']](model.parameters(),lr=float(p['lr']),weight_decay=float(p['decay']))
  criterion=nn.CrossEntropyLoss() if p['loss']=='CrossEntropy' else nn.MSELoss();best=float('inf');stale=0;best_state=None
  for epoch in range(int(p['epochs'])):
   while job['status']=='paused':time.sleep(.2)
   if job['status']=='stopping':job.update(status='stopped',message='训练已停止，未保存未完成模型');return
   random.shuffle(train);model.train();total=correct=0;loss_sum=0
   for at in range(0,len(train),int(p['batch'])):
    while job['status']=='paused':time.sleep(.2)
    if job['status']=='stopping':job.update(status='stopped',message='训练已停止');return
    x,y=getbatch(train[at:at+int(p['batch'])],bool(p.get('augment')));opt.zero_grad();out=model(x);loss=criterion(out,y) if p['loss']=='CrossEntropy' else criterion(out.softmax(1),nn.functional.one_hot(y,len(classes)).float());loss.backward();opt.step();total+=len(y);loss_sum+=loss.item()*len(y);correct+=(out.argmax(1)==y).sum().item()
   model.eval();vl=vc=vt=0
   with torch.no_grad():
    for at in range(0,len(val),int(p['batch'])):
     x,y=getbatch(val[at:at+int(p['batch'])]);o=model(x);l=criterion(o,y) if p['loss']=='CrossEntropy' else criterion(o.softmax(1),nn.functional.one_hot(y,len(classes)).float());vl+=l.item()*len(y);vc+=(o.argmax(1)==y).sum().item();vt+=len(y)
   point={'epoch':epoch+1,'loss':loss_sum/total,'val_loss':vl/vt,'accuracy':correct/total,'val_accuracy':vc/vt};job['curves'].append(point);job.update(epoch=epoch+1,message='反向传播与验证完成',**{k:v for k,v in point.items() if k!='epoch'})
   if vl/vt<best:best=vl/vt;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   else:stale+=1
   if p.get('early') and stale>=int(p.get('patience',10)):break
  ident=uuid.uuid4().hex[:10];model.load_state_dict(best_state);torch.save(best_state,DATA/(ident+'.pt'))
  metrics={'curves':job['curves'],'best_val_loss':best,'parameters':sum(x.numel() for x in model.parameters())};name=f"{p['mode']}_v{len(rows('models'))+1}"
  with db() as c:c.execute('INSERT INTO models VALUES(?,?,?,?,?,?,?)',(ident,name,json.dumps(p),json.dumps(classes,ensure_ascii=False),json.dumps(metrics),time.strftime('%Y-%m-%d %H:%M:%S'),p['dataset']))
  record('模型训练',name,metrics);job.update(status='completed',message='训练完成，已保存验证损失最低的模型',model_id=ident);(DATA/'last-job.json').write_text(json.dumps(job),encoding='utf-8')
 except Exception as e:job.update(status='error',message=str(e))
def load(ident):
 meta=next((x for x in rows('models') if x['id']==ident),None)
 if meta is None:raise ValueError('模型不存在')
 obj={'config':json.loads(meta['config']),'classes':json.loads(meta['classes'])};m=Network(obj['config']);m.load_state_dict(torch.load(DATA/(ident+'.pt'),map_location='cpu',weights_only=True));m.eval();return m,obj
@app.post('/api/predict')
def predict():
 m,o=load(request.form['model']);results=[]
 for f in request.files.getlist('files'):
  start=time.perf_counter();raw=f.read();im=valid_image(raw)
  with torch.no_grad():prob=m(tensor(im,o['config']).unsqueeze(0)).softmax(1)[0].tolist()
  order=sorted(zip(o['classes'],prob),key=lambda x:-x[1]);item={'file':f.filename,'predicted':order[0][0],'probabilities':order,'elapsed_ms':round((time.perf_counter()-start)*1000,2),'status':'成功'}
  if session.get('user'):
   review_id=uuid.uuid4().hex[:12];image_id=review_id+'.jpg';folder=DATA/'review_images';folder.mkdir(exist_ok=True);im.save(folder/image_id,format='JPEG',quality=92)
   with db() as c:c.execute('INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?,?)',(review_id,Path(f.filename).name,image_id,request.form['model'],item['predicted'],json.dumps(order,ensure_ascii=False),'','待人工复核','待复核',time.strftime('%Y-%m-%d %H:%M:%S')))
   item['review_id']=review_id
  results.append(item)
 if not results:raise ValueError('请选择图片')
 record('图像分类',request.form['model'],results);return jsonify(results=results)
@app.post('/api/evaluate')
def evaluate():
 from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
 ident=request.json['model'];m,o=load(ident);meta=next(x for x in rows('models') if x['id']==ident);test=[i for i in dataset(meta['dataset']) if i['split']=='test'];truth=[];pred=[]
 with torch.no_grad():
  for at in range(0,len(test),32):
   x,y=batch(test[at:at+32],o['config'],o['classes']);truth+=y.tolist();pred+=m(x).argmax(1).tolist()
 pr,re,f,_=precision_recall_fscore_support(truth,pred,average='macro',zero_division=0);result={'accuracy':accuracy_score(truth,pred),'precision':pr,'recall':re,'f1':f,'matrix':confusion_matrix(truth,pred,labels=list(range(len(o['classes'])))).tolist(),'classes':o['classes'],'samples':len(test)};record('独立测试评估',meta['name'],result);return jsonify(result)

@app.get('/api/reviews')
def list_reviews():return jsonify(reviews=rows('reviews')[::-1])
@app.get('/api/review-image/<ident>')
def review_image(ident):return send_file(DATA/'review_images'/Path(ident).name)
@app.patch('/api/reviews/<ident>')
def confirm_review(ident):
 data=request.json or {};label=data.get('correct_label','').strip();note=data.get('note','').strip()[:1000]
 if not label or len(label)>80:raise ValueError('请输入正确类别（1–80 个字符）')
 with db() as c:
  row=c.execute('SELECT * FROM reviews WHERE id=?',(ident,)).fetchone()
  if not row:raise ValueError('复核记录不存在')
  c.execute("UPDATE reviews SET correct_label=?,note=?,status='已确认' WHERE id=?",(label,note,ident))
 record('人工复核',row['filename'],{'review_id':ident,'predicted':row['predicted'],'correct_label':label,'model':row['model']})
 return jsonify(ok=True)
@app.get('/api/reviews/export')
def export_reviews():
 confirmed=[r for r in rows('reviews') if r['status']=='已确认']
 if not confirmed:raise ValueError('暂无已确认的错例可导出')
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for r in confirmed:
   if not r['correct_label']:continue
   path=DATA/'review_images'/Path(r['image_id']).name
   if path.exists():z.write(path,f"{r['correct_label']}/{r['id']}_{Path(r['filename']).name}")
  z.writestr('复核说明.txt','本压缩包仅含人工确认类别的错例图片。导入平台前请检查类别标签、隐私与图片质量。\n')
 buf.seek(0);return send_file(buf,mimetype='application/zip',as_attachment=True,download_name='已确认错例数据.zip')
@app.get('/api/models/<ident>/download')
def download(ident):return send_file(DATA/(Path(ident).name+'.pt'),as_attachment=True,download_name=ident+'.pt')

@app.post('/api/datasets/<ident>/label')
def rename_label(ident):
 if job['status'] in ['running','paused','stopping']:raise ValueError('训练中不能修改标签')
 data=request.json;name=data.get('name','').strip()
 if not name or len(name)>80:raise ValueError('类别名称长度需为 1–80')
 if any(m['dataset']==ident for m in rows('models')):raise ValueError('已有模型绑定此数据集，请导入新版本数据集后修改标签')
 items=dataset(ident)
 if name in set(i['label'] for i in items):raise ValueError('类别已存在，不允许合并已划分的类别')
 for i in items:
  if i['label']==data['old']:i['label']=name
 with db() as c:c.execute('UPDATE datasets SET images=? WHERE id=?',(json.dumps(items,ensure_ascii=False),ident))
 return jsonify(ok=True)

if __name__=='__main__':app.run(host='127.0.0.1',port=5088,threaded=True,debug=False)
