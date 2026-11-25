"""Train a richer used-car resale model. Run: python -m src.train"""
import datetime, glob, os, re, warnings
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings('ignore')
RNG=42; YEAR=datetime.datetime.now().year
REQ={'name','year','selling_price','km_driven','fuel'}
NUM_EXTRA=['mileage','engine','max_power','seats']; CAT_EXTRA=['seller_type','transmission','owner']

def number(v):
    if pd.isna(v): return np.nan
    m=re.search(r'-?\d+(?:\.\d+)?',str(v).replace(',',''))
    return float(m.group()) if m else np.nan

def find_csv(folder):
    found=[]
    for p in glob.glob(os.path.join(folder,'**','*.csv'),recursive=True):
        try:
            c=set(pd.read_csv(p,nrows=1).columns.str.strip().str.lower())
            if REQ<=c: found.append((len(c & set(NUM_EXTRA+CAT_EXTRA)),os.path.getsize(p),p))
        except Exception: pass
    return max(found,default=(0,0,None))[2]

def load():
    p=find_csv('data')
    if p: print('Using local dataset:',p); return pd.read_csv(p)
    try:
        import kagglehub
        p=find_csv(kagglehub.dataset_download('nehalbirla/vehicle-dataset-from-cardekho'))
        if p: print('Using Kaggle dataset:',p); return pd.read_csv(p)
    except Exception as e: print('Kaggle download unavailable:',e)
    raise FileNotFoundError('Put a compatible CarDekho CSV in data/. Required: '+', '.join(sorted(REQ)))

def prepare(raw):
    d=raw.copy(); d.columns=[x.strip().lower() for x in d.columns]
    if not REQ<=set(d.columns): raise ValueError('Missing required columns: '+str(sorted(REQ-set(d.columns))))
    d=d.drop_duplicates().dropna(subset=list(REQ)).copy(); d['name']=d.name.astype(str).str.strip()
    d['brand']=d.name.str.split().str[0].str.title().replace({'Land':'Land Rover'})
    d['model_name']=d.name.str.split(n=1).str[1].fillna('Other').str.strip()
    for c in ['year','selling_price','km_driven']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['year','selling_price','km_driven']).copy(); d['age']=(YEAR-d.year).clip(lower=1)
    d['fuel']=d.fuel.astype(str).str.strip().str.title()
    nums=['age','km_driven']; cats=['brand','model_name','fuel']
    for c in NUM_EXTRA:
        if c in d:
            d[c]=d[c].map(number); nums.append(c)
    for c in CAT_EXTRA:
        if c in d:
            d[c]=d[c].fillna('Unknown').astype(str).str.strip().str.title(); cats.append(c)
    for c,n in [('brand',10),('model_name',5)]:
        rare=d[c].value_counts(); d[c]=d[c].where(d[c].map(rare)>=n,'Other')
    d=d[(d.km_driven>500)&(d.km_driven<1_000_000)&(d.selling_price>=20_000)&(d.year<=YEAR)].copy()
    lp=np.log1p(d.selling_price); q1,q3=lp.quantile([.25,.75]); iq=q3-q1
    d=d[lp.between(q1-2*iq,q3+2*iq)].copy()
    if len(d)<50: raise ValueError('Fewer than 50 usable records remain.')
    return d,nums,cats,nums+cats

def make_model(nums,cats,reg):
    pre=ColumnTransformer([
      ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),
      ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=5,sparse_output=False))]),cats)])
    return TransformedTargetRegressor(regressor=Pipeline([('prep',pre),('reg',reg)]),func=np.log1p,inverse_func=np.expm1)

def train():
    d,nums,cats,features=prepare(load()); X=d[features]; y=d.selling_price
    xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=RNG)
    regs={'Random Forest':RandomForestRegressor(n_estimators=220,min_samples_leaf=2,n_jobs=-1,random_state=RNG),
          'Extra Trees':ExtraTreesRegressor(n_estimators=220,min_samples_leaf=2,n_jobs=-1,random_state=RNG),
          'Hist Gradient Boosting':HistGradientBoostingRegressor(max_iter=300,learning_rate=.07,max_leaf_nodes=20,l2_regularization=1,random_state=RNG)}
    cv=KFold(n_splits=5,shuffle=True,random_state=RNG); scores={}
    for name,reg in regs.items():
        sc=cross_val_score(make_model(nums,cats,reg),xt,yt,cv=cv,scoring='r2',n_jobs=-1)
        scores[name]=float(sc.mean()); print(f'{name}: CV R2={sc.mean():.4f} (+/- {sc.std():.4f})')
    best_name=max(scores,key=scores.get); best=make_model(nums,cats,regs[best_name]); best.fit(xt,yt); pred=best.predict(xv)
    metrics={'r2':float(r2_score(yv,pred)),'mae':float(mean_absolute_error(yv,pred)),'rmse':float(np.sqrt(mean_squared_error(yv,pred))),'mape':float(np.mean(np.abs((yv-pred)/yv))*100)}
    resid=np.log1p(yv.to_numpy())-np.log1p(pred); lo,hi=np.percentile(resid,[10,90])
    defaults={c:(float(d[c].median()) if c in nums and d[c].notna().any() else (str(d[c].mode().iloc[0]) if c in cats and len(d[c].mode()) else 0)) for c in features}
    os.makedirs('models',exist_ok=True)
    joblib.dump({'model':best,'features':features,'numeric_features':nums,'categorical_features':cats,'brands':sorted(d.brand.unique().tolist()),
      'models_by_brand':{str(b):sorted(g.model_name.unique().tolist()) for b,g in d.groupby('brand')},'fuels':sorted(d.fuel.unique().tolist()),
      'category_options':{c:sorted(d[c].astype(str).unique().tolist()) for c in cats if c not in ['brand','model_name','fuel']},'defaults':defaults,
      'res_lo':float(lo),'res_hi':float(hi),'current_year':YEAR,'metrics':metrics,'best_model_name':best_name,'cv_scores':scores,'training_rows':len(d)},'models/used_car_price_model.joblib')
    print('\nTest metrics:',metrics); print('Saved models/used_car_price_model.joblib')
if __name__=='__main__': train()
