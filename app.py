"""Used-car valuation: ML estimate + historical price lookup + condition scenario."""
import datetime, json, os
import joblib, numpy as np, pandas as pd, streamlit as st
MODEL='models/used_car_price_model.joblib'; HISTORY='data/historical_car_prices.csv'; ADJ='config/adjustments.json'
st.set_page_config(page_title='Used Car Price Predictor India',page_icon='🚗',layout='wide')
st.title('🚗 Used Car Price Predictor – India')
st.caption('AI resale estimate, historical purchase-price lookup, seller asking-price comparison and vehicle-condition review.')
if not os.path.exists(MODEL):
    st.warning('Model not trained yet. Add a CarDekho CSV to data/ and run:'); st.code('python -m src.train'); st.stop()
a=joblib.load(MODEL); model=a['model']; features=a['features']; nums=a['numeric_features']; cats=a['categorical_features']; defaults=a['defaults']; brands=a['brands']; bybrand=a['models_by_brand']; fuels=a['fuels']; year_now=a['current_year']
BASE={'excellent':.03,'good':0,'average':-.04,'poor':-.10,'minor_accident':-.05,'major_accident':-.15,'flood_damage':-.25,'repainted':-.02,'full_service':.02,'no_service_records':-.04,'insurance_expired':-.015,'insurance_third_party':-.005,'tyres_worn':-.02,'modified':-.02}
try:
    with open(ADJ,encoding='utf-8') as f: BASE.update(json.load(f))
except Exception: pass

def load_history(upload):
    try:
        if upload is not None: return pd.read_csv(upload)
        if os.path.exists(HISTORY): return pd.read_csv(HISTORY)
    except Exception as e: st.error(f'Historical file error: {e}')
    return None

def lookup(df,brand,model_name,purchase_year,variant):
    if df is None: return None,'No historical price file supplied.'
    d=df.copy(); d.columns=[c.strip().lower() for c in d.columns]
    need={'brand','model','purchase_year'}
    if not need<=set(d.columns): return None,'File needs brand, model and purchase_year columns.'
    d['by']=d.brand.astype(str).str.strip().str.casefold(); d['mo']=d.model.astype(str).str.strip().str.casefold(); d['py']=pd.to_numeric(d.purchase_year,errors='coerce')
    m=d[(d.by==brand.casefold())&(d.mo==model_name.casefold())&(d.py==int(purchase_year))]
    if variant and 'variant' in d:
        exact=m[m.variant.astype(str).str.strip().str.casefold()==variant.casefold()]
        if not exact.empty: m=exact
    if m.empty: return None,'No exact brand/model/purchase-year match.'
    col='on_road_price_inr' if 'on_road_price_inr' in m else 'ex_showroom_price_inr'
    if col not in m: return None,'Add on_road_price_inr or ex_showroom_price_inr.'
    value=pd.to_numeric(m.iloc[0][col],errors='coerce')
    if pd.isna(value): return None,'Matched row has no valid price.'
    return float(value),f'{col} · {m.iloc[0].get("source","your historical file")}'

def rupee(x): return f'₹{x:,.0f}'

st.subheader('1. Car specifications')
c1,c2=st.columns(2)
with c1:
    brand=st.selectbox('Brand',brands); model_name=st.selectbox('Model',bybrand.get(brand,['Other']))
    mfg_year=st.number_input('Manufacturing year',1990,year_now,max(1990,year_now-5))
    fuel=st.selectbox('Fuel',fuels); km=st.number_input('Kilometres driven',501,999999,50000,1000)
with c2:
    extra={}
    for col,label in [('transmission','Transmission'),('owner','Previous owners'),('seller_type','Seller type')]:
        if col in cats: extra[col]=st.selectbox(label,a['category_options'].get(col,['Unknown']))
    labels={'mileage':'Mileage (km/l)','engine':'Engine (cc)','max_power':'Max power (bhp)','seats':'Seats'}
    for col in nums:
        if col not in ['age','km_driven']:
            default=float(defaults.get(col,0) or 0)
            if col=='seats': extra[col]=st.number_input(labels[col],2,12,int(np.clip(round(default or 5),2,12)))
            else: extra[col]=st.number_input(labels.get(col,col),min_value=0.0,value=max(0.0,default),step=.1)

st.subheader('2. Purchase price and selling price')
p1,p2,p3=st.columns(3)
with p1: purchase_year=st.number_input('Year owner purchased',1990,year_now,int(mfg_year))
with p2: original_manual=st.number_input('Actual original purchase price (₹)',0,10_00_00_000,0,10000,help='Use invoice/on-road price when known.')
with p3: asking=st.number_input("Owner's expected selling price (₹)",0,10_00_00_000,0,10000)
variant=st.text_input('Variant / trim (optional)',placeholder='e.g. VXi, ZXi+, SX(O)')
upload=st.file_uploader('Upload historical purchase-price CSV (optional)',type=['csv'])
history=load_history(upload); historical,lookup_note=lookup(history,brand,model_name,purchase_year,variant)
original=original_manual if original_manual>0 else historical
if historical is not None: st.info(f'Historical file match: {rupee(historical)} ({lookup_note}). Manual invoice price overrides it.')
else: st.caption(lookup_note)

st.subheader('3. Insurance, damage and condition')
d1,d2=st.columns(2)
with d1:
    condition=st.selectbox('Overall condition',['excellent','good','average','poor'],index=1,format_func=str.title)
    accident=st.selectbox('Accident history',['none','minor_accident','major_accident'],format_func=lambda x:{'none':'No known accident','minor_accident':'Minor accident repaired','major_accident':'Major accident / structural repair'}[x])
    flood=st.checkbox('Flood / water damage history'); repainted=st.checkbox('Major panels repainted')
with d2:
    insurance=st.selectbox('Insurance',['active_comprehensive','insurance_third_party','insurance_expired'],format_func=lambda x:{'active_comprehensive':'Active comprehensive','insurance_third_party':'Third-party only','insurance_expired':'Expired'}[x])
    service=st.selectbox('Service records',['full_service','partial_service','no_service_records'],format_func=lambda x:{'full_service':'Full service history','partial_service':'Partial records','no_service_records':'No service records'}[x])
    tyres=st.checkbox('Tyres need replacement'); modified=st.checkbox('Non-standard modifications')

if st.button('Calculate valuation',type='primary',use_container_width=True):
    row={c:defaults.get(c,0) for c in features}; row.update({'age':max(year_now-int(mfg_year),1),'km_driven':float(km),'brand':brand,'model_name':model_name,'fuel':fuel}); row.update(extra)
    pred=max(0,float(model.predict(pd.DataFrame([row])[features])[0])); lo=max(0,float(np.expm1(np.log1p(pred)+a['res_lo']))); hi=max(lo,float(np.expm1(np.log1p(pred)+a['res_hi'])))
    factor=1+BASE.get(condition,0)+BASE.get(accident,0)+BASE.get(insurance,0)+BASE.get(service,0)+BASE.get('flood_damage',-.25 if flood else 0)*(1 if flood else 0)+BASE.get('repainted',0)*(1 if repainted else 0)+BASE.get('tyres_worn',0)*(1 if tyres else 0)+BASE.get('modified',0)*(1 if modified else 0)
    factor=max(.35,factor); adjusted=pred*factor; low=lo*factor; high=hi*factor
    x,y,z=st.columns(3); x.metric('AI market estimate',rupee(pred)); y.metric('Condition-adjusted estimate',rupee(adjusted),f'{(factor-1)*100:+.1f}% scenario'); z.metric('Indicative range',f'{rupee(low)} – {rupee(high)}')
    comp={'AI estimate':pred,'Condition-adjusted estimate':adjusted}
    if original and original>0:
        comp['Original purchase price']=float(original); st.write(f'Change from original purchase price: **{(adjusted/original-1)*100:+.1f}%**')
    if asking>0:
        comp['Owner asking price']=float(asking); st.write(f'Asking price vs adjusted estimate: **{(asking/adjusted-1)*100:+.1f}%**')
    st.bar_chart(pd.Series(comp,name='Price (₹)'))
    if asking>0:
        if asking<=high: st.success('Asking price is within or below the indicative model range.')
        elif asking<=high*1.10: st.warning('Asking price is slightly above the model range; negotiation may help.')
        else: st.error('Asking price is notably above the indicative model range.')
    st.caption('Condition adjustments are editable assumptions in config/adjustments.json, not effects learned from CarDekho. Have the car independently inspected. Historical prices are only as reliable as the CSV source.')
with st.expander('Model performance'):
    st.write('Best model:',a.get('best_model_name')); st.write('Training rows:',a.get('training_rows')); st.json(a.get('metrics',{}))
with st.expander('Other factors that can affect resale'):
    st.write('Exact variant, registration city/state, local demand, ownership documents, service network, tyre/brake/battery condition, pending loans, accident/flood inspection and current comparable listings. City/local demand are not yet model inputs.')
