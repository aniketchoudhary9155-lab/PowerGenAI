import os
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title='PowerGenAI', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')

# -----------------------------
# Styling
# -----------------------------
st.markdown('''
<style>
.stApp { background: linear-gradient(135deg,#07111f 0%,#0b1728 45%,#101d31 100%); }
.main .block-container { max-width: 1550px; padding-top: 1.2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#06101d 0%,#0b1828 100%); }
section[data-testid="stSidebar"] * { color:#e5edf7; }
h1,h2,h3 { color:#f8fafc !important; }
p,.stMarkdown,label { color:#cbd5e1; }
.metric-card { background:linear-gradient(145deg,rgba(30,64,175,.25),rgba(15,23,42,.75)); border:1px solid rgba(96,165,250,.18); border-radius:18px; padding:18px; min-height:115px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
.metric-title { font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.6px; }
.metric-value { font-size:1.55rem;font-weight:800;color:#f8fafc;margin-top:7px; }
.metric-sub { font-size:.76rem;color:#60a5fa;margin-top:5px; }
.hero { background:linear-gradient(135deg,#0f2745,#0a1628); border:1px solid rgba(96,165,250,.2); border-radius:24px; padding:28px; margin-bottom:22px; }
.hero-title { font-size:2.35rem;font-weight:900;color:white; }
.hero-subtitle { color:#93c5fd;font-size:1rem; }
.badge-good { color:#86efac;background:rgba(34,197,94,.14);padding:5px 11px;border-radius:999px;font-weight:700; }
.badge-warn { color:#fcd34d;background:rgba(245,158,11,.14);padding:5px 11px;border-radius:999px;font-weight:700; }
.badge-danger { color:#fca5a5;background:rgba(239,68,68,.14);padding:5px 11px;border-radius:999px;font-weight:700; }
</style>
''', unsafe_allow_html=True)

REQUIRED = {
    'station':'Power_Station', 'capacity':'Monitored_Capacity', 'total_maint':'Total_Maintenance',
    'planned':'Planned_Maintenance', 'forced':'Forced_Maintenance', 'other':'Other_Reasons',
    'programme':'Programme', 'actual':'Actual', 'shortfall':'Excess_Shortfall', 'deviation':'Deviation'
}


def metric_card(title, value, subtitle=''):
    st.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div><div class="metric-sub">{subtitle}</div></div>', unsafe_allow_html=True)


def fmt_mw(x):
    try: return f'{float(x):,.2f} MW'
    except: return 'N/A'


def fmt_pct(x):
    try: return f'{float(x):,.2f}%'
    except: return 'N/A'


def badge(text, level='good'):
    cls={'good':'badge-good','warn':'badge-warn','danger':'badge-danger'}.get(level,'badge-warn')
    st.markdown(f'<span class="{cls}">{text}</span>', unsafe_allow_html=True)


def normalize_columns(df):
    df = df.copy()
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        key = key.replace('__','_')
        mapping = {
            'power_station':'Power_Station', 'monitored_cap._mw':'Monitored_Capacity',
            'monitored_cap_mw':'Monitored_Capacity', 'monitored_capacity':'Monitored_Capacity',
            'total_cap._under_maintenance':'Total_Maintenance', 'total_cap_under_maintenance':'Total_Maintenance',
            'total_maintenance':'Total_Maintenance', 'planned_maintenance':'Planned_Maintenance',
            'forced_maintenance':'Forced_Maintenance', 'other_reasons':'Other_Reasons',
            'programme_mw':'Programme', 'programme':'Programme', 'actual_mw':'Actual', 'actual':'Actual',
            'excess_shortfall_mw':'Excess_Shortfall', 'excess_shortfall':'Excess_Shortfall',
            'deviation_mw':'Deviation', 'deviation':'Deviation', 'date':'Date'
        }
        if key in mapping: rename[c] = mapping[key]
    df.rename(columns=rename, inplace=True)
    # Derived/compatibility columns
    if 'Total_Maintenance' not in df and all(c in df for c in ['Planned_Maintenance','Forced_Maintenance','Other_Reasons']):
        df['Total_Maintenance'] = df[['Planned_Maintenance','Forced_Maintenance','Other_Reasons']].sum(axis=1)
    if 'Excess_Shortfall' not in df and all(c in df for c in ['Actual','Programme']):
        df['Excess_Shortfall'] = df['Actual'] - df['Programme']
    if 'Deviation' not in df and all(c in df for c in ['Actual','Programme']):
        df['Deviation'] = df['Actual'] - df['Programme']
    for c in [x for x in REQUIRED.values() if x != 'Power_Station']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if 'Power_Station' in df.columns: df['Power_Station'] = df['Power_Station'].astype(str).str.strip()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    return df


@st.cache_data(show_spinner=False)
def load_data(path):
    if path.lower().endswith(('.xlsx','.xls')):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    return normalize_columns(raw)


@st.cache_data(show_spinner=False)
def station_summary(df):
    g = df.groupby('Power_Station', dropna=False)
    out = g.agg(
        Records=('Power_Station','size'),
        Capacity=('Monitored_Capacity','mean'),
        Programme=('Programme','sum'),
        Actual=('Actual','sum'),
        Planned_Maintenance=('Planned_Maintenance','sum'),
        Forced_Maintenance=('Forced_Maintenance','sum'),
        Other_Reasons=('Other_Reasons','sum'),
        Shortfall=('Excess_Shortfall','sum'),
        Deviation=('Deviation','sum')
    ).reset_index()
    out['Total_Maintenance'] = out[['Planned_Maintenance','Forced_Maintenance','Other_Reasons']].sum(axis=1)
    out['Achievement_%'] = np.where(out['Programme'] != 0, out['Actual']/out['Programme']*100, 0)
    out['Maintenance_%'] = np.where(out['Capacity'] != 0, out['Total_Maintenance']/out['Capacity']*100, 0)
    out['Forced_Maintenance_%'] = np.where(out['Capacity'] != 0, out['Forced_Maintenance']/out['Capacity']*100, 0)
    out['Capacity_Utilization_%'] = np.where(out['Capacity'] != 0, out['Actual']/out['Capacity']*100, 0)
    out['Deviation_%'] = np.where(out['Programme'] != 0, out['Deviation'].abs()/out['Programme'].abs()*100, 0)
    # Bounded engineering dashboard score; transparent and deterministic.
    achievement_score = np.clip(out['Achievement_%'],0,120)/120*30
    utilization_score = np.clip(out['Capacity_Utilization_%'],0,100)/100*20
    maintenance_score = 20*(1-np.clip(out['Maintenance_%'],0,100)/100)
    forced_score = 15*(1-np.clip(out['Forced_Maintenance_%'],0,100)/100)
    deviation_score = 15*(1-np.clip(out['Deviation_%'],0,100)/100)
    out['Health_Score'] = np.clip(achievement_score+utilization_score+maintenance_score+forced_score+deviation_score,0,100)
    out['Status'] = pd.cut(out['Health_Score'], [-np.inf,50,70,85,np.inf], labels=['CRITICAL','HIGH RISK','WARNING','GOOD'])
    return out.sort_values('Actual', ascending=False)


@st.cache_resource(show_spinner=False)
def train_fallback_model(df, sample_size=60000):
    cols = ['Power_Station','Monitored_Capacity','Programme','Planned_Maintenance','Forced_Maintenance','Other_Reasons']
    d = df[cols+['Actual']].replace([np.inf,-np.inf],np.nan).dropna()
    if len(d) > sample_size: d = d.sample(sample_size, random_state=42)
    X = d[cols]; y = d['Actual']
    cat = ['Power_Station']; num = [c for c in cols if c not in cat]
    prep = ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num','passthrough',num)])
    model = ExtraTreesRegressor(n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2)
    pipe = Pipeline([('prep',prep),('model',model)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    metrics={'R2':r2_score(yte,pred),'MAE':mean_absolute_error(yte,pred),'RMSE':np.sqrt(mean_squared_error(yte,pred))}
    return pipe, metrics


def predict_one(model, station, capacity, programme, planned, forced, other):
    x = pd.DataFrame([{'Power_Station':station,'Monitored_Capacity':capacity,'Programme':programme,'Planned_Maintenance':planned,'Forced_Maintenance':forced,'Other_Reasons':other}])
    return float(model.predict(x)[0])


def status_for(row):
    if row['Health_Score'] >= 85: return 'GOOD','good'
    if row['Health_Score'] >= 70: return 'WARNING','warn'
    if row['Health_Score'] >= 50: return 'HIGH RISK','warn'
    return 'CRITICAL','danger'


def make_alerts(df):
    d=df.copy()
    d['Achievement_%']=np.where(d['Programme']!=0,d['Actual']/d['Programme']*100,0)
    d['Maintenance_%']=np.where(d['Monitored_Capacity']!=0,d['Total_Maintenance']/d['Monitored_Capacity']*100,0)
    d['Forced_%']=np.where(d['Monitored_Capacity']!=0,d['Forced_Maintenance']/d['Monitored_Capacity']*100,0)
    d['Deviation_%']=np.where(d['Programme']!=0,d['Deviation'].abs()/d['Programme'].abs()*100,0)
    rows=[]
    for _,r in d.iterrows():
        if r['Achievement_%'] < 70: rows.append((r['Power_Station'],'Generation Shortfall','CRITICAL' if r['Achievement_%']<50 else 'HIGH',r['Achievement_%']))
        if r['Maintenance_%'] > 50: rows.append((r['Power_Station'],'High Maintenance','HIGH',r['Maintenance_%']))
        if r['Forced_%'] > 25: rows.append((r['Power_Station'],'High Forced Maintenance','CRITICAL',r['Forced_%']))
        if r['Deviation_%'] > 50: rows.append((r['Power_Station'],'High Deviation','HIGH',r['Deviation_%']))
    return pd.DataFrame(rows, columns=['Power Station','Alert Type','Severity','Value'])


# -----------------------------
# Load dataset
# -----------------------------
def find_default_data():
    candidates=[
        os.path.join(os.path.dirname(__file__),'PowerGeneration sorted data.xlsx'),
        os.path.join(os.path.dirname(__file__),'data','PowerGeneration sorted data.xlsx'),
        os.path.join(os.path.dirname(__file__),'data','processed','powergeneration_features.csv'),
        os.path.join(os.path.dirname(__file__),'data','processed','powergeneration_features.xlsx'),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)

default_path=find_default_data()
with st.sidebar:
    st.markdown('<div style="text-align:center;font-size:2.8rem">⚡</div><div style="text-align:center;font-size:1.6rem;font-weight:900">PowerGenAI</div><div style="text-align:center;color:#94a3b8">Power Station AI Decision Support</div>', unsafe_allow_html=True)
    st.divider()
    uploaded=st.file_uploader('📤 Upload CSV / Excel', type=['csv','xlsx','xls'])
    data_path=uploaded
    if uploaded is None and default_path:
        data_path=default_path
    if uploaded is not None:
        raw=uploaded.read()
        if uploaded.name.lower().endswith('.csv'):
            df=normalize_columns(pd.read_csv(io.BytesIO(raw)))
        else:
            df=normalize_columns(pd.read_excel(io.BytesIO(raw)))
    elif default_path:
        df=load_data(default_path)
    else:
        st.error('No dataset found. Upload your PowerGeneration Excel/CSV file.'); st.stop()
    if 'Power_Station' not in df.columns:
        st.error('Power_Station column not found.'); st.stop()
    summary=station_summary(df)
    st.divider()
    st.caption(f'Rows: {len(df):,}')
    st.caption(f'Stations: {df.Power_Station.nunique():,}')
    st.caption(f'Columns: {len(df.columns):,}')
    st.divider()
    page=st.radio('NAVIGATION',[
        '📊 Executive Dashboard','🏭 Station Intelligence','🔮 AI Generation Prediction','🎛️ Scenario Laboratory',
        '🔧 Maintenance Intelligence','🚨 Alert Center','🕵️ Anomaly Detection','📉 Shortfall & Deviation',
        '🎯 AI Optimization','🧠 Explainable AI','📈 Model Performance','📊 Data Analytics','📑 Reports','📋 Data Explorer'
    ])

# -----------------------------
# Pages
# -----------------------------
if page=='📊 Executive Dashboard':
    st.markdown('<div class="hero"><div class="hero-title">⚡ PowerGenAI Control Center</div><div class="hero-subtitle">AI-based power generation analysis, maintenance intelligence, anomaly detection and operational decision support.</div></div>', unsafe_allow_html=True)
    station_filter=st.selectbox('🏭 Station Filter',['All Stations']+sorted(df.Power_Station.unique()))
    v=df if station_filter=='All Stations' else df[df.Power_Station==station_filter]
    total_cap=v.Monitored_Capacity.sum(); prog=v.Programme.sum(); actual=v.Actual.sum(); maint=v.Total_Maintenance.sum(); sf=v.loc[v.Excess_Shortfall<0,'Excess_Shortfall'].sum()
    ach=actual/prog*100 if prog else 0
    forced=v.Forced_Maintenance.sum(); forced_pct=forced/total_cap*100 if total_cap else 0
    k=[('Stations',v.Power_Station.nunique(),'Monitored'),('Capacity',fmt_mw(total_cap),'Monitored capacity'),('Programme',fmt_mw(prog),'Target'),('Actual',fmt_mw(actual),'Recorded output'),('Shortfall',fmt_mw(sf),'Negative = deficit'),('Achievement',fmt_pct(ach),'Actual / programme'),('Maintenance',fmt_mw(maint),'Total maintenance'),('Forced Maint.',fmt_pct(forced_pct),'Forced / capacity')]
    for i in range(0,8,4):
        cols=st.columns(4)
        for j in range(4):
            with cols[j]: metric_card(*k[i+j])
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        d=v[['Programme','Actual']].sum().reset_index(); d.columns=['Metric','MW']; st.plotly_chart(px.bar(d,x='Metric',y='MW',text_auto='.2f',title='Programme vs Actual',template='plotly_dark'),use_container_width=True)
    with c2:
        m=v[['Planned_Maintenance','Forced_Maintenance','Other_Reasons']].sum().reset_index(); m.columns=['Type','MW']; st.plotly_chart(px.pie(m,names='Type',values='MW',hole=.5,title='Maintenance Composition',template='plotly_dark'),use_container_width=True)
    st.subheader('🏆 Station Performance Ranking')
    st.dataframe(summary[['Power_Station','Actual','Programme','Achievement_%','Total_Maintenance','Shortfall','Deviation','Health_Score','Status']].round(2).head(25),use_container_width=True,hide_index=True)

elif page=='🏭 Station Intelligence':
    st.title('🏭 Station Intelligence')
    station=st.selectbox('Select Power Station',sorted(df.Power_Station.unique()))
    d=df[df.Power_Station==station].copy(); s=summary[summary.Power_Station==station].iloc[0]
    status,level=status_for(s)
    st.markdown(f'### {station}'); badge(status,level)
    c=st.columns(5)
    vals=[('Health Score',f"{s.Health_Score:.1f}/100",'Composite score'),('Capacity',fmt_mw(s.Capacity),'Mean monitored'),('Actual',fmt_mw(s.Actual),'Total'),('Achievement',fmt_pct(s['Achievement_%']),'Programme achieved'),('Shortfall',fmt_mw(s.Shortfall),'Total')]
    for col,item in zip(c,vals):
        with col: metric_card(*item)
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        x=d[['Programme','Actual']].mean().reset_index(); x.columns=['Metric','Average MW']; st.plotly_chart(px.bar(x,x='Metric',y='Average MW',text_auto='.2f',title='Average Programme vs Actual',template='plotly_dark'),use_container_width=True)
    with c2:
        x=d[['Planned_Maintenance','Forced_Maintenance','Other_Reasons']].sum().reset_index(); x.columns=['Type','MW']; st.plotly_chart(px.bar(x,x='Type',y='MW',text_auto='.2f',title='Maintenance Breakdown',template='plotly_dark'),use_container_width=True)
    st.subheader('Engineering indicators')
    ind=pd.DataFrame({'Indicator':['Capacity Utilization','Maintenance Impact','Forced Maintenance Impact','Deviation'], 'Value (%)':[s['Capacity_Utilization_%'],s['Maintenance_%'],s['Forced_Maintenance_%'],s['Deviation_%']]})
    st.dataframe(ind.round(2),hide_index=True,use_container_width=True)

elif page=='🔮 AI Generation Prediction':
    st.title('🔮 AI Generation Prediction')
    try:
        model,metrics=train_fallback_model(df)
        st.caption(f"Fallback/embedded model trained from your dataset. R²={metrics['R2']:.4f} | MAE={metrics['MAE']:.3f} MW | RMSE={metrics['RMSE']:.3f} MW")
    except Exception as e:
        st.error(f'Model training failed: {e}'); st.stop()
    station=st.selectbox('🏭 Station',sorted(df.Power_Station.unique()))
    sd=df[df.Power_Station==station]
    base=sd[['Monitored_Capacity','Programme','Planned_Maintenance','Forced_Maintenance','Other_Reasons']].mean()
    c1,c2=st.columns(2)
    with c1:
        capacity=st.number_input('Monitored Capacity (MW)',0.0,float(max(base.Monitored_Capacity,1)),float(base.Monitored_Capacity))
        programme=st.number_input('Programme (MW)',0.0,float(max(base.Programme*2,1)),float(base.Programme))
        planned=st.number_input('Planned Maintenance (MW)',0.0,float(max(capacity,1)),float(base.Planned_Maintenance))
    with c2:
        forced=st.number_input('Forced Maintenance (MW)',0.0,float(max(capacity,1)),float(base.Forced_Maintenance))
        other=st.number_input('Other Reasons (MW)',0.0,float(max(capacity,1)),float(base.Other_Reasons))
    if st.button('⚡ RUN AI PREDICTION',use_container_width=True):
        pred=predict_one(model,station,capacity,programme,planned,forced,other); diff=pred-programme; ach=pred/programme*100 if programme else 0; avail=max(capacity-planned-forced-other,0)
        c=st.columns(4)
        for col,item in zip(c,[('Predicted Generation',fmt_mw(pred),'AI estimate'),('Expected Excess/Shortfall',fmt_mw(diff),'Prediction - programme'),('Achievement',fmt_pct(ach),'Expected'),('Available Capacity',fmt_mw(avail),'Capacity after maintenance')]):
            with col: metric_card(*item)
        badge('LOW RISK' if ach>=95 else 'MEDIUM RISK' if ach>=75 else 'HIGH RISK','good' if ach>=95 else 'warn' if ach>=75 else 'danger')

elif page=='🎛️ Scenario Laboratory':
    st.title('🎛️ Scenario Laboratory')
    model,metrics=train_fallback_model(df)
    station=st.selectbox('Station',sorted(df.Power_Station.unique()))
    sd=df[df.Power_Station==station]; b=sd[['Monitored_Capacity','Programme','Planned_Maintenance','Forced_Maintenance','Other_Reasons']].mean()
    capacity=float(b.Monitored_Capacity); programme=float(b.Programme); planned=float(b.Planned_Maintenance); forced=float(b.Forced_Maintenance); other=float(b.Other_Reasons)
    st.caption('Compare current conditions with automatically generated maintenance-reduction scenarios.')
    scenarios={'Current':1.0,'Forced Maintenance -10%':.90,'Forced Maintenance -20%':.80,'Forced Maintenance -30%':.70,'No Forced Maintenance':0.0}
    rows=[]
    for name,mult in scenarios.items():
        f=forced*mult; p=predict_one(model,station,capacity,programme,planned,f,other); rows.append([name, f, p, p-programme, p/programme*100 if programme else 0])
    res=pd.DataFrame(rows,columns=['Scenario','Forced Maintenance','Predicted Actual','Expected Shortfall/Excess','Achievement %'])
    st.dataframe(res.round(2),use_container_width=True,hide_index=True)
    best=res.loc[res['Predicted Actual'].idxmax()]
    st.success(f"🏆 Best simulated scenario: {best['Scenario']} — predicted generation {best['Predicted Actual']:.2f} MW, achievement {best['Achievement %']:.2f}%.")
    st.plotly_chart(px.bar(res,x='Scenario',y='Predicted Actual',text_auto='.2f',title='Scenario Comparison',template='plotly_dark'),use_container_width=True)

elif page=='🔧 Maintenance Intelligence':
    st.title('🔧 Maintenance Intelligence')
    totals=df[['Planned_Maintenance','Forced_Maintenance','Other_Reasons']].sum(); total=totals.sum(); cap=df.Monitored_Capacity.sum()
    c=st.columns(4)
    for col,item in zip(c,[('Total Maintenance',fmt_mw(total),'All categories'),('Planned',fmt_mw(totals.Planned_Maintenance),'Planned'),('Forced',fmt_mw(totals.Forced_Maintenance),'Forced'),('Maintenance Impact',fmt_pct(total/cap*100 if cap else 0),'Maintenance / capacity')]):
        with col: metric_card(*item)
    st.plotly_chart(px.pie(values=totals.values,names=totals.index,hole=.5,title='Maintenance Composition',template='plotly_dark'),use_container_width=True)
    x=summary.sort_values('Maintenance_%',ascending=False).head(20)
    st.subheader('Stations with highest maintenance impact')
    st.dataframe(x[['Power_Station','Capacity','Total_Maintenance','Maintenance_%','Forced_Maintenance_%','Actual','Shortfall','Health_Score','Status']].round(2),use_container_width=True,hide_index=True)

elif page=='🚨 Alert Center':
    st.title('🚨 Smart Alert Center')
    a=make_alerts(df)
    c=st.columns(4)
    counts=a.Severity.value_counts() if not a.empty else pd.Series(dtype=int)
    for col,sev in zip(c,['CRITICAL','HIGH','MEDIUM','LOW']):
        with col: metric_card(sev,int(counts.get(sev,0)),'Alert records')
    if a.empty: st.success('✅ No threshold-based alerts detected.')
    else:
        typ=st.multiselect('Alert Type',sorted(a['Alert Type'].unique()),default=sorted(a['Alert Type'].unique()))
        sev=st.multiselect('Severity',sorted(a['Severity'].unique()),default=sorted(a['Severity'].unique()))
        view=a[a['Alert Type'].isin(typ)&a['Severity'].isin(sev)]
        st.dataframe(view,use_container_width=True,hide_index=True)
        z=view['Alert Type'].value_counts().reset_index(); z.columns=['Alert Type','Count']; st.plotly_chart(px.bar(z,x='Alert Type',y='Count',text_auto=True,title='Alerts by Type',template='plotly_dark'),use_container_width=True)

elif page=='🕵️ Anomaly Detection':
    st.title('🕵️ AI Anomaly Detection')
    features=['Monitored_Capacity','Programme','Actual','Planned_Maintenance','Forced_Maintenance','Other_Reasons','Excess_Shortfall','Deviation']
    d=df[features+['Power_Station']].replace([np.inf,-np.inf],np.nan).fillna(0).copy()
    sample_n=min(len(d),80000); work=d.sample(sample_n,random_state=42) if len(d)>sample_n else d
    contamination=st.slider('Expected anomaly fraction',0.005,0.10,0.02,0.005)
    iso=IsolationForest(n_estimators=150,contamination=contamination,random_state=42,n_jobs=-1)
    work['Anomaly_Label']=iso.fit_predict(work[features]); work['Anomaly_Score']=-iso.score_samples(work[features]); work['Anomaly']=np.where(work.Anomaly_Label==-1,'ANOMALY','NORMAL')
    c=st.columns(3); an=int((work.Anomaly=='ANOMALY').sum())
    for col,item in zip(c,[('Records Analyzed',f'{len(work):,}','Sample if dataset is large'),('Anomalies',f'{an:,}','Isolation Forest'),('Anomaly Rate',fmt_pct(an/len(work)*100 if len(work) else 0),'Detected fraction')]):
        with col: metric_card(*item)
    st.plotly_chart(px.scatter(work.sample(min(12000,len(work)),random_state=1),x='Programme',y='Actual',color='Anomaly',hover_data=['Power_Station','Deviation'],title='Programme vs Actual — Anomaly View',template='plotly_dark'),use_container_width=True)
    st.subheader('Detected anomalies')
    st.dataframe(work[work.Anomaly=='ANOMALY'].sort_values('Anomaly_Score',ascending=False).head(100),use_container_width=True,hide_index=True)

elif page=='📉 Shortfall & Deviation':
    st.title('📉 Shortfall & Deviation Intelligence')
    short=df.loc[df.Excess_Shortfall<0,'Excess_Shortfall']; excess=df.loc[df.Excess_Shortfall>0,'Excess_Shortfall']; dev=df.Deviation
    c=st.columns(5)
    vals=[('Total Shortfall',fmt_mw(short.sum()),'Deficit records'),('Avg Shortfall',fmt_mw(short.mean() if len(short) else 0),'Negative records'),('Total Excess',fmt_mw(excess.sum()),'Excess records'),('Avg Deviation',fmt_mw(dev.mean()),'All records'),('Max |Deviation|',fmt_mw(dev.abs().max()),'Absolute maximum')]
    for col,item in zip(c,vals):
        with col: metric_card(*item)
    x=summary.sort_values('Shortfall').head(20); st.subheader('Worst stations by total shortfall'); st.dataframe(x[['Power_Station','Programme','Actual','Shortfall','Achievement_%','Deviation','Deviation_%']].round(2),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(px.histogram(df,x='Excess_Shortfall',nbins=60,title='Shortfall / Excess Distribution',template='plotly_dark'),use_container_width=True)
    with c2: st.plotly_chart(px.scatter(df.sample(min(12000,len(df)),random_state=42),x='Programme',y='Actual',hover_data=['Power_Station','Deviation'],title='Programme vs Actual',template='plotly_dark'),use_container_width=True)

elif page=='🎯 AI Optimization':
    st.title('🎯 AI Optimization Engine')
    model,metrics=train_fallback_model(df)
    station=st.selectbox('Station',sorted(df.Power_Station.unique()))
    sd=df[df.Power_Station==station]; b=sd[['Monitored_Capacity','Programme','Planned_Maintenance','Forced_Maintenance','Other_Reasons']].mean()
    capacity,programme,planned,forced,other=map(float,[b.Monitored_Capacity,b.Programme,b.Planned_Maintenance,b.Forced_Maintenance,b.Other_Reasons])
    st.info('This optimizer searches simulated maintenance scenarios. It is a decision-support model, not a real plant control command.')
    rows=[]
    for reduction in np.arange(0,0.51,0.05):
        f=forced*(1-reduction); pred=predict_one(model,station,capacity,programme,planned,f,other); ach=pred/programme*100 if programme else 0
        rows.append([reduction*100,f,pred,pred-programme,ach])
    opt=pd.DataFrame(rows,columns=['Forced Maintenance Reduction %','Forced Maintenance','Predicted Actual','Shortfall/Excess','Achievement %'])
    target=st.slider('Minimum desired achievement (%)',50,120,95)
    feasible=opt[opt['Achievement %']>=target]
    best=feasible.iloc[0] if not feasible.empty else opt.iloc[opt['Predicted Actual'].idxmax()]
    st.success(f"🏆 Recommended simulated scenario: reduce forced maintenance by {best['Forced Maintenance Reduction %']:.0f}% → predicted generation {best['Predicted Actual']:.2f} MW and achievement {best['Achievement %']:.2f}%.")
    st.dataframe(opt.round(2),use_container_width=True,hide_index=True)
    st.plotly_chart(px.line(opt,x='Forced Maintenance Reduction %',y='Predicted Actual',markers=True,title='Optimization Curve',template='plotly_dark'),use_container_width=True)

elif page=='🧠 Explainable AI':
    st.title('🧠 Explainable AI')
    model,metrics=train_fallback_model(df)
    # Permutation-style proxy using controlled perturbation of numeric inputs on a representative row.
    station=st.selectbox('Station',sorted(df.Power_Station.unique()),key='xai_station')
    r=df[df.Power_Station==station].iloc[0]
    base=predict_one(model,station,float(r.Monitored_Capacity),float(r.Programme),float(r.Planned_Maintenance),float(r.Forced_Maintenance),float(r.Other_Reasons))
    fields=['Monitored_Capacity','Programme','Planned_Maintenance','Forced_Maintenance','Other_Reasons']
    impacts=[]
    for f in fields:
        vals={k:float(r[k]) for k in fields}; delta=max(abs(vals[f])*0.10,1.0); vals[f]+=delta
        new=predict_one(model,station,vals['Monitored_Capacity'],vals['Programme'],vals['Planned_Maintenance'],vals['Forced_Maintenance'],vals['Other_Reasons'])
        impacts.append([f,new-base])
    imp=pd.DataFrame(impacts,columns=['Feature','Prediction Change for +10%/min 1-unit perturbation']).sort_values('Prediction Change for +10%/min 1-unit perturbation')
    st.metric('Base AI Prediction',fmt_mw(base)); st.dataframe(imp.round(4),use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(imp,x='Prediction Change for +10%/min 1-unit perturbation',y='Feature',orientation='h',title='Local Sensitivity Explanation',template='plotly_dark'),use_container_width=True)
    st.caption('This is local sensitivity analysis, not SHAP. It explains how the embedded model responds to small controlled changes in each input.')

elif page=='📈 Model Performance':
    st.title('📈 Model Performance')
    model,metrics=train_fallback_model(df)
    c=st.columns(3)
    for col,item in zip(c,[('R²',f"{metrics['R2']:.4f}",'Higher is better'),('MAE',f"{metrics['MAE']:.4f} MW",'Lower is better'),('RMSE',f"{metrics['RMSE']:.4f} MW",'Lower is better')]):
        with col: metric_card(*item)
    st.info('The embedded model intentionally excludes Excess/Shortfall and Deviation from model inputs because those are derived from Actual and would leak the target into training.')
    st.subheader('Model inputs'); st.code('Power_Station + Monitored_Capacity + Programme + Planned_Maintenance + Forced_Maintenance + Other_Reasons')

elif page=='📊 Data Analytics':
    st.title('📊 Data Analytics & Research')
    nums=['Monitored_Capacity','Programme','Actual','Planned_Maintenance','Forced_Maintenance','Other_Reasons','Excess_Shortfall','Deviation']
    corr=df[nums].corr()
    st.subheader('Correlation Matrix'); st.plotly_chart(px.imshow(corr,text_auto='.2f',aspect='auto',title='Feature Correlations',template='plotly_dark'),use_container_width=True)
    xcol=st.selectbox('X-axis',nums,index=1); ycol=st.selectbox('Y-axis',nums,index=2); sample=df.sample(min(15000,len(df)),random_state=42)
    st.plotly_chart(px.scatter(sample,x=xcol,y=ycol,hover_data=['Power_Station'],title=f'{xcol} vs {ycol}',template='plotly_dark'),use_container_width=True)
    st.subheader('Descriptive Statistics'); st.dataframe(df[nums].describe().T.round(3),use_container_width=True)

elif page=='📑 Reports':
    st.title('📑 Professional Performance Report')
    station=st.selectbox('Station',sorted(df.Power_Station.unique()))
    d=df[df.Power_Station==station]; s=summary[summary.Power_Station==station].iloc[0]
    status,_=status_for(s)
    report=f'''# PowerGenAI Station Report\n\n## Station\n{station}\n\n## Health\n- Health Score: {s.Health_Score:.2f}/100\n- Status: {status}\n\n## Generation\n- Records: {len(d):,}\n- Programme: {s.Programme:.2f} MW\n- Actual: {s.Actual:.2f} MW\n- Achievement: {s['Achievement_%']:.2f}%\n- Shortfall/Excess: {s.Shortfall:.2f} MW\n- Deviation: {s.Deviation:.2f} MW\n\n## Maintenance\n- Planned: {s.Planned_Maintenance:.2f} MW\n- Forced: {s.Forced_Maintenance:.2f} MW\n- Other: {s.Other_Reasons:.2f} MW\n- Total maintenance: {s.Total_Maintenance:.2f} MW\n- Maintenance impact: {s['Maintenance_%']:.2f}%\n\n## Interpretation\nThis report summarizes the supplied power-generation dataset. Optimization outputs in PowerGenAI are simulated decision-support scenarios and should not be treated as direct plant-control instructions.\n'''
    st.markdown(report); st.download_button('⬇️ Download Markdown Report',report,file_name=f'PowerGenAI_{station}_Report.md',mime='text/markdown',use_container_width=True)

elif page=='📋 Data Explorer':
    st.title('📋 Data Explorer & Quality')
    c=st.columns(4)
    for col,item in zip(c,[('Rows',f'{len(df):,}','Records'),('Columns',f'{len(df.columns):,}','Fields'),('Stations',f'{df.Power_Station.nunique():,}','Unique entities'),('Missing Cells',f'{int(df.isna().sum().sum()):,}','Before filtering')]):
        with col: metric_card(*item)
    station=st.selectbox('Station Filter',['All']+sorted(df.Power_Station.unique())); filtered=df if station=='All' else df[df.Power_Station==station]
    c1,c2,c3=st.columns(3)
    with c1: min_prog,max_prog=float(filtered.Programme.min()),float(filtered.Programme.max()); pr=st.slider('Programme range',min_prog,max_prog,(min_prog,max_prog))
    with c2: min_act,max_act=float(filtered.Actual.min()),float(filtered.Actual.max()); ar=st.slider('Actual range',min_act,max_act,(min_act,max_act))
    with c3: search=st.text_input('🔎 Search station/text')
    filtered=filtered[filtered.Programme.between(*pr)&filtered.Actual.between(*ar)]
    if search: filtered=filtered[filtered.astype(str).apply(lambda x:x.str.contains(search,case=False,na=False)).any(axis=1)]
    st.dataframe(filtered,use_container_width=True,hide_index=True)
    st.download_button('⬇️ Download Filtered CSV',filtered.to_csv(index=False).encode(),file_name='PowerGenAI_filtered.csv',mime='text/csv',use_container_width=True)

st.markdown('<hr><div style="text-align:center;color:#64748b;padding:12px">⚡ <b>PowerGenAI</b> — AI-Based Power Station Performance & Decision Support System</div>',unsafe_allow_html=True)
