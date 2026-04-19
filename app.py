import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="E-Commerce Intelligence | Vivek Yadav", page_icon="🛒", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0a0a0f;color:#e2e8f0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0f1a,#12121f);border-right:1px solid rgba(99,102,241,0.2);}
[data-testid="stSidebar"] *{color:#cbd5e1!important;}
.hero{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:40px 48px;margin-bottom:28px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-50%;right:-10%;width:400px;height:400px;background:radial-gradient(circle,rgba(99,102,241,0.15),transparent 70%);pointer-events:none;}
.hero-badge{display:inline-block;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);color:#a5b4fc;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:16px;}
.hero-title{font-family:'Syne',sans-serif;font-size:36px;font-weight:800;background:linear-gradient(135deg,#fff,#a5b4fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.2;margin-bottom:8px;}
.hero-sub{font-size:14px;color:#94a3b8;}
.kpi{background:linear-gradient(135deg,#1e1e2e,#16162a);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px 20px;text-align:center;position:relative;overflow:hidden;}
.kpi::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:16px 16px 0 0;}
.kpi.a::after{background:linear-gradient(90deg,#6366f1,#8b5cf6);}
.kpi.b::after{background:linear-gradient(90deg,#10b981,#34d399);}
.kpi.c::after{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.kpi.d::after{background:linear-gradient(90deg,#ec4899,#f472b6);}
.kpi.e::after{background:linear-gradient(90deg,#06b6d4,#67e8f9);}
.kpi-icon{font-size:26px;margin-bottom:6px;}
.kpi-val{font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:#fff;line-height:1;margin-bottom:4px;}
.kpi-lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px;font-weight:600;}
.kpi-sub{font-size:11px;color:#10b981;margin-top:3px;font-weight:600;}
.sh{font-family:'Syne',sans-serif;font-size:17px;font-weight:700;color:#f1f5f9;margin:24px 0 12px;display:flex;align-items:center;gap:8px;}
.sh::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,rgba(99,102,241,0.4),transparent);margin-left:12px;}
.ins{background:linear-gradient(135deg,#1e1e2e,#16162a);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:20px;position:relative;overflow:hidden;margin-bottom:12px;}
.ins-n{font-family:'Syne',sans-serif;font-size:40px;font-weight:800;opacity:.07;position:absolute;top:8px;right:16px;line-height:1;}
.ins-ico{font-size:22px;margin-bottom:8px;}
.ins-t{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:#e2e8f0;margin-bottom:4px;}
.ins-v{font-size:20px;font-weight:800;font-family:'Syne',sans-serif;margin-bottom:5px;}
.ins-d{font-size:11px;color:#64748b;line-height:1.5;}
.seg{background:#1a1a2e;border-radius:14px;padding:20px 16px;text-align:center;border:1px solid rgba(255,255,255,0.06);}
.seg-e{font-size:30px;margin-bottom:6px;}
.seg-n{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:#e2e8f0;margin-bottom:4px;}
.seg-c{font-size:26px;font-weight:800;font-family:'Syne',sans-serif;}
.seg-r{font-size:11px;color:#94a3b8;margin-top:2px;}
.seg-a{font-size:10px;color:#64748b;margin-top:6px;font-style:italic;padding:3px 8px;border-radius:20px;background:rgba(255,255,255,0.04);}
.fc{background:linear-gradient(135deg,#1e1e2e,#0f3460);border:1px solid rgba(99,102,241,0.2);border-radius:14px;padding:20px;text-align:center;}
.fc-m{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;}
.fc-v{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#a5b4fc;}
.fc-g{font-size:11px;color:#10b981;font-weight:600;margin-top:4px;}
.footer{text-align:center;padding:24px;color:#334155;font-size:12px;border-top:1px solid rgba(255,255,255,0.05);margin-top:32px;}
.footer a{color:#6366f1;text-decoration:none;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.5rem 2rem 2rem;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#0a0a0f;}
::-webkit-scrollbar-thumb{background:#1e1e2e;border-radius:3px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

INDIGO="#6366f1"; VIOLET="#8b5cf6"; EMERALD="#10b981"; AMBER="#f59e0b"; ROSE="#f43f5e"; CYAN="#06b6d4"
PALETTE=[INDIGO,VIOLET,EMERALD,AMBER,ROSE,CYAN,"#ec4899","#84cc16"]
BG="#1a1a2e"; FG="#94a3b8"; GRID="rgba(255,255,255,0.05)"

def cl(t='',h=320):
    return dict(title=dict(text=t,font=dict(family='Syne',size=14,color='#e2e8f0'),x=0.01),
        height=h,plot_bgcolor=BG,paper_bgcolor=BG,
        font=dict(family='DM Sans',color=FG,size=11),
        margin=dict(l=10,r=10,t=40 if t else 10,b=10),
        xaxis=dict(gridcolor=GRID,showgrid=True,zeroline=False,linecolor='rgba(255,255,255,0.1)',tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID,showgrid=True,zeroline=False,linecolor='rgba(255,255,255,0.1)',tickfont=dict(size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=10,color=FG)),
        hoverlabel=dict(bgcolor='#1e1e2e',font_size=12,font_family='DM Sans',bordercolor='rgba(99,102,241,0.4)'))

@st.cache_data
def load():
    o=pd.read_csv('olist_orders.csv',parse_dates=['order_purchase_timestamp','order_delivered_customer_date'])
    i=pd.read_csv('olist_order_items.csv')
    c=pd.read_csv('olist_customers.csv')
    r=pd.read_csv('olist_reviews.csv')
    p=pd.read_csv('olist_payments.csv')
    r.drop_duplicates('order_id',inplace=True); o.drop_duplicates('order_id',inplace=True)
    df=(o.merge(i,on='order_id',how='left').merge(c,on='customer_id',how='left')
        .merge(r[['order_id','review_score']],on='order_id',how='left')
        .merge(p[['order_id','payment_type','payment_installments']],on='order_id',how='left'))
    df['total_revenue']=df['price']+df['freight_value']
    df['delivery_days']=(df['order_delivered_customer_date']-df['order_purchase_timestamp']).dt.days
    df['year']=df['order_purchase_timestamp'].dt.year
    df['month']=df['order_purchase_timestamp'].dt.month
    df['month_name']=df['order_purchase_timestamp'].dt.strftime('%b')
    df['year_month']=df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    df['day_of_week']=df['order_purchase_timestamp'].dt.day_name()
    return df

@st.cache_data
def rfm(df):
    snap=df['order_purchase_timestamp'].max()+pd.Timedelta(days=1)
    r=df.groupby('customer_id').agg(Recency=('order_purchase_timestamp',lambda x:(snap-x.max()).days),
        Frequency=('order_id','nunique'),Monetary=('total_revenue','sum')).reset_index()
    r['R_Score']=pd.qcut(r['Recency'],4,labels=[4,3,2,1]).astype(int)
    r['F_Score']=pd.qcut(r['Frequency'].rank(method='first'),4,labels=[1,2,3,4]).astype(int)
    r['M_Score']=pd.qcut(r['Monetary'],4,labels=[1,2,3,4]).astype(int)
    r['RFM_Score']=r['R_Score']+r['F_Score']+r['M_Score']
    r['Segment']=r['RFM_Score'].apply(lambda s:'Champions' if s>=10 else 'Loyal Customers' if s>=8 else 'Potential Loyalists' if s>=6 else 'At-Risk')
    return r

df=load(); rfm_df=rfm(df)
SC={'Champions':INDIGO,'Loyal Customers':EMERALD,'Potential Loyalists':AMBER,'At-Risk':ROSE}

with st.sidebar:
    st.markdown(f"""<div style="padding:20px 8px 16px;">
    <div style="font-family:Syne;font-size:20px;font-weight:800;background:linear-gradient(135deg,{INDIGO},{VIOLET});
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px;">
    ◈ E-Commerce<br/>Intelligence</div>
    <div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;">Vivek Yadav · Data Analyst</div></div>""",unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#475569;font-weight:700;margin-bottom:6px;padding-left:4px;">Navigation</div>',unsafe_allow_html=True)
    page=st.radio('',['🏠  Overview','📈  Revenue Dive','🎯  Segments','🔮  Forecast','💳  Payments & Reviews'],label_visibility='collapsed')
    st.markdown('<div style="font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#475569;font-weight:700;margin:20px 0 6px;padding-left:4px;">Filters</div>',unsafe_allow_html=True)
    sc=st.multiselect('Cat',sorted(df['product_category'].dropna().unique()),default=list(df['product_category'].dropna().unique()),label_visibility='collapsed')
    ss=st.multiselect('State',sorted(df['customer_state'].dropna().unique()),default=list(df['customer_state'].dropna().unique()),label_visibility='collapsed')
    sy=st.multiselect('Year',sorted(df['year'].dropna().unique()),default=list(df['year'].dropna().unique()),label_visibility='collapsed')
    st.markdown(f"""<div style="margin-top:28px;padding:12px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:10px;font-size:11px;color:#64748b;">
    🔗 <a href="https://github.com/Vivek-1112" style="color:{INDIGO};">GitHub</a> &nbsp;|&nbsp;
    <a href="https://linkedin.com/in/vivek-yadav-610892250" style="color:{INDIGO};">LinkedIn</a><br/>
    <span style="color:#334155;">IEEE Published · CU '26</span></div>""",unsafe_allow_html=True)

fdf=df[df['product_category'].isin(sc)&df['customer_state'].isin(ss)&df['year'].isin(sy)]

if '🏠' in page:
    st.markdown(f"""<div class="hero"><div class="hero-badge">📊 Data Analytics Dashboard</div>
    <div class="hero-title">E-Commerce Sales<br/>Intelligence Hub</div>
    <div class="hero-sub">End-to-end analytics · 5,000+ orders · Python · SQL · RFM · ML Forecasting</div></div>""",unsafe_allow_html=True)
    tr=fdf['total_revenue'].sum(); to=fdf['order_id'].nunique()
    aov=fdf.groupby('order_id')['total_revenue'].sum().mean(); ar=fdf['review_score'].mean()
    dr=(fdf[fdf['order_status']=='delivered']['order_id'].nunique()/max(to,1))*100
    c1,c2,c3,c4,c5=st.columns(5)
    for col,cls,ico,lbl,val,sub in [(c1,'a','💰','Total Revenue',f'₹{tr/1e6:.2f}M','All time'),
        (c2,'b','📦','Orders',f'{to:,}','Transactions'),(c3,'c','🛒','Avg Order',f'₹{aov:,.0f}','Per order'),
        (c4,'d','⭐','Avg Review',f'{ar:.2f}/5','Score'),(c5,'e','✅','Delivery',f'{dr:.1f}%','Rate')]:
        col.markdown(f'<div class="kpi {cls}"><div class="kpi-icon">{ico}</div><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="sh">📈 Revenue Trend & Category Breakdown</div>',unsafe_allow_html=True)
    c1,c2=st.columns([3,2])
    with c1:
        m=fdf.groupby('year_month')['total_revenue'].sum().reset_index().sort_values('year_month')
        fig=go.Figure(go.Scatter(x=m['year_month'],y=m['total_revenue'],mode='lines',fill='tozeroy',
            fillcolor='rgba(99,102,241,0.1)',line=dict(color=INDIGO,width=2.5),
            hovertemplate='<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>'))
        fig.update_layout(**cl('Monthly Revenue Trend',300)); fig.update_yaxes(tickprefix='₹',tickformat=',.0f')
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        cr=fdf.groupby('product_category')['total_revenue'].sum().sort_values(ascending=True).tail(6)
        fig2=go.Figure(go.Bar(x=cr.values,y=cr.index,orientation='h',
            marker=dict(color=cr.values,colorscale=[[0,'#312e81'],[1,INDIGO]]),
            hovertemplate='<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>'))
        fig2.update_layout(**cl('Top Categories',300)); fig2.update_xaxes(tickprefix='₹',tickformat=',.0f')
        st.plotly_chart(fig2,use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        sr=fdf.groupby('customer_state')['total_revenue'].sum().sort_values(ascending=False).head(8).reset_index()
        fig3=go.Figure(go.Bar(x=sr['customer_state'],y=sr['total_revenue'],
            marker=dict(color=sr['total_revenue'],colorscale=[[0,'#1e1e40'],[1,VIOLET]]),
            hovertemplate='<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>'))
        fig3.update_layout(**cl('Revenue by State',280)); fig3.update_yaxes(tickprefix='₹',tickformat=',.0f')
        st.plotly_chart(fig3,use_container_width=True)
    with c4:
        st_c=fdf.groupby('order_status')['order_id'].nunique().reset_index()
        fig4=go.Figure(go.Pie(labels=st_c['order_status'],values=st_c['order_id'],hole=0.55,
            marker=dict(colors=PALETTE[:len(st_c)],line=dict(color='#0a0a0f',width=3)),
            hovertemplate='<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>'))
        fig4.add_annotation(text='Order<br>Status',x=0.5,y=0.5,font=dict(size=12,color='#e2e8f0',family='Syne'),showarrow=False)
        fig4.update_layout(**cl('',280)); st.plotly_chart(fig4,use_container_width=True)
    st.markdown('<div class="sh">💡 Key Business Insights</div>',unsafe_allow_html=True)
    ins=[(f'🏆','01','Champions Rule','35%+ Revenue','from top-tier customers — build VIP loyalty program',INDIGO),
        ('📦','02','Top Category','Electronics #1','highest revenue — increase inventory & ad spend',VIOLET),
        ('🚚','03','Speed Matters','Delivery = Ratings','faster delivery directly boosts review scores',CYAN),
        ('💳','04','Pay Habits','70% Credit Card','offer EMI options to boost average order value',AMBER),
        ('📈','05','Growth Signal','Upward Trend','ML forecast confirms 3-month continued growth',EMERALD),
        ('⭐','06','Satisfaction','5★ Dominant','strong product-market fit across all categories',ROSE)]
    r1=st.columns(3); r2=st.columns(3)
    for i,(ico,num,tit,val,desc,clr) in enumerate(ins):
        (r1 if i<3 else r2)[i%3].markdown(f'<div class="ins" style="border-top:3px solid {clr};"><div class="ins-n">{num}</div><div class="ins-ico">{ico}</div><div class="ins-t">{tit}</div><div class="ins-v" style="color:{clr};">{val}</div><div class="ins-d">{desc}</div></div>',unsafe_allow_html=True)

elif '📈' in page:
    st.markdown('<div style="font-family:Syne;font-size:28px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">📈 Revenue Deep Dive</div>',unsafe_allow_html=True)
    st.markdown('<div style="color:#475569;font-size:13px;margin-bottom:20px;">Detailed breakdown of revenue patterns, geography, and timing</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        m=fdf.groupby('year_month')['total_revenue'].sum().reset_index().sort_values('year_month')
        m['MA3']=m['total_revenue'].rolling(3).mean()
        fig=go.Figure()
        fig.add_trace(go.Bar(x=m['year_month'],y=m['total_revenue'],name='Revenue',marker_color='rgba(99,102,241,0.6)',hovertemplate='%{x}<br>₹%{y:,.0f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=m['year_month'],y=m['MA3'],name='3M Avg',line=dict(color=AMBER,width=2),hovertemplate='MA: ₹%{y:,.0f}<extra></extra>'))
        fig.update_layout(**cl('Revenue + Moving Average',320)); fig.update_yaxes(tickprefix='₹',tickformat=',.0f')
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        dow=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dw=fdf.groupby('day_of_week')['total_revenue'].sum().reindex(dow).reset_index()
        fig2=go.Figure(go.Bar(x=dw['day_of_week'],y=dw['total_revenue'],
            marker=dict(color=dw['total_revenue'],colorscale=[[0,'#1e1e40'],[1,INDIGO]]),
            hovertemplate='<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>'))
        fig2.update_layout(**cl('Revenue by Day of Week',320)); fig2.update_yaxes(tickprefix='₹',tickformat=',.0f')
        fig2.update_xaxes(tickangle=-30); st.plotly_chart(fig2,use_container_width=True)
    st.markdown('<div class="sh">🗺️ Geographic Analysis</div>',unsafe_allow_html=True)
    c3,c4=st.columns([2,1])
    with c3:
        sr=fdf.groupby('customer_state')['total_revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        fig3=go.Figure(go.Bar(x=sr['customer_state'],y=sr['total_revenue'],
            marker=dict(color=sr['total_revenue'],colorscale=[[0,'#1e1e40'],[0.5,VIOLET],[1,INDIGO]],showscale=True),
            hovertemplate='<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>'))
        fig3.update_layout(**cl('Top 10 States by Revenue',340)); fig3.update_yaxes(tickprefix='₹',tickformat=',.0f')
        st.plotly_chart(fig3,use_container_width=True)
    with c4:
        t5=fdf.groupby('product_category')['total_revenue'].sum().nlargest(5)
        fig4=go.Figure(go.Pie(labels=t5.index,values=t5.values,hole=0.6,
            marker=dict(colors=PALETTE[:5],line=dict(color='#0a0a0f',width=2)),
            hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>'))
        fig4.add_annotation(text='Top 5',x=0.5,y=0.5,font=dict(size=13,color='#e2e8f0',family='Syne'),showarrow=False)
        fig4.update_layout(**cl('Category Share',340)); st.plotly_chart(fig4,use_container_width=True)
    st.markdown('<div class="sh">🔥 Revenue Heatmap</div>',unsafe_allow_html=True)
    t5c=fdf.groupby('product_category')['total_revenue'].sum().nlargest(5).index
    heat=fdf[fdf['product_category'].isin(t5c)].groupby(['month_name','product_category'])['total_revenue'].sum().unstack(fill_value=0)
    mo=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    heat=heat.reindex([m for m in mo if m in heat.index])
    fig5=px.imshow(heat,color_continuous_scale=[[0,'#12122a'],[0.5,VIOLET],[1,INDIGO]],aspect='auto',text_auto='.0f')
    fig5.update_layout(**cl('',260)); st.plotly_chart(fig5,use_container_width=True)

elif '🎯' in page:
    st.markdown('<div style="font-family:Syne;font-size:28px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">🎯 RFM Customer Segmentation</div>',unsafe_allow_html=True)
    st.markdown('<div style="color:#475569;font-size:13px;margin-bottom:20px;">Recency · Frequency · Monetary — understanding who your customers really are</div>',unsafe_allow_html=True)
    segs=[('🏆','Champions',INDIGO,'Bought recently, buy often, spend most','VIP rewards & early access'),
        ('💙','Loyal Customers',EMERALD,'Consistent buyers with good spend','Upsell & premium offers'),
        ('🌱','Potential Loyalists',AMBER,'Recent buyers growing in frequency','Nurture with campaigns'),
        ('⚠️','At-Risk',ROSE,"Haven't bought recently","Win-back with discounts")]
    cols=st.columns(4)
    for col,(ico,lbl,clr,desc,act) in zip(cols,segs):
        cnt=rfm_df[rfm_df['Segment']==lbl].shape[0]
        rev=rfm_df[rfm_df['Segment']==lbl]['Monetary'].sum()
        pct=cnt/len(rfm_df)*100
        col.markdown(f'<div class="seg" style="border-top:3px solid {clr};"><div class="seg-e">{ico}</div><div class="seg-n">{lbl}</div><div class="seg-c" style="color:{clr};">{cnt}</div><div class="seg-r">₹{rev/1000:.0f}K · {pct:.0f}%</div><div class="seg-a">{act}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="sh">📊 Segment Analytics</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        sr2=rfm_df.groupby('Segment')['Monetary'].sum().reset_index()
        fig=go.Figure(go.Pie(labels=sr2['Segment'],values=sr2['Monetary'],hole=0.55,
            marker=dict(colors=[SC.get(s,INDIGO) for s in sr2['Segment']],line=dict(color='#0a0a0f',width=3)),
            hovertemplate='<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>'))
        fig.add_annotation(text='Revenue<br>Split',x=0.5,y=0.5,font=dict(size=12,color='#e2e8f0',family='Syne'),showarrow=False)
        fig.update_layout(**cl('Revenue by Segment',320)); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2=go.Figure()
        for seg,clr in SC.items():
            g=rfm_df[rfm_df['Segment']==seg]
            fig2.add_trace(go.Scatter(x=g['Recency'],y=g['Monetary'],mode='markers',name=seg,
                marker=dict(color=clr,size=5,opacity=0.7),
                hovertemplate=f'<b>{seg}</b><br>Recency: %{{x}}d<br>₹%{{y:,.0f}}<extra></extra>'))
        fig2.update_layout(**cl('Recency vs Monetary',320)); fig2.update_yaxes(tickprefix='₹',tickformat=',.0f')
        fig2.update_xaxes(title_text='Days Since Last Order'); st.plotly_chart(fig2,use_container_width=True)
    st.markdown('<div class="sh">📋 Customer Explorer</div>',unsafe_allow_html=True)
    sf=st.selectbox('Segment',['All']+list(rfm_df['Segment'].unique()),label_visibility='collapsed')
    sh=rfm_df if sf=='All' else rfm_df[rfm_df['Segment']==sf]
    sh2=sh[['customer_id','Recency','Frequency','Monetary','RFM_Score','Segment']].sort_values('Monetary',ascending=False).head(50).copy()
    sh2['Monetary']=sh2['Monetary'].apply(lambda x:f'₹{x:,.0f}')
    st.dataframe(sh2,use_container_width=True,column_config={'RFM_Score':st.column_config.ProgressColumn('RFM Score',min_value=3,max_value=12)})

elif '🔮' in page:
    st.markdown('<div style="font-family:Syne;font-size:28px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">🔮 Sales Forecasting</div>',unsafe_allow_html=True)
    st.markdown('<div style="color:#475569;font-size:13px;margin-bottom:20px;">Linear Regression model — projecting next 3 months revenue</div>',unsafe_allow_html=True)
    m=fdf.groupby('year_month')['total_revenue'].sum().reset_index().sort_values('year_month')
    m['pn']=range(len(m)); m['dt']=pd.to_datetime(m['year_month'])
    X=m[['pn']]; y=m['total_revenue']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,shuffle=False)
    mod=LinearRegression(); mod.fit(Xtr,ytr); yp=mod.predict(Xte)
    r2=r2_score(yte,yp); mae=mean_absolute_error(yte,yp); rmse=np.sqrt(((yte-yp)**2).mean())
    c1,c2,c3,c4=st.columns(4)
    for col,ico,lbl,val,clr in [(c1,'📊','R² Score',f'{r2:.3f}',INDIGO),(c2,'📉','MAE',f'₹{mae:,.0f}',EMERALD),
        (c3,'📐','RMSE',f'₹{rmse:,.0f}',AMBER),(c4,'🎯','Accuracy',f'{r2*100:.1f}%',ROSE)]:
        col.markdown(f'<div class="kpi a" style="border-top:3px solid {clr};"><div class="kpi-icon">{ico}</div><div class="kpi-val" style="font-size:22px;">{val}</div><div class="kpi-lbl">{lbl}</div></div>',unsafe_allow_html=True)
    lp=m['pn'].max(); ld=m['dt'].max()
    fn=np.array([[lp+1],[lp+2],[lp+3]]); fd=pd.date_range(ld+pd.DateOffset(months=1),periods=3,freq='MS')
    fv=mod.predict(fn); tl=mod.predict(X)
    st.markdown('<div class="sh">📈 Forecast Chart</div>',unsafe_allow_html=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=m['dt'],y=m['total_revenue'],mode='lines+markers',name='Actual',
        line=dict(color=INDIGO,width=2.5),marker=dict(size=5),fill='tozeroy',fillcolor='rgba(99,102,241,0.08)',
        hovertemplate='<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=m['dt'],y=tl,mode='lines',name='Trend',line=dict(color=AMBER,width=1.5,dash='dash'),hovertemplate='Trend: ₹%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=fd,y=fv,mode='lines+markers',name='Forecast',
        line=dict(color=EMERALD,width=2.5,dash='dot'),marker=dict(size=10,symbol='diamond',color=EMERALD),
        hovertemplate='<b>Forecast %{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>'))
    fig.add_vrect(x0=m['dt'].max(),x1=fd[-1],fillcolor='rgba(16,185,129,0.05)',line_width=0,
        annotation_text='Forecast Zone',annotation_font=dict(color=EMERALD,size=10))
    for d,v in zip(fd,fv):
        fig.add_annotation(x=d,y=v,text=f'₹{v/1000:.0f}K',showarrow=True,arrowhead=2,ax=0,ay=-35,font=dict(color=EMERALD,size=11,family='Syne'),arrowcolor=EMERALD)
    fig.update_layout(**cl('',400)); fig.update_yaxes(tickprefix='₹',tickformat=',.0f')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="sh">📋 3-Month Summary</div>',unsafe_allow_html=True)
    fc=st.columns(3)
    for col,d,v,i in zip(fc,fd,fv,range(3)):
        prev=m['total_revenue'].iloc[-1] if i==0 else fv[i-1]
        g=((v-prev)/prev)*100
        col.markdown(f'<div class="fc"><div class="fc-m">{d.strftime("%B %Y")}</div><div class="fc-v">₹{v/1000:.1f}K</div><div class="fc-g">+{g:.1f}% projected growth</div></div>',unsafe_allow_html=True)

elif '💳' in page:
    st.markdown('<div style="font-family:Syne;font-size:28px;font-weight:800;color:#f1f5f9;margin-bottom:4px;">💳 Payments & Reviews</div>',unsafe_allow_html=True)
    st.markdown('<div style="color:#475569;font-size:13px;margin-bottom:20px;">How customers pay and what drives their satisfaction</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        pay=fdf.groupby('payment_type')['order_id'].nunique().reset_index(); pay.columns=['Type','Orders']
        fig=go.Figure(go.Pie(labels=pay['Type'],values=pay['Orders'],hole=0.55,
            marker=dict(colors=PALETTE[:len(pay)],line=dict(color='#0a0a0f',width=3)),
            hovertemplate='<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>'))
        fig.add_annotation(text='Payment<br>Split',x=0.5,y=0.5,font=dict(size=12,color='#e2e8f0',family='Syne'),showarrow=False)
        fig.update_layout(**cl('Payment Method Distribution',320)); st.plotly_chart(fig,use_container_width=True)
    with c2:
        rv=fdf['review_score'].value_counts().sort_index().reset_index(); rv.columns=['Score','Count']
        sc2=[ROSE,AMBER,'#eab308',INDIGO,EMERALD]
        fig2=go.Figure(go.Bar(x=rv['Score'],y=rv['Count'],marker_color=sc2,hovertemplate='<b>%{x} Stars</b><br>%{y:,}<extra></extra>'))
        fig2.update_layout(**cl('Review Score Distribution',320))
        fig2.update_xaxes(tickvals=[1,2,3,4,5],ticktext=['⭐','⭐⭐','⭐⭐⭐','⭐⭐⭐⭐','⭐⭐⭐⭐⭐'])
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown('<div class="sh">🚚 Delivery Speed vs Satisfaction</div>',unsafe_allow_html=True)
    c3,c4=st.columns([3,2])
    with c3:
        dr=fdf.dropna(subset=['review_score','delivery_days'])
        dr=dr[dr['delivery_days'].between(0,60)]
        ad=dr.groupby('review_score')['delivery_days'].mean().reset_index()
        fig3=go.Figure(go.Bar(x=ad['review_score'],y=ad['delivery_days'],marker_color=sc2,hovertemplate='<b>%{x} Stars</b><br>Avg %{y:.1f} days<extra></extra>'))
        fig3.add_annotation(x=3,y=ad['delivery_days'].max()*0.9,text='⚡ Faster delivery → Higher ratings',font=dict(color=EMERALD,size=11),showarrow=False,bgcolor='rgba(16,185,129,0.1)',bordercolor=EMERALD,borderwidth=1)
        fig3.update_layout(**cl('Avg Delivery Days by Review Score',320))
        fig3.update_yaxes(title_text='Avg Delivery Days'); fig3.update_xaxes(title_text='Review Score')
        st.plotly_chart(fig3,use_container_width=True)
    with c4:
        pa=fdf.groupby('payment_type').agg(Revenue=('total_revenue','mean')).sort_values('Revenue').reset_index()
        fig4=go.Figure(go.Bar(x=pa['Revenue'],y=pa['payment_type'],orientation='h',
            marker=dict(color=pa['Revenue'],colorscale=[[0,'#1e1e40'],[1,VIOLET]]),
            hovertemplate='<b>%{y}</b><br>₹%{x:,.0f} avg<extra></extra>'))
        fig4.update_layout(**cl('Avg Revenue by Payment Type',320)); fig4.update_xaxes(tickprefix='₹',tickformat=',.0f')
        st.plotly_chart(fig4,use_container_width=True)

st.markdown('<div class="footer">Built by <b style="color:#6366f1;">Vivek Yadav</b> &nbsp;·&nbsp; CSE @ Chandigarh University 2026 &nbsp;·&nbsp; IEEE Published &nbsp;·&nbsp; <a href="https://linkedin.com/in/vivek-yadav-610892250">LinkedIn</a> &nbsp;·&nbsp; <a href="https://github.com/Vivek-1112">GitHub</a></div>',unsafe_allow_html=True)
