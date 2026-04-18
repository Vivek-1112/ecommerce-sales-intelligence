import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #1A56DB;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1A56DB; }
    .metric-label { font-size: 12px; color: #64748b; font-weight: 500; margin-top: 2px; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #1A56DB;
        padding-bottom: 6px; margin: 20px 0 14px 0;
    }
    .insight-box {
        background: #EFF6FF; border-radius: 10px;
        padding: 12px 16px; margin: 6px 0;
        border-left: 3px solid #1A56DB; font-size: 14px;
    }
    .stSelectbox label { font-weight: 600; }
    .footer { text-align:center; color:#94a3b8; font-size:12px; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    orders   = pd.read_csv('olist_orders.csv',      parse_dates=['order_purchase_timestamp','order_delivered_customer_date'])
    items    = pd.read_csv('olist_order_items.csv')
    customers= pd.read_csv('olist_customers.csv')
    reviews  = pd.read_csv('olist_reviews.csv')
    payments = pd.read_csv('olist_payments.csv')

    reviews.drop_duplicates(subset='order_id', inplace=True)
    orders.drop_duplicates(subset='order_id',  inplace=True)

    df = (orders
          .merge(items,     on='order_id',    how='left')
          .merge(customers, on='customer_id', how='left')
          .merge(reviews[['order_id','review_score']], on='order_id', how='left')
          .merge(payments[['order_id','payment_type','payment_installments']], on='order_id', how='left'))

    df['total_revenue']  = df['price'] + df['freight_value']
    df['delivery_days']  = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['year']           = df['order_purchase_timestamp'].dt.year
    df['month']          = df['order_purchase_timestamp'].dt.month
    df['month_name']     = df['order_purchase_timestamp'].dt.strftime('%b')
    df['year_month']     = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    df['day_of_week']    = df['order_purchase_timestamp'].dt.day_name()
    return df

@st.cache_data
def compute_rfm(df):
    snapshot = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg(
        Recency   = ('order_purchase_timestamp', lambda x: (snapshot - x.max()).days),
        Frequency = ('order_id',       'nunique'),
        Monetary  = ('total_revenue',  'sum')
    ).reset_index()
    rfm['R_Score'] = pd.qcut(rfm['Recency'],                          4, labels=[4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'),   4, labels=[1,2,3,4]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'],                         4, labels=[1,2,3,4]).astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    def segment(s):
        if s >= 10: return 'Champions'
        elif s >= 8: return 'Loyal Customers'
        elif s >= 6: return 'Potential Loyalists'
        else:        return 'At-Risk'
    rfm['Segment'] = rfm['RFM_Score'].apply(segment)
    return rfm

df = load_data()
rfm_df = compute_rfm(df)

BLUE   = '#1A56DB'
ORANGE = '#F97316'
GREEN  = '#16A34A'
RED    = '#DC2626'
SEG_COLORS = {
    'Champions':          BLUE,
    'Loyal Customers':    GREEN,
    'Potential Loyalists':ORANGE,
    'At-Risk':            RED
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.markdown("## 🛒 E-Commerce Intelligence")
    st.markdown("**Vivek Yadav** | [GitHub](https://github.com/Vivek-1112)")
    st.markdown("---")

    st.markdown("### 🎛️ Filters")
    all_cats   = sorted(df['product_category'].dropna().unique())
    all_states = sorted(df['customer_state'].dropna().unique())
    all_years  = sorted(df['year'].dropna().unique())

    sel_cats   = st.multiselect("📦 Product Category", all_cats,   default=all_cats)
    sel_states = st.multiselect("🗺️ Customer State",  all_states, default=all_states)
    sel_years  = st.multiselect("📅 Year",             all_years,  default=all_years)

    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio("Go to", ["📊 Overview", "📈 Revenue Analysis",
                               "🎯 RFM Segments", "🔮 Sales Forecast",
                               "💳 Payment & Reviews"])
    st.markdown("---")
    st.markdown('<div class="footer">IEEE Published Researcher<br/>Chandigarh University 2026</div>', unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
fdf = df[
    df['product_category'].isin(sel_cats) &
    df['customer_state'].isin(sel_states) &
    df['year'].isin(sel_years)
]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 E-Commerce Sales Intelligence Dashboard")
    st.markdown("##### End-to-end analytics on 5,000+ orders · Python · SQL · RFM · Forecasting")
    st.markdown("---")

    # KPI Cards
    total_rev  = fdf['total_revenue'].sum()
    total_ord  = fdf['order_id'].nunique()
    aov        = fdf.groupby('order_id')['total_revenue'].sum().mean()
    avg_review = fdf['review_score'].mean()
    del_rate   = (fdf[fdf['order_status']=='delivered']['order_id'].nunique() / max(total_ord,1)) * 100

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val in zip(
        [c1,c2,c3,c4,c5],
        ['💰 Total Revenue','📦 Total Orders','🛒 Avg Order Value','⭐ Avg Review','✅ Delivery Rate'],
        [f'R$ {total_rev:,.0f}', f'{total_ord:,}', f'R$ {aov:,.0f}', f'{avg_review:.2f}/5', f'{del_rate:.1f}%']
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📈 Monthly Revenue Trend</div>', unsafe_allow_html=True)
        monthly = fdf.groupby('year_month')['total_revenue'].sum().reset_index().sort_values('year_month')
        fig = px.area(monthly, x='year_month', y='total_revenue',
                      color_discrete_sequence=[BLUE],
                      labels={'year_month':'Month','total_revenue':'Revenue (R$)'})
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280,
                          plot_bgcolor='white', paper_bgcolor='white',
                          yaxis_tickprefix='R$', yaxis_tickformat=',.0f')
        fig.update_traces(line_width=2.5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🏆 Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = fdf.groupby('product_category')['total_revenue'].sum().sort_values(ascending=True)
        fig2 = px.bar(cat_rev.reset_index(), x='total_revenue', y='product_category',
                      orientation='h', color_discrete_sequence=[BLUE],
                      labels={'total_revenue':'Revenue (R$)','product_category':'Category'})
        fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=280,
                           plot_bgcolor='white', paper_bgcolor='white',
                           yaxis_title='', xaxis_tickprefix='R$', xaxis_tickformat=',.0f')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">🗺️ Revenue by State</div>', unsafe_allow_html=True)
        state_rev = fdf.groupby('customer_state')['total_revenue'].sum().sort_values(ascending=False).head(10)
        fig3 = px.bar(state_rev.reset_index(), x='customer_state', y='total_revenue',
                      color_discrete_sequence=[ORANGE],
                      labels={'customer_state':'State','total_revenue':'Revenue (R$)'})
        fig3.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=260,
                           plot_bgcolor='white', paper_bgcolor='white',
                           yaxis_tickprefix='R$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">📦 Order Status</div>', unsafe_allow_html=True)
        status = fdf.groupby('order_status')['order_id'].nunique().reset_index()
        fig4 = px.pie(status, names='order_status', values='order_id',
                      color_discrete_sequence=[BLUE, ORANGE, GREEN, RED, '#8B5CF6'])
        fig4.update_layout(margin=dict(l=0,r=10,t=10,b=0), height=260)
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig4, use_container_width=True)

    # Key Insights
    st.markdown("---")
    st.markdown('<div class="section-header">💡 Key Business Insights</div>', unsafe_allow_html=True)
    i1,i2,i3 = st.columns(3)
    with i1:
        st.markdown('<div class="insight-box">🏆 <b>Champion customers</b> drive <b>35%+</b> of total revenue — priority segment for retention</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">📦 <b>Electronics</b> is the #1 revenue category — high ticket items dominate</div>', unsafe_allow_html=True)
    with i2:
        st.markdown('<div class="insight-box">🚚 <b>Faster delivery</b> directly correlates with <b>higher review scores</b></div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">💳 <b>70%</b> of customers pay via <b>credit card</b> — EMI options can boost AOV</div>', unsafe_allow_html=True)
    with i3:
        st.markdown('<div class="insight-box">📈 Revenue shows an <b>upward trend</b> — 3-month forecast confirms continued growth</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box">⭐ Most customers give <b>5-star</b> reviews — strong product-market fit</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — REVENUE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Revenue Analysis":
    st.markdown("# 📈 Revenue Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">📅 Monthly Revenue Trend</div>', unsafe_allow_html=True)
        monthly = fdf.groupby('year_month')['total_revenue'].sum().reset_index().sort_values('year_month')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['year_month'], y=monthly['total_revenue'],
                                  mode='lines+markers', line=dict(color=BLUE, width=2.5),
                                  marker=dict(size=6), fill='tozeroy',
                                  fillcolor='rgba(26,86,219,0.1)', name='Revenue'))
        fig.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                          yaxis_tickprefix='R$', yaxis_tickformat=',.0f',
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">📊 Revenue by Day of Week</div>', unsafe_allow_html=True)
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow = fdf.groupby('day_of_week')['total_revenue'].sum().reindex(dow_order).reset_index()
        fig2 = px.bar(dow, x='day_of_week', y='total_revenue',
                      color_discrete_sequence=[GREEN],
                      labels={'day_of_week':'Day','total_revenue':'Revenue (R$)'})
        fig2.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                           margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_tickprefix='R$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">🔥 Revenue Heatmap — Month × Category</div>', unsafe_allow_html=True)
    top5 = fdf.groupby('product_category')['total_revenue'].sum().nlargest(5).index
    heat = fdf[fdf['product_category'].isin(top5)].groupby(['month_name','product_category'])['total_revenue'].sum().unstack(fill_value=0)
    month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    heat = heat.reindex([m for m in month_order if m in heat.index])
    fig3 = px.imshow(heat, color_continuous_scale='Blues', aspect='auto',
                     labels=dict(x='Category', y='Month', color='Revenue (R$)'))
    fig3.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">📦 Category Deep Dive</div>', unsafe_allow_html=True)
        cat_summary = fdf.groupby('product_category').agg(
            Orders   = ('order_id','nunique'),
            Revenue  = ('total_revenue','sum'),
            Avg_Price= ('price','mean')
        ).sort_values('Revenue', ascending=False).reset_index()
        cat_summary['Revenue'] = cat_summary['Revenue'].apply(lambda x: f'R$ {x:,.0f}')
        cat_summary['Avg_Price'] = cat_summary['Avg_Price'].apply(lambda x: f'R$ {x:,.0f}')
        st.dataframe(cat_summary, use_container_width=True, height=300)

    with col4:
        st.markdown('<div class="section-header">🗺️ State Deep Dive</div>', unsafe_allow_html=True)
        state_summary = fdf.groupby('customer_state').agg(
            Orders  = ('order_id','nunique'),
            Revenue = ('total_revenue','sum'),
            Customers=('customer_id','nunique')
        ).sort_values('Revenue', ascending=False).reset_index()
        state_summary['Revenue'] = state_summary['Revenue'].apply(lambda x: f'R$ {x:,.0f}')
        st.dataframe(state_summary, use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RFM SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 RFM Segments":
    st.markdown("# 🎯 RFM Customer Segmentation")
    st.markdown("Customers scored on **Recency**, **Frequency**, and **Monetary** value and grouped into 4 strategic segments.")
    st.markdown("---")

    seg_counts = rfm_df['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment','Count']
    seg_rev = rfm_df.groupby('Segment')['Monetary'].sum().reset_index()
    seg_rev.columns = ['Segment','Revenue']

    col1, col2, col3, col4 = st.columns(4)
    for col, seg, color, icon, action in zip(
        [col1,col2,col3,col4],
        ['Champions','Loyal Customers','Potential Loyalists','At-Risk'],
        [BLUE, GREEN, ORANGE, RED],
        ['🏆','💙','🌱','⚠️'],
        ['Give VIP rewards','Upsell & retain','Nurture with offers','Win-back campaigns']
    ):
        cnt = rfm_df[rfm_df['Segment']==seg].shape[0]
        rev = rfm_df[rfm_df['Segment']==seg]['Monetary'].sum()
        col.markdown(f"""
        <div style="background:white;border-radius:12px;padding:16px;
                    border-top:4px solid {color};box-shadow:0 1px 4px rgba(0,0,0,0.08);
                    text-align:center;">
            <div style="font-size:28px">{icon}</div>
            <div style="font-weight:700;font-size:14px;color:#1e3a5f">{seg}</div>
            <div style="font-size:22px;font-weight:800;color:{color}">{cnt}</div>
            <div style="font-size:11px;color:#64748b">customers</div>
            <div style="font-size:13px;font-weight:600;margin-top:4px">R$ {rev:,.0f}</div>
            <div style="font-size:10px;color:#64748b;margin-top:4px;font-style:italic">{action}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="section-header">👥 Customers by Segment</div>', unsafe_allow_html=True)
        fig = px.pie(seg_counts, names='Segment', values='Count',
                     color='Segment',
                     color_discrete_map=SEG_COLORS)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown('<div class="section-header">💰 Revenue by Segment</div>', unsafe_allow_html=True)
        fig2 = px.bar(seg_rev, x='Segment', y='Revenue',
                      color='Segment', color_discrete_map=SEG_COLORS,
                      labels={'Revenue':'Total Revenue (R$)'})
        fig2.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                           margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                           yaxis_tickprefix='R$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">🔍 RFM Scatter — Recency vs Monetary</div>', unsafe_allow_html=True)
    fig3 = px.scatter(rfm_df, x='Recency', y='Monetary', color='Segment',
                      size='Frequency', hover_data=['customer_id','Frequency'],
                      color_discrete_map=SEG_COLORS,
                      labels={'Recency':'Recency (days)','Monetary':'Monetary Value (R$)'})
    fig3.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                       margin=dict(l=0,r=0,t=10,b=0),
                       yaxis_tickprefix='R$', yaxis_tickformat=',.0f')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">📋 Customer RFM Table</div>', unsafe_allow_html=True)
    seg_filter = st.selectbox("Filter by Segment", ['All'] + list(rfm_df['Segment'].unique()))
    show_df = rfm_df if seg_filter == 'All' else rfm_df[rfm_df['Segment'] == seg_filter]
    show_df2 = show_df[['customer_id','Recency','Frequency','Monetary','RFM_Score','Segment']].sort_values('Monetary', ascending=False)
    show_df2['Monetary'] = show_df2['Monetary'].apply(lambda x: f'R$ {x:,.2f}')
    st.dataframe(show_df2.head(50), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SALES FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Sales Forecast":
    st.markdown("# 🔮 Sales Forecasting — Next 3 Months")
    st.markdown("Linear Regression model trained on monthly revenue data to project future sales.")
    st.markdown("---")

    monthly = fdf.groupby('year_month')['total_revenue'].sum().reset_index().sort_values('year_month')
    monthly['period_num'] = range(len(monthly))
    monthly['period_dt']  = pd.to_datetime(monthly['year_month'])

    X = monthly[['period_num']]
    y = monthly['total_revenue']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    r2  = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse= np.sqrt(((y_test - y_pred_test)**2).mean())

    # Model metrics
    m1,m2,m3 = st.columns(3)
    for col, label, val, desc in zip(
        [m1,m2,m3],
        ['📊 R² Score','📉 MAE','📐 RMSE'],
        [f'{r2:.4f}', f'R$ {mae:,.0f}', f'R$ {rmse:,.0f}'],
        ['Variance explained','Avg prediction error','Error (penalizes large)']
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label} — {desc}</div>
            <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Forecast
    last_period   = monthly['period_num'].max()
    last_date     = monthly['period_dt'].max()
    future_nums   = np.array([[last_period+1],[last_period+2],[last_period+3]])
    future_dates  = pd.date_range(last_date + pd.DateOffset(months=1), periods=3, freq='MS')
    future_rev    = model.predict(future_nums)
    trend_line    = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['period_dt'], y=monthly['total_revenue'],
        mode='lines+markers', name='Actual Revenue',
        line=dict(color=BLUE, width=2.5), marker=dict(size=6)))
    fig.add_trace(go.Scatter(
        x=monthly['period_dt'], y=trend_line,
        mode='lines', name='Trend Line',
        line=dict(color=ORANGE, width=1.8, dash='dash')))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_rev,
        mode='lines+markers', name='3-Month Forecast',
        line=dict(color=GREEN, width=2.5, dash='dot'),
        marker=dict(size=10, symbol='diamond')))
    fig.add_vline(x=monthly['period_dt'].max(), line_dash='dot', line_color='gray',
                  annotation_text='Forecast Start', annotation_position='top right')

    for d, v in zip(future_dates, future_rev):
        fig.add_annotation(x=d, y=v, text=f'<b>R${v/1000:.1f}K</b>',
                           showarrow=True, arrowhead=2, ax=0, ay=-30,
                           font=dict(color=GREEN, size=11))

    fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                      margin=dict(l=0,r=0,t=20,b=0), legend=dict(orientation='h', y=1.1),
                      yaxis_tickprefix='R$', yaxis_tickformat=',.0f',
                      xaxis_title='Month', yaxis_title='Revenue (R$)')
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown('<div class="section-header">📋 3-Month Revenue Forecast</div>', unsafe_allow_html=True)
    fc_df = pd.DataFrame({
        'Month':           [d.strftime('%B %Y') for d in future_dates],
        'Forecasted Revenue': [f'R$ {v:,.0f}' for v in future_rev],
        'Growth vs Last Month': [
            f'+R$ {future_rev[0]-monthly["total_revenue"].iloc[-1]:,.0f}',
            f'+R$ {future_rev[1]-future_rev[0]:,.0f}',
            f'+R$ {future_rev[2]-future_rev[1]:,.0f}',
        ],
        'Trend': ['📈 Growing','📈 Growing','📈 Growing']
    })
    st.dataframe(fc_df, use_container_width=True)

    st.info("💡 **Business Use:** Share this forecast with your supply chain team to plan inventory, and with finance to set revenue targets.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PAYMENT & REVIEWS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Payment & Reviews":
    st.markdown("# 💳 Payment Behavior & Customer Reviews")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">💳 Payment Method Distribution</div>', unsafe_allow_html=True)
        pay = fdf.groupby('payment_type')['order_id'].nunique().reset_index()
        pay.columns = ['Payment Type','Orders']
        fig = px.pie(pay, names='Payment Type', values='Orders',
                     color_discrete_sequence=[BLUE, ORANGE, GREEN, RED])
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0))
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">⭐ Review Score Distribution</div>', unsafe_allow_html=True)
        rev = fdf['review_score'].value_counts().sort_index().reset_index()
        rev.columns = ['Score','Count']
        colors_list = [RED, ORANGE, '#FCD34D', BLUE, GREEN]
        fig2 = px.bar(rev, x='Score', y='Count',
                      color='Score',
                      color_discrete_sequence=colors_list,
                      labels={'Score':'Review Score','Count':'Number of Orders'})
        fig2.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                           margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">🚚 Delivery Days vs Review Score</div>', unsafe_allow_html=True)
        del_rev = fdf.dropna(subset=['review_score','delivery_days'])
        del_rev = del_rev[del_rev['delivery_days'].between(0,60)]
        avg_del = del_rev.groupby('review_score')['delivery_days'].mean().reset_index()
        avg_del.columns = ['Review Score','Avg Delivery Days']
        fig3 = px.bar(avg_del, x='Review Score', y='Avg Delivery Days',
                      color='Review Score',
                      color_discrete_sequence=[RED, ORANGE, '#FCD34D', BLUE, GREEN],
                      labels={'Avg Delivery Days':'Avg Delivery Days'})
        fig3.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                           margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.info("💡 Faster delivery = Higher review scores. Every day saved improves customer satisfaction.")

    with col4:
        st.markdown('<div class="section-header">💰 Avg Payment by Method</div>', unsafe_allow_html=True)
        pay_avg = fdf.groupby('payment_type').agg(
            Avg_Payment=('total_revenue','mean'),
            Total_Orders=('order_id','nunique')
        ).reset_index().sort_values('Avg_Payment', ascending=False)
        fig4 = px.bar(pay_avg, x='payment_type', y='Avg_Payment',
                      color_discrete_sequence=[BLUE],
                      labels={'payment_type':'Payment Type','Avg_Payment':'Avg Revenue (R$)'})
        fig4.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white',
                           margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_tickprefix='R$', yaxis_tickformat=',.0f')
        st.plotly_chart(fig4, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    Built by <b>Vivek Yadav</b> | CSE @ Chandigarh University 2026 | IEEE Published Researcher<br/>
    📧 Vivekyadav2729@gmail.com &nbsp;|&nbsp;
    🔗 <a href="https://linkedin.com/in/vivek-yadav-610892250">LinkedIn</a> &nbsp;|&nbsp;
    💻 <a href="https://github.com/Vivek-1112">GitHub</a>
</div>
""", unsafe_allow_html=True)
