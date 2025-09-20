# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")
st.title("📊 Telco Customer Churn Dashboard")

# -------------------------------
# Load Dataset
# -------------------------------
try:
    data = pd.read_csv("telco.csv")
except FileNotFoundError:
    st.error("Default dataset 'telco.csv' not found.")
    st.stop()

data.columns = data.columns.str.lower().str.replace(' ', '_')
data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')
data['total_charges'].fillna(data['total_charges'].median(), inplace=True)
data['churn_label'] = data['churn_label'].map({'Yes':1,'No':0})

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filter Data")
contract_options = data['contract'].unique().tolist()
selected_contracts = st.sidebar.multiselect("Contract Type", contract_options, default=contract_options)
min_tenure, max_tenure = int(data['tenure_in_months'].min()), int(data['tenure_in_months'].max())
selected_tenure = st.sidebar.slider("Tenure in Months", min_tenure, max_tenure, (min_tenure, max_tenure))

filtered_data = data[data['contract'].isin(selected_contracts)]
filtered_data = filtered_data[(filtered_data['tenure_in_months'] >= selected_tenure[0]) &
                              (filtered_data['tenure_in_months'] <= selected_tenure[1])]

# -------------------------------
# 1️⃣ KPI Metrics
# -------------------------------
total_customers = len(filtered_data)
churn_count = filtered_data['churn_label'].sum()
churn_rate = round(churn_count / total_customers * 100, 2) if total_customers>0 else 0
avg_monthly_charge = round(filtered_data['monthly_charge'].mean(),2) if total_customers>0 else 0

kpi_names = ["💼 Total Customers", "📉 Churn Count", "📊 Churn Rate", "💰 Avg Monthly Charge"]
kpi_values = [total_customers, churn_count, f"{churn_rate}%", f"${avg_monthly_charge}"]

kpi_table = go.Figure(data=[go.Table(
    header=dict(values=["KPI","Value"], fill_color='midnightblue', font=dict(color='white',size=18), align='center'),
    cells=dict(values=[kpi_names, kpi_values], fill_color='lavender', font=dict(color='black',size=16), align='center', height=40)
)])
kpi_table.update_layout(title=dict(text="📌 Key Performance Indicators", font=dict(size=20)), margin=dict(l=10,r=10,t=30,b=10))
st.plotly_chart(kpi_table, use_container_width=True)

# -------------------------------
# 2️⃣ Insights & Recommendations
# -------------------------------
insights = [
    "Month-to-month contracts have higher churn.",
    "Higher monthly charges increase churn risk.",
    "No device protection plan correlates with higher churn.",
    "Fiber optic internet users churn more than DSL.",
    "Longer tenure and long-term contracts reduce churn."
]
recommendations = [
    "Incentivize month-to-month customers to switch to annual contracts.",
    "Review pricing for high monthly charges to reduce churn.",
    "Promote device protection plans to improve retention.",
    "Provide special support or perks for Fiber optic users.",
    "Encourage longer contract durations through loyalty programs."
]

insight_table = go.Figure(data=[go.Table(
    header=dict(values=["📝 Insight", "✅ Recommendation"], fill_color='darkgreen', font=dict(color='white', size=18), align='center'),
    cells=dict(values=[insights, recommendations],
               fill_color=[['#e6ffe6','#ccffcc','#e6ffe6','#ccffcc','#e6ffe6'],
                           ['#ffd9b3','#ffb84d','#ffd9b3','#ffb84d','#ffd9b3']],
               font=dict(color='black', size=16), align='left', height=60)
)])
insight_table.update_layout(title=dict(text="📌 Insights & Recommendations", font=dict(size=22)), margin=dict(l=10,r=10,t=50,b=10))
st.plotly_chart(insight_table, use_container_width=True)

# -------------------------------
# 3️⃣ Top 5 Feature Importances
# -------------------------------
categorical_cols = filtered_data.select_dtypes(include='object').columns.tolist()
categorical_cols = [c for c in categorical_cols if c != 'churn_label']
for col in categorical_cols:
    le = LabelEncoder()
    filtered_data[col] = le.fit_transform(filtered_data[col].astype(str))

X = filtered_data.drop('churn_label', axis=1)
y = filtered_data['churn_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

importances = rf.feature_importances_
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
top5_feat = feat_imp.sort_values(by='Importance', ascending=False).head(5)

fig_feat = px.bar(top5_feat, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Magma', title='🔥 Top 5 Feature Importances')
fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_feat, use_container_width=True)

# -------------------------------
# 4️⃣ Key Visualizations
# -------------------------------
filtered_data['churn_label_text'] = filtered_data['churn_label'].map({0:'No',1:'Yes'})

fig1 = px.histogram(filtered_data, x='churn_label_text', color='churn_label_text',
                    color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='📊 Churn Distribution')
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.box(filtered_data, x='churn_label_text', y='monthly_charge', color='churn_label_text',
              color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='💵 Monthly Charge vs Churn')
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.box(filtered_data, x='churn_label_text', y='tenure_in_months', color='churn_label_text',
              color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='⏳ Tenure vs Churn')
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.histogram(filtered_data, x='avg_monthly_gb_download', color='churn_label_text',
                    color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='📶 Avg Monthly GB Download vs Churn')
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.histogram(filtered_data, x='number_of_referrals', color='churn_label_text',
                    color_discrete_map={'No':'#2ca02c','Yes':'#d62728'}, title='🤝 Number of Referrals vs Churn')
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------
# 5️⃣ ROC Curve
# -------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                             line=dict(color='blue', width=3)))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Guess',
                             line=dict(color='red', width=2, dash='dash')))
roc_fig.update_layout(title=f"📈 ROC Curve - AUC: {roc_auc:.2f}",
                      xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
st.plotly_chart(roc_fig, use_container_width=True)

# -------------------------------
# 6️⃣ Confusion Matrix & Classification Report
# -------------------------------
st.subheader("🧮 Model Evaluation (Holdout Set)")
st.text("Confusion Matrix:")
st.text(confusion_matrix(y_test, y_pred))
st.text("\nClassification Report:")
st.text(classification_report(y_test, y_pred))
