import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

# =========================================================
# 1. إعدادات الشاشة والترويسة الرسمية
# =========================================================
st.set_page_config(
    page_title="Sabafon AI Network Guardian", page_icon="📡", layout="wide"
)

# ترويسة مشروع التخرج
st.markdown(
    """
    <div style="background-color:#004B87;padding:15px;border-radius:10px;text-align:center;color:white;">
        <h2>📡 مشروع بحث تخرج: نظام Sabafon AI Guardian</h2>
        <h4>الأكاديمية العليا للقرآن الكريم وعلومه - كلية الإعلام - قسم الإعلام</h4>
        <p><b>إعداد الطلاب:</b> عاصف الزبيري | خالد العبيدي | جلال سوار | أحمد خميس</p>
        <p><b>تحت إشراف الدكتور:</b> محمد القليصي</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# =========================================================
# 2. توليد البيانات الافتراضية محاكاة لشبكة سبافون (Data Simulation)
# =========================================================
@st.cache_data
def generate_sabafon_data():
    np.random.seed(42)
    n_towers = 250

    # إحداثيات تقريبية لمواقع الأبراج في صنعاء والمحافظات
    latitudes = 15.3694 + np.random.uniform(-0.08, 0.08, n_towers)
    longitudes = 44.1910 + np.random.uniform(-0.08, 0.08, n_towers)

    # المؤشرات التقنية للشبكة
    temperature = np.random.uniform(25, 85, n_towers)  # حرارة البرج C°
    power_instability = np.random.uniform(
        0, 35, n_towers
    )  # نسبة تذبذب الطاقة %
    traffic_load = np.random.uniform(
        10, 100, n_towers
    )  # الضغط وحركة المرور %
    dropped_calls = np.random.uniform(
        0, 12, n_towers
    )  # نسبة الانقطاع في المكالمات %

    # تحديد خطر العطل (1 = خطر عطل وشيك، 0 = مستقر)
    risk_score = (
        (temperature > 70).astype(int)
        + (power_instability > 20).astype(int)
        + (dropped_calls > 7).astype(int)
    )
    failure_risk = [1 if risk >= 2 else 0 for risk in risk_score]

    df = pd.DataFrame(
        {
            "Tower_ID": [f"SAB-TOWER-{i+100}" for i in range(n_towers)],
            "Latitude": latitudes,
            "Longitude": longitudes,
            "Temperature_C": temperature,
            "Power_Instability_%": power_instability,
            "Traffic_Load_%": traffic_load,
            "Dropped_Calls_%": dropped_calls,
            "Failure_Risk": failure_risk,
        }
    )
    return df


df_data = generate_sabafon_data()

# =========================================================
# 3. تدريب نموذج الذكاء الاصطناعي (Machine Learning Model)
# =========================================================
X = df_data[
    [
        "Temperature_C",
        "Power_Instability_%",
        "Traffic_Load_%",
        "Dropped_Calls_%",
    ]
]
y = df_data["Failure_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
ai_model = RandomForestClassifier(n_estimators=100, random_state=42)
ai_model.fit(X_train, y_train)

# =========================================================
# 4. لوحة الإحصائيات (KPI Dashboard)
# =========================================================
total_towers = len(df_data)
at_risk_towers = len(df_data[df_data["Failure_Risk"] == 1])
safe_towers = len(df_data[df_data["Failure_Risk"] == 0])
model_acc = ai_model.score(X_test, y_test) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("إجمالي أبراج سبافون", total_towers)
c2.metric("أبراج تعمل بكفاءة 🟢", safe_towers)
c3.metric("أبراج تحت الخطر (تنبؤ) 🔴", at_risk_towers)
c4.metric("دقة الذكاء الاصطناعي", f"{model_acc:.1f}%")

st.markdown("---")

# =========================================================
# 5. عرض الخريطة التفاعلية
# =========================================================
st.subheader("🗺️ خريطة التغطية والمراقبة الاستباقية لأبراج سبافون")

fig = px.scatter_mapbox(
    df_data,
    lat="Latitude",
    lon="Longitude",
    color="Failure_Risk",
    color_discrete_map={0: "green", 1: "red"},
    size="Traffic_Load_%",
    hover_name="Tower_ID",
    hover_data=[
        "Temperature_C",
        "Power_Instability_%",
        "Traffic_Load_%",
        "Dropped_Calls_%",
    ],
    zoom=10,
    height=480,
)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 6. أداة الفحص والتنبؤ الفوري لمستشعرات البرج
# =========================================================
st.markdown("---")
st.subheader("🧪 أداة المهندس: فحص واختبار حالة برج محدد")

col_left, col_right = st.columns(2)

with col_left:
    input_temp = st.slider("درجة حرارة المولد والبرج (C°)", 20.0, 100.0, 40.0)
    input_power = st.slider("نسبة تذبذب التيار الكهربائي (%)", 0.0, 50.0, 10.0)

with col_right:
    input_load = st.slider("نسبة الضغط وحركة المرور (%)", 0.0, 100.0, 60.0)
    input_drops = st.slider("نسبة انقطاع المكالمات (%)", 0.0, 20.0, 2.0)

# إجراء التنبؤ
sample_input = pd.DataFrame(
    [[input_temp, input_power, input_load, input_drops]], columns=X.columns
)
pred = ai_model.predict(sample_input)[0]
prob = ai_model.predict_proba(sample_input)[0][1] * 100

st.write("### 📊 نتيجة تحليل نظام الذكاء الاصطناعي:")
if pred == 1:
    st.error(
        f"⚠️ **تحذير: عطل وشيك!** احتمالية الفشل هي **{prob:.1f}%**. يوصى بإرسال فريق الصيانة الميداني قبل انقطاع الخدمة."
    )
else:
    st.success(
        f"✅ **البرج بحالة جيدة.** نسبة الخطر المتوقعة هي **{prob:.1f}%** فقط."
)
