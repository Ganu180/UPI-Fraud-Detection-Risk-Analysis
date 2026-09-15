import streamlit as st
import requests
import pandas as pd
from datetime import datetime


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="🔐",
    layout="wide"
)

API_URL = "https://upi-fraud-detection-api-k9r1.onrender.com"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 16px;
    color: #7a7a7a;
    margin-bottom: 30px;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🔐 Fraud Platform")

    st.write(
        "Machine Learning + Anomaly Detection + Risk Scoring"
    )

    st.divider()

    st.subheader("API Status")

    try:

        health = requests.get(
            f"{API_URL}/health",
            timeout=2
        )

        if health.status_code == 200:
            st.success("FastAPI Online")

        else:
            st.error("FastAPI Error")

    except:
        st.error("FastAPI Offline")

    st.divider()

    st.write("### System")

    st.write("🤖 LightGBM Classification")
    st.write("🔎 Isolation Forest")
    st.write("📊 Behavioral Risk Engine")
    st.write("⚡ FastAPI Backend")
    st.write("🖥️ Streamlit Dashboard")

    st.divider()

    if st.button("Clear Transaction History"):

        st.session_state.history = []
        st.session_state.latest_result = None

        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<p class="main-title">🔐 UPI Fraud Detection & Risk Analysis</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Synthetic UPI-style transaction fraud detection platform'
    '</p>',
    unsafe_allow_html=True
)


# =========================================================
# TRANSACTION INPUT
# =========================================================

st.subheader("💳 Transaction Details")

with st.form("transaction_form"):

    col1, col2, col3 = st.columns(3)


    # -----------------------------------------------------
    # COLUMN 1
    # -----------------------------------------------------

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        sender_age = st.number_input(
            "Sender Age",
            min_value=18,
            max_value=100,
            value=30
        )

        receiver_type = st.selectbox(
            "Receiver Type",
            [
                "Person",
                "Merchant",
                "OnlineShopping",
                "Utility"
            ]
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "P2P",
                "P2M",
                "BillPayment",
                "Recharge"
            ]
        )


    # -----------------------------------------------------
    # COLUMN 2
    # -----------------------------------------------------

    with col2:

        device_id = st.text_input(
            "Device ID",
            "DEV00001"
        )

        location = st.selectbox(
            "Location",
            [
                "Mumbai",
                "Pune",
                "Delhi",
                "Bengaluru",
                "Hyderabad",
                "Chennai",
                "Kolkata",
                "Ahmedabad"
            ]
        )

        hour = st.slider(
            "Transaction Hour",
            0,
            23,
            12
        )

        account_age_days = st.number_input(
            "Account Age (Days)",
            min_value=0,
            value=365
        )


    # -----------------------------------------------------
    # COLUMN 3
    # -----------------------------------------------------

    with col3:

        transactions_last_24h = st.number_input(
            "Transactions Last 24h",
            min_value=0,
            value=2
        )

        avg_amount_30d = st.number_input(
            "Average Amount 30 Days",
            min_value=1.0,
            value=800.0
        )

        failed_attempts = st.number_input(
            "Failed Attempts",
            min_value=0,
            value=0
        )

        distance = st.number_input(
            "Distance From Usual Location (KM)",
            min_value=0.0,
            value=2.0
        )


    col4, col5 = st.columns(2)


    with col4:

        new_receiver = st.selectbox(
            "New Receiver?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )


    with col5:

        new_device = st.selectbox(
            "New Device?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )


    analyze = st.form_submit_button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )


# =========================================================
# API REQUEST
# =========================================================

if analyze:

    transaction = {

        "amount": amount,
        "sender_age": sender_age,
        "receiver_type": receiver_type,
        "transaction_type": transaction_type,
        "device_id": device_id,
        "location": location,
        "hour": hour,
        "account_age_days": account_age_days,
        "transactions_last_24h": transactions_last_24h,
        "avg_amount_30d": avg_amount_30d,
        "new_receiver": new_receiver,
        "new_device": new_device,
        "failed_attempts": failed_attempts,
        "distance_from_usual_location_km": distance
    }


    try:

        with st.spinner(
            "Analyzing transaction..."
        ):

            response = requests.post(
                f"{API_URL}/predict",
                json=transaction,
                timeout=10
            )


        if response.status_code == 200:

            result = response.json()

            st.session_state.latest_result = result


            # ---------------------------------------------
            # SAVE TRANSACTION HISTORY
            # ---------------------------------------------

            history_record = {

                "Time": datetime.now().strftime(
                    "%H:%M:%S"
                ),

                "Amount": amount,

                "Type": transaction_type,

                "Location": location,

                "Fraud Probability":
                    result["fraud_probability"],

                "Anomaly Score":
                    result["anomaly_score"],

                "Risk Score":
                    result["risk_score"],

                "Risk Level":
                    result["risk_level"],

                "Risk Signals":
                    result["risk_signal_count"]
            }


            st.session_state.history.append(
                history_record
            )


        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.code(response.text)


    except requests.exceptions.ConnectionError:

        st.error(
            "FastAPI server is not running."
        )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )


# =========================================================
# LATEST RESULT
# =========================================================

result = st.session_state.latest_result


if result:

    st.divider()

    st.subheader("📊 Fraud Risk Analysis")


    # =====================================================
    # MAIN METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Fraud Probability",
        f"{result['fraud_probability']}%"
    )


    c2.metric(
        "Anomaly Score",
        f"{result['anomaly_score']}/100"
    )


    c3.metric(
        "Behavior Score",
        f"{result['behavior_score']}/100"
    )


    c4.metric(
        "Risk Signals",
        result["risk_signal_count"]
    )


    # =====================================================
    # RISK SCORE
    # =====================================================

    st.subheader("🎯 Overall Risk Score")

    risk_score = result["risk_score"]

    st.progress(
        min(float(risk_score), 100) / 100
    )

    st.write(
        f"### {risk_score} / 100"
    )


    # =====================================================
    # RISK LEVEL
    # =====================================================

    risk = result["risk_level"]


    if risk == "HIGH":

        st.error(
            f"🚨 HIGH RISK TRANSACTION — Risk Score: {risk_score}/100"
        )


    elif risk == "MEDIUM":

        st.warning(
            f"⚠️ MEDIUM RISK TRANSACTION — Risk Score: {risk_score}/100"
        )


    else:

        st.success(
            f"✅ LOW RISK TRANSACTION — Risk Score: {risk_score}/100"
        )


    # =====================================================
    # ADDITIONAL DETAILS
    # =====================================================

    d1, d2, d3 = st.columns(3)


    d1.metric(
        "Anomaly Flag",
        result["anomaly_flag"]
    )


    d2.metric(
        "Behavior Score",
        result["behavior_score"]
    )


    d3.metric(
        "Detected Signals",
        result["risk_signal_count"]
    )


    # =====================================================
    # RISK REASONS
    # =====================================================

    st.subheader(
        "🔎 Why was this transaction flagged?"
    )


    for reason in result["reasons"]:

        st.write(
            f"• {reason}"
        )


# =========================================================
# TRANSACTION HISTORY
# =========================================================

if len(st.session_state.history) > 0:

    st.divider()

    st.subheader(
        "📜 Transaction Analysis History"
    )


    history_df = pd.DataFrame(
        st.session_state.history
    )


    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # ANALYTICS
    # =====================================================

    st.subheader(
        "📈 Session Analytics"
    )


    a1, a2, a3, a4 = st.columns(4)


    total_transactions = len(
        history_df
    )


    high_risk_count = (
        history_df["Risk Level"] == "HIGH"
    ).sum()


    medium_risk_count = (
        history_df["Risk Level"] == "MEDIUM"
    ).sum()


    low_risk_count = (
        history_df["Risk Level"] == "LOW"
    ).sum()


    a1.metric(
        "Transactions",
        total_transactions
    )


    a2.metric(
        "High Risk",
        int(high_risk_count)
    )


    a3.metric(
        "Medium Risk",
        int(medium_risk_count)
    )


    a4.metric(
        "Low Risk",
        int(low_risk_count)
    )


    # =====================================================
    # RISK LEVEL CHART
    # =====================================================

    chart1, chart2 = st.columns(2)


    with chart1:

        st.write(
            "#### Risk Level Distribution"
        )

        risk_counts = (
            history_df["Risk Level"]
            .value_counts()
        )

        st.bar_chart(
            risk_counts
        )


    # =====================================================
    # RISK SCORE TREND
    # =====================================================

    with chart2:

        st.write(
            "#### Risk Score Trend"
        )

        trend_df = history_df[
            ["Risk Score"]
        ].copy()

        st.line_chart(
            trend_df
        )


    # =====================================================
    # FRAUD PROBABILITY CHART
    # =====================================================

    st.write(
        "#### Fraud Probability by Transaction"
    )


    probability_df = history_df[
        [
            "Fraud Probability",
            "Anomaly Score",
            "Risk Score"
        ]
    ]


    st.line_chart(
        probability_df
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "This project uses synthetic UPI-style transaction data "
    "for educational and portfolio purposes. "
    "It does not use real bank or NPCI transaction data."
)