import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load('model.pkl') 
except:
    st.error("Erro: O arquivo do modelo Random Forest ('model.pkl') não foi encontrado.")

st.set_page_config(page_title="Churn Predictor | Random Forest", layout="wide")

st.title("📊 Preditor de Churn (Random Forest)")
st.markdown("Insira os dados do cliente para calcular o risco de cancelamento.")

with st.sidebar:
    st.header("Parâmetros do Contrato")
    tenure = st.slider("Meses de Contrato (Tenure)", 0, 72, 12)
    contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Carga Mensal", value=50.0)
    total_charges = st.number_input("Carga Total", value=500.0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Perfil e Social")
    senior = st.radio("Idoso?", ["Não", "Sim"])
    partner = st.radio("Possui Parceiro?", ["Yes", "No"])
    dependents = st.radio("Dependentes?", ["Yes", "No"])
    billing = st.radio("Fatura Digital?", ["Yes", "No"])

with col2:
    st.subheader("Serviços")
    internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Segurança Online", ["Yes", "No", "No internet service"])
    support = st.selectbox("Suporte Técnico", ["Yes", "No", "No internet service"])
    payment = st.selectbox("Método de Pagamento", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

input_dict = {
    'seniorcitizen': 1 if senior == "Sim" else 0,
    'partner': partner,
    'dependents': dependents,
    'tenure': tenure,
    'internetservice': internet,
    'onlinesecurity': security,
    'onlinebackup': "No", 
    'deviceprotection': "No",
    'techsupport': support,
    'streamingtv': "No",
    'streamingmovies': "No",
    'contract': contract,
    'paperlessbilling': billing,
    'paymentmethod': payment,
    'monthlycharges': monthly_charges,
    'totalcharges': total_charges
}

df_input = pd.DataFrame([input_dict])

st.divider()

if st.button("Executar Predição Random Forest", use_container_width=True):
    try:
        
        binary_cols = [
            'partner', 'dependents', 'paperlessbilling', 'onlinesecurity', 
            'onlinebackup', 'deviceprotection', 'techsupport', 
            'streamingtv', 'streamingmovies'
        ]
        for col in binary_cols:
            df_input[col] = df_input[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})

        df_input['contract'] = df_input['contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
        df_input['internetservice'] = df_input['internetservice'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
        
        df_input['paymentmethod'] = df_input['paymentmethod'].apply(lambda x: 1 if x == 'Electronic check' else 0)

        df_input['totalcharges'] = pd.to_numeric(df_input['totalcharges'], errors='coerce').fillna(0)

        expected_columns = [
            'seniorcitizen', 'partner', 'dependents', 'tenure', 'internetservice',
            'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
            'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
            'paymentmethod', 'monthlycharges', 'totalcharges'
        ]
        df_input = df_input[expected_columns]

        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]

        if prediction == 1:
            st.error(f"### ⚠️ ALTA PROBABILIDADE DE CHURN: {probability:.2%}")
        else:
            st.success(f"### ✅ BAIXA PROBABILIDADE DE CHURN: {probability:.2%}")
            
    except Exception as e:
        st.error(f"Erro ao processar predição: {e}")