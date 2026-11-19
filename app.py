"""
Fase 8 — Interfaz: el usuario ingresa los datos de un cliente y obtiene
la probabilidad de abandono, la decisión y una explicación comprensible.

Ejecutar:  streamlit run app.py
(Antes hay que haber corrido src/train.py y src/threshold.py para tener los modelos y models/config.json)
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict import predecir

st.set_page_config(page_title="Predicción de abandono", page_icon="📉", layout="wide")
st.title("¿Este cliente va a abandonar la compañía?")
st.caption("Ingresá los datos del cliente. El modelo estima la probabilidad de abandono y "
           "recomienda si conviene contactarlo, según los costos del negocio "
           "(no detectar un abandono = USD 120; contactar de más = USD 15).")

SI_NO = ["No", "Yes"]
etiqueta = {"Yes": "Sí", "No": "No", "No internet service": "Sin internet",
            "No phone service": "Sin teléfono", "Month-to-month": "Mes a mes",
            "One year": "1 año", "Two year": "2 años", "Fiber optic": "Fibra óptica",
            "Electronic check": "Cheque electrónico", "Mailed check": "Cheque por correo",
            "Bank transfer (automatic)": "Transferencia automática",
            "Credit card (automatic)": "Tarjeta de crédito automática",
            "Male": "Masculino", "Female": "Femenino", "DSL": "DSL"}
fmt = lambda v: etiqueta.get(v, v)

with st.form("cliente"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Contrato y pagos")
        contract = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"], format_func=fmt)
        tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=100, value=2)
        monthly = st.number_input("Cargo mensual (USD)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
        payment = st.selectbox("Forma de pago", ["Electronic check", "Mailed check",
                                                 "Bank transfer (automatic)", "Credit card (automatic)"], format_func=fmt)
        paperless = st.selectbox("Factura digital", SI_NO, index=1, format_func=fmt)
    with c2:
        st.subheader("Servicios")
        internet = st.selectbox("Internet", ["Fiber optic", "DSL", "No"], format_func=fmt)
        opciones_inet = ["No internet service"] if internet == "No" else ["No", "Yes"]
        security = st.selectbox("Seguridad online", opciones_inet, format_func=fmt)
        backup = st.selectbox("Backup online", opciones_inet, format_func=fmt)
        protection = st.selectbox("Protección de equipos", opciones_inet, format_func=fmt)
        support = st.selectbox("Soporte técnico", opciones_inet, format_func=fmt)
        tv = st.selectbox("Streaming TV", opciones_inet, format_func=fmt)
        movies = st.selectbox("Streaming películas", opciones_inet, format_func=fmt)
    with c3:
        st.subheader("Cliente y teléfono")
        gender = st.selectbox("Género", ["Female", "Male"], format_func=fmt)
        senior = st.selectbox("Adulto mayor", [0, 1], format_func=lambda v: "Sí" if v else "No")
        partner = st.selectbox("Tiene pareja", SI_NO, format_func=fmt)
        dependents = st.selectbox("Tiene dependientes", SI_NO, format_func=fmt)
        phone = st.selectbox("Servicio telefónico", SI_NO, index=1, format_func=fmt)
        lines = st.selectbox("Múltiples líneas", ["No", "Yes"] if phone == "Yes" else ["No phone service"], format_func=fmt)
    enviado = st.form_submit_button("Predecir", type="primary")

if enviado:
    cliente = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,  # aproximación: antigüedad × cargo mensual
    }
    r = predecir(cliente)[0]

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Probabilidad de abandono", f"{r['probabilidad_abandono']:.0%}")
    if r["decision"] == "Contactar":
        m2.error(f"**Recomendación: CONTACTAR** (supera el umbral de {r['umbral']:.0%})")
    else:
        m2.success(f"**Recomendación: NO CONTACTAR** (debajo del umbral de {r['umbral']:.0%})")
    st.progress(min(r["probabilidad_abandono"], 1.0))

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**🔺 Lo que más aumenta el riesgo**")
        for f in r["factores_que_suben_riesgo"] or ["—"]:
            st.markdown(f"- {f}")
    with e2:
        st.markdown("**🔻 Lo que más lo reduce**")
        for f in r["factores_que_bajan_riesgo"] or ["—"]:
            st.markdown(f"- {f}")

    st.info(r["explicacion"])
    st.caption("Modelo: Regresión Logística. La explicación compara a este cliente con el cliente promedio "
               "de los datos de entrenamiento. Son asociaciones, no causas. El umbral es bajo (14 %) porque "
               "no detectar un abandono cuesta 8 veces más que contactar a alguien de más.")
