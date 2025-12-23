import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# 1. CONFIGURACIÓN INICIAL
st.set_page_config(page_title="Predicción de Autos", layout="wide")

# Estilo para el botón rojo
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        width: 100%; 
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF2B2B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚗 Predicción del Precio de Carros con Random Forest")
st.subheader("Utilice la barra lateral para ingresar las características del vehículo.")

# 2. FUNCIÓN DE CARGA Y ENTRENAMIENTO
@st.cache_resource 
def load_and_train_model():
    mensaje_carga = st.empty()
    mensaje_carga.info("Iniciando carga y entrenamiento del modelo (Esto solo ocurre la primera vez)...")
    
    try:
        df = pd.read_csv('car_price_prediction.csv')
        
        # Limpieza rápida
        df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce').fillna(0)
        df['Is_Turbo'] = np.where(df['Engine volume'].astype(str).str.contains('Turbo', na=False), 1, 0)
        df['Engine volume'] = pd.to_numeric(df['Engine volume'].str.replace(' Turbo', '', regex=False), errors='coerce')
        df['Mileage'] = pd.to_numeric(df['Mileage'].str.replace(' km', '', regex=False), errors='coerce')

        # Filtrar outliers básicos
        df = df[df['Price'] >= 100].copy()
        Q1, Q3 = df['Price'].quantile(0.25), df['Price'].quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df[(df['Price'] >= Q1 - 1.5*IQR) & (df['Price'] <= Q3 + 1.5*IQR)].copy()

        selected_columns = ['Price', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Fuel type', 'Gear box type']
        df_selected = df_cleaned[selected_columns].copy()

        y = df_selected['Price']
        categorical_cols = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type']
        df_categorical_encoded = pd.get_dummies(df_selected[categorical_cols], drop_first=True)
        
        X = pd.concat([df_selected[['Prod. year']], df_categorical_encoded], axis=1)
        training_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_rf.fit(X_train, y_train)
        
        importancias = pd.Series(model_rf.feature_importances_, index=training_columns)
        
        mensaje_carga.empty()
        success_placeholder = st.empty() 
        success_placeholder.success("✅ Modelo listo para usar.")
        return model_rf, training_columns, df_selected, importancias, success_placeholder

    except Exception as e:
        st.error(f"Error al cargar datos o entrenar: {e}")
        return None, None, None, None

model_rf, training_columns, df_selected, importancias_raw, success_placeholder = load_and_train_model()

# 3. INTERFAZ (SIDEBAR)
if model_rf is not None:
    st.sidebar.header("Variables de Predicción")
    
    prod_year = st.sidebar.number_input("Año:", int(df_selected['Prod. year'].min()), int(df_selected['Prod. year'].max()), 2015)
    manufacturer = st.sidebar.selectbox("Fabricante:", options=df_selected["Manufacturer"].unique())
    
    df_mod = df_selected[df_selected["Manufacturer"] == manufacturer]
    model_car = st.sidebar.selectbox("Modelo:", options=df_mod["Model"].unique())
    
    df_cat = df_mod[df_mod["Model"] == model_car]
    category = st.sidebar.selectbox("Categoría:", options=df_cat["Category"].unique())
    
    fuel_type = st.sidebar.selectbox("Combustible:", options=df_selected["Fuel type"].unique())
    gear_box_type = st.sidebar.selectbox("Transmisión:", options=df_selected["Gear box type"].unique())
    is_turbo = st.sidebar.checkbox("¿Es Turbo?", value=False)
    

    # BOTÓN DE PREDICCIÓN EN EL SIDEBAR
    if st.sidebar.button("Obtener Predicción"):
        success_placeholder.empty()
        X_pred = pd.DataFrame(data=0, index=[0], columns=training_columns)
        X_pred['Prod. year'] = prod_year
        if 'Is_Turbo' in training_columns:
            X_pred['Is_Turbo'] = 1 if is_turbo else 0
             
        for col, val in zip(['Manufacturer', 'Model', 'Category', 'Fuel type', 'Gear box type'], 
                            [manufacturer, model_car, category, fuel_type, gear_box_type]):
            col_name = f"{col}_{val}"
            if col_name in training_columns:
                X_pred[col_name] = 1

        try:
            prediction = model_rf.predict(X_pred)[0]
            st.markdown("---")
            
            # --- RESULTADOS ---
            col_izq, col_der = st.columns(2)
            with col_izq:
                st.markdown(f"""
                    <div style="background-color:#1E3A8A; color:white; padding:25px; border-radius:15px; text-align:center;">
                        <h3 style="color:white; margin:0;">Precio Predicho</h3>
                        <h1 style="color:white; margin:10px 0;">${prediction:,.2f} USD</h1>
                    </div>
                """, unsafe_allow_html=True)

            with col_der:
                st.markdown(f"""
                    <div style="background-color:#374151; color:white; padding:20px; border-radius:15px; font-size: 14px;">
                        <h4 style="color:white; margin-top:0;">Resumen del Vehículo</h4>
                        <hr style="margin:10px 0; border:0.5px solid #4B5563;">
                        <b>Fabricante:</b> {manufacturer}<br>
                        <b>Modelo:</b> {model_car} | <b>Año:</b> {prod_year}<br>
                        <b>Categoría:</b> {category}<br>
                        <b>Combustible:</b> {fuel_type} | <b>Transmisión:</b> {gear_box_type}
                    </div>
                """, unsafe_allow_html=True)

            # --- GRÁFICO DE IMPORTANCIA ---
            st.write("##")
            st.subheader("📊 Importancia de Variables en esta Predicción")
            
            feat_imp = importancias_raw.sort_values(ascending=False).head(7)
            # Creamos un DF para que Plotly asigne colores distintos por nombre
            df_grafico = pd.DataFrame({
                'Variable': feat_imp.index,
                'Importancia': feat_imp.values
            })
            
            fig = px.bar(
                df_grafico,
                x='Importancia',
                y='Variable',
                orientation='h',
                color='Variable',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                labels={'Importancia': 'Nivel de Influencia', 'Variable': 'Característica'}
            )
            fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
