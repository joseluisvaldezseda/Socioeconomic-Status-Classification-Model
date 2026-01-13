import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuration
st.set_page_config(page_title="Mexico NSE Analytics Engine", layout="wide")

# Modern Vibrant UI Styling
st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    div.block-container {padding-top: 1.5rem;}
    .stMetric {
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #f0f2f6; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    [data-testid="stSidebar"] {background-color: #f8f9fa;}
    h1, h2, h3 {color: #1e293b; font-weight: 800;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Load raw data
    df = pd.read_parquet('MexicoNSE_Predicciones.parquet', engine='pyarrow')
    
    # 1. Clean Column Names (Fix BOM and Duplicates)
    df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 2. Text Repair Logic (Fix Encoding Artifacts like Ã³, Ã±)
    def repair_encoding(text):
        if not isinstance(text, str): return text
        try:
            return text.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

    text_cols = ['nom_ent', 'nom_mun', 'nom_loc']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(repair_encoding).str.strip()

    # 3. GLOBAL CATEGORICAL ORDERING
    nse_order = ['AB', 'C+', 'C', 'C-', 'D+', 'D', 'E']
    df['nse_predicho'] = pd.Categorical(df['nse_predicho'], categories=nse_order, ordered=True)
    
    return df

def main():
    try:
        df = load_data()
        nse_order = ['AB', 'C+', 'C', 'C-', 'D+', 'D', 'E']
        
        # VIBRANT PROFESSIONAL PALETTE
        nse_colors = {
            'AB': '#3E5CC9', # Premium Indigo
            'C+': '#19C5BD', # Bright Teal
            'C':  '#2ECC71', # Vibrant Green
            'C-': '#F4D03F', # Bright Gold
            'D+': '#F39C12', # Vibrant Orange
            'D':  '#E74C3C', # Alizarin Red
            'E':  '#9B59B6'  # Amethyst Purple
        }
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.markdown("### Exploration Filters")
        
        state_list = ["National"] + sorted(df['nom_ent'].unique().tolist())
        state = st.selectbox("Select Territory", state_list)
        
        view_df = df.copy()
        if state != "National":
            view_df = view_df[view_df['nom_ent'] == state]
            
            municipalities = sorted(view_df['nom_mun'].unique().tolist())
            mun_choice = st.multiselect("Municipalities", municipalities)
            if mun_choice:
                view_df = view_df[view_df['nom_mun'].isin(mun_choice)]

        st.markdown("---")
        nse_filter = st.multiselect(
            "Filter NSE Levels", 
            options=nse_order, 
            default=nse_order
        )
        view_df = view_df[view_df['nse_predicho'].isin(nse_filter)]

    # --- HEADER SECTION ---
    st.title("Socio-Economic Intelligence Portal")
    st.markdown(f"**Current Scope:** {state} | **Data Points:** {len(view_df):,} AGEB units")
    st.markdown("---")

    # --- KPI DASHBOARD ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total AGEBs", f"{len(view_df):,}")
    with kpi2:
        st.metric("Population", f"{int(view_df['pobtot'].sum()):,}")
    with kpi3:
        avg_edu = view_df['calc_graproes'].mean()
        st.metric("Avg Schooling", f"{avg_edu:.1f} yrs")
    with kpi4:
        top_nse = view_df['nse_predicho'].mode()[0] if not view_df.empty else "N/A"
        st.metric("Market Leader", str(top_nse))

    # --- PRIMARY ANALYTICS ---
    row1_left, row1_right = st.columns([4, 6])

    with row1_left:
        st.subheader("Demographic Distribution")
        dist = view_df['nse_predicho'].value_counts().sort_index().reset_index()
        dist.columns = ['NSE', 'Units']
        
        fig_dist = px.bar(
            dist, x='NSE', y='Units',
            color='NSE',
            category_orders={"NSE": nse_order},
            color_discrete_map=nse_colors,
            text_auto='.2s',
            template="plotly_white"
        )
        fig_dist.update_layout(showlegend=False, height=450, font=dict(size=12))
        st.plotly_chart(fig_dist, use_container_width=True)

    with row1_right:
        st.subheader("Asset Saturation by Segment")
        assets = ['calc_vph_inter', 'calc_vph_autom', 'calc_vph_pc', 'calc_vph_lavad', 'calc_vph_refri','calc_vph_3ymasc','calc_pder_ss','calc_vph_telef']
        radar_df = view_df.groupby('nse_predicho', observed=True)[assets].mean().reset_index()
        
        fig_line = go.Figure()
        for level in nse_order:
            if level in radar_df['nse_predicho'].values:
                row = radar_df[radar_df['nse_predicho'] == level].iloc[0]
                fig_line.add_trace(go.Scatter(
                    x=['Internet', 'Vehicle', 'Computer', 'Washer', 'Fridge', '3+ Rooms', 'Social Security', 'Telephone'], 
                    y=row[assets],
                    name=level,
                    line=dict(color=nse_colors[level], width=4),
                    marker=dict(size=10),
                    line_shape='spline'
                ))
        
        fig_line.update_layout(
            template="plotly_white", height=450,
            yaxis=dict(range=[0, 105], title="Saturation %", gridcolor='#f0f0f0'),
            xaxis=dict(gridcolor='#f0f0f0'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # --- SECONDARY ANALYTICS ---
    st.markdown("---")
    row2_left, row2_right = st.columns(2)

    with row2_left:
        st.subheader("Educational Attainment Variance")
        fig_box = px.box(
            view_df, x='nse_predicho', y='calc_graproes',
            color='nse_predicho',
            category_orders={"nse_predicho": nse_order},
            color_discrete_map=nse_colors,
            points=False,
            template="plotly_white"
        )
        fig_box.update_layout(showlegend=False, height=450, yaxis_title="Schooling Grade")
        st.plotly_chart(fig_box, use_container_width=True)

    with row2_right:
        st.subheader("Digital Access vs Mobility")
        # --- SCATTER PLOT MAX VISIBILITY ---
        fig_scatter = px.scatter(
            view_df, x='calc_vph_inter', y='calc_vph_autom',
            color='nse_predicho', 
            category_orders={"nse_predicho": nse_order},
            color_discrete_map=nse_colors,
            size='pobtot',
            size_max=20,       # Aumentado drásticamente de 18 a 40
            opacity=1.0,       # Opacidad total para colores sólidos y vibrantes
            labels={'calc_vph_inter': 'Internet Access %', 'calc_vph_autom': 'Vehicle %'},
            template="plotly_white",
            hover_data=['nom_mun', 'nom_loc']
        )
        
        # Forzar un tamaño mínimo y un contorno oscuro para que los puntos "brillen"
        fig_scatter.update_traces(
            marker=dict(
                line=dict(width=1.2, color='#1e293b'), # Contorno oscuro para máximo contraste
                sizemin=6  # Los puntos más pequeños ahora serán visibles
            )
        )
        
        fig_scatter.update_layout(
            height=450, 
            legend_title="NSE Level",
            xaxis=dict(gridcolor='#f0f0f0', zeroline=False),
            yaxis=dict(gridcolor='#f0f0f0', zeroline=False),
            margin=dict(t=20, b=20, l=0, r=0)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- DATA EXPLORER ---
    st.markdown("---")
    with st.expander("View Granular Census Data"):
        st.dataframe(
            view_df[['nom_ent', 'nom_mun', 'nom_loc', 'ageb', 'pobtot', 'calc_graproes', 'nse_predicho']].head(1000),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
