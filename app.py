import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import unicodedata

# 1. Configuración de la página del dashboard
st.set_page_config(
    page_title="Dashboard Resultados Saber Pro",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dashboard de Resultados Saber Pro en Antioquia")

# 2. Carga y preprocesamiento de datos
# Asegúrate de que esta ruta sea correcta para tu archivo de datos
file = "Resultados_únicos_Saber_11_SOLO ANTIOQUIA.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Copia toda la lógica de limpieza y transformación de tu script original aquí
    # Esta función se ejecutará solo una vez, lo que optimiza el rendimiento.
    
    # Normalización de acentos y nombres de municipios
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].apply(lambda x: unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode('utf-8') if isinstance(x, str) else x)
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace(r'\bDONMATIAS\b', 'DON MATIAS', regex=True)
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('EL SANTUARIO', 'SANTUARIO')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('SANTUARIO', 'EL SANTUARIO')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace(r'YONDO (CASABE)', 'YONDO')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace(r'\bSANTAFE DE ANTIOQUIA\b', 'SANTA FE DE ANTIOQUIA', regex=True)
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('SAN ANDRES DE CUERQUIA','SAN ANDRES')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('SAN ANDRES','SAN ANDRES DE CUERQUIA')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('SAN VICENTE FERRER','SAN VICENTE')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('SAN VICENTE','SAN VICENTE FERRER')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('PUERTO NARE (LA MAGDALENA)','PUERTO NARE')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('CIUDAD BOLIVAR','BOLIVAR')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('BOLIVAR','CIUDAD BOLIVAR')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('EL CARMEN DE VIBORAL', 'CARMEN DE VIBORAL')
    df['COLE_MCPIO_UBICACION'] = df['COLE_MCPIO_UBICACION'].str.replace('CARMEN DE VIBORAL', 'EL CARMEN DE VIBORAL')
    df.loc[df['COLE_COD_MCPIO_UBICACION'] == 5664, 'COLE_MCPIO_UBICACION'] = 'SAN PEDRO DE LOS MILAGROS'

    # Tratamiento de datos faltantes
    df.loc[df['FAMI_TIENEINTERNET'].isna(), 'FAMI_TIENEINTERNET'] = 'Sin información'
    df.loc[df['FAMI_TIENECOMPUTADOR'].isna(), 'FAMI_TIENECOMPUTADOR'] = 'Sin información'
    educacion_madre = {'No sabe':'Sin información', 'No Aplica':'Sin información'}
    df['FAMI_EDUCACIONMADRE'] = df['FAMI_EDUCACIONMADRE'].replace(educacion_madre)
    df.loc[df['FAMI_EDUCACIONMADRE'].isna(), 'FAMI_EDUCACIONMADRE'] = 'Sin información'
    educacion_padre = {'No sabe':'Sin información', 'No Aplica':'Sin información'}
    df['FAMI_EDUCACIONPADRE'] = df['FAMI_EDUCACIONPADRE'].replace(educacion_padre)
    df.loc[df['FAMI_EDUCACIONPADRE'].isna(), 'FAMI_EDUCACIONPADRE'] = 'Sin información'
    df.loc[df['ESTU_GENERO'].isna(), 'ESTU_GENERO'] = 'Sin información'

    # Creación de la columna 'year' y 'periodo'
    df['year'] = df['PERIODO'].astype(str).str[:4].astype(int)
    df['periodo'] = df['PERIODO'].astype(str).str[-1]

    # Muestreo de datos
    # df_sample = df.groupby('year').apply(lambda x: x.sample(n=600, random_state=42, replace=True)).reset_index(drop=True)
    return df

# Carga los datos usando la función optimizada
df_sample = load_data(file)

# 3. Procesamiento para los gráficos principales
columnas_a_seleccionar = [
    'PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS',
    'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA', 'PUNT_GLOBAL'
]
df_sample_final = df_sample.set_index('year')[columnas_a_seleccionar]

df_mediana_anual_sample = df_sample_final.groupby('year').median()
df_materias_mediana_sample = df_mediana_anual_sample.drop(columns='PUNT_GLOBAL')
mediana_general = np.nanmedian(df_materias_mediana_sample.values)

# 4. Creación del dashboard interactivo con Streamlit y Plotly
st.write("### Evolución de Puntajes Medianos por Materia")
st.write(
    """
    Este gráfico muestra la evolución de los puntajes medianos por materia
    desde 2017 hasta 2022, comparados con la mediana general.
    """
)

# Selector de materia interactivo
materias_a_graficar = df_materias_mediana_sample.columns.tolist()
materia_seleccionada = st.selectbox(
    "Selecciona la materia para el análisis:",
    materias_a_graficar
)

# Gráfico de línea interactivo (reemplazando matplotlib)
fig = px.line(
    df_materias_mediana_sample,
    x=df_materias_mediana_sample.index,
    y=materia_seleccionada,
    markers=True,
    labels={'x': 'Año', 'y': 'Puntaje Mediano'},
    title=f'Evolución del Puntaje Mediano en {materia_seleccionada}'
)

fig.add_hline(
    y=mediana_general,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Mediana General: {mediana_general:.2f}",
    annotation_position="bottom right"
)

st.plotly_chart(fig, use_container_width=True)

# 5. Adición de más gráficos interactivos (opcional, pero recomendado)
st.write("---")
st.write("### Puntaje Promedio por Naturaleza del Colegio")
st.write(
    """
    Compara los puntajes promedio de los estudiantes de colegios
    de carácter público y privado.
    """
)

average_scores_by_nature = df_sample.groupby('COLE_NATURALEZA')[
    ['PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA', 'PUNT_GLOBAL']
].mean().reset_index()

fig_bar = px.bar(
    average_scores_by_nature,
    x='COLE_NATURALEZA',
    y='PUNT_GLOBAL',
    color='COLE_NATURALEZA',
    title='Puntaje Promedio Global por Naturaleza del Colegio'
)

st.plotly_chart(fig_bar, use_container_width=True)


# graficar los promedios 
# extraemos el promedio global por materia
promedio_ingles = df_sample['PUNT_INGLES'].mean().round(2)
promedio_matematicas = df_sample['PUNT_MATEMATICAS'].mean().round(2)
promedio_lectura_critica = df_sample['PUNT_LECTURA_CRITICA'].mean().round(2)
promedio_ciencias_naturales = df_sample['PUNT_C_NATURALES'].mean().round(2)
promedio_ciencias_sociales = df_sample['PUNT_SOCIALES_CIUDADANAS'].mean().round(2)

# Crea los datos para el gráfico en un DataFrame de Pandas (ideal para Plotly)
df_promedios = pd.DataFrame({
    'Materia': ['Inglés', 'Matemáticas', 'Lectura Crítica', 'Ciencias Naturales', 'Ciencias Sociales'],
    'Puntaje Promedio': [promedio_ingles, promedio_matematicas, promedio_lectura_critica, promedio_ciencias_naturales, promedio_ciencias_sociales]
})

# Crea el gráfico de barras interactivo con Plotly Express
fig = px.bar(
    df_promedios,
    x='Materia',
    y='Puntaje Promedio',
    title='Puntajes Promedio Globales por Materia en Antioquia desde 2017 hasta 2022',
    labels={'Materia': 'Materias', 'Puntaje Promedio': 'Puntaje Promedio'},
    color='Materia',
    text='Puntaje Promedio'
)

# Configura el rango del eje Y para que vaya de 0 a 100, como lo tenías en matplotlib
fig.update_yaxes(range=[0, 100])

# Muestra el gráfico en Streamlit
st.plotly_chart(fig, use_container_width=True)

# Muestra un separador para organizar la interfaz
st.markdown("---")
st.header("Análisis de Puntajes por Naturaleza y Recursos del Colegio")

# Gráfico de conteo de estudiantes por naturaleza del colegio
st.subheader("Número de Estudiantes por Año y Naturaleza del Colegio")
conteo_df = df_sample.groupby(['year', 'COLE_NATURALEZA']).size().reset_index(name='numero_de_estudiantes')
fig_conteo = px.bar(
    conteo_df,
    x='year',
    y='numero_de_estudiantes',
    color='COLE_NATURALEZA',
    barmode='group',
    title='Conteo de Estudiantes por Año y Naturaleza del Colegio'
)
st.plotly_chart(fig_conteo, use_container_width=True)

# Gráfico de puntajes promedio por naturaleza del colegio
st.subheader("Puntaje Promedio por Naturaleza del Colegio y Materia")
st.write("Este gráfico compara los puntajes promedio en todas las materias, diferenciando entre colegios públicos y privados.")

average_scores_by_nature = df_sample.groupby('COLE_NATURALEZA')[
    ['PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA', 'PUNT_GLOBAL']
].mean().reset_index()

# Derretimos el DataFrame para que Plotly pueda graficarlo fácilmente
melted_scores = average_scores_by_nature.melt(
    id_vars='COLE_NATURALEZA',
    var_name='Materia',
    value_name='Puntaje Promedio'
)

# Creamos el gráfico de barras interactivo
fig_avg_nature = px.bar(
    melted_scores,
    x='COLE_NATURALEZA',
    y='Puntaje Promedio',
    color='Materia',
    barmode='group',
    title='Puntaje Promedio por Naturaleza del Colegio'
)
st.plotly_chart(fig_avg_nature, use_container_width=True)

# Puntaje Promedio por Educación de los Padres
st.subheader("Puntaje Promedio según el Nivel de Educación de los Padres")
st.write(
    "Estos gráficos muestran cómo el puntaje global varía en función del nivel de educación de la madre y del padre."
)

# Definimos el orden personalizado para las categorías de educación
education_order = [
    'Sin información',
    'Ninguno',
    'Primaria incompleta',
    'Primaria completa',
    'Secundaria (Bachillerato) incompleta',
    'Secundaria (Bachillerato) completa',
    'Técnica o tecnológica incompleta',
    'Técnica o tecnológica completa',
    'Educación profesional incompleta',
    'Educación profesional completa',
    'Postgrado'
]

# Gráfico de Puntaje Global vs. Educación de la Madre
avg_score_by_mom_edu = df_sample.groupby('FAMI_EDUCACIONMADRE')['PUNT_GLOBAL'].mean().reindex(education_order).dropna().reset_index()
fig_mom_edu = px.bar(
    avg_score_by_mom_edu,
    x='FAMI_EDUCACIONMADRE',
    y='PUNT_GLOBAL',
    title='Puntaje Global Promedio vs. Educación de la Madre',
    labels={'FAMI_EDUCACIONMADRE': 'Nivel de Educación de la Madre', 'PUNT_GLOBAL': 'Puntaje Global Promedio'}
)
fig_mom_edu.update_xaxes(tickangle=45)
st.plotly_chart(fig_mom_edu, use_container_width=True)

# Gráfico de Puntaje Global vs. Educación del Padre
avg_score_by_dad_edu = df_sample.groupby('FAMI_EDUCACIONPADRE')['PUNT_GLOBAL'].mean().reindex(education_order).dropna().reset_index()
fig_dad_edu = px.bar(
    avg_score_by_dad_edu,
    x='FAMI_EDUCACIONPADRE',
    y='PUNT_GLOBAL',
    title='Puntaje Global Promedio vs. Educación del Padre',
    labels={'FAMI_EDUCACIONPADRE': 'Nivel de Educación del Padre', 'PUNT_GLOBAL': 'Puntaje Global Promedio'}
)
fig_dad_edu.update_xaxes(tickangle=45)
st.plotly_chart(fig_dad_edu, use_container_width=True)

# Gráficos de Puntaje Global vs. Recursos Familiares
st.subheader("Puntaje Promedio según Recursos Familiares")
st.write(
    "Estos gráficos exploran la relación entre el puntaje global y la disponibilidad de recursos como el estrato de la vivienda, computador e internet en el hogar."
)

family_resource_columns = ['FAMI_ESTRATOVIVIENDA', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET']
column_names_spanish = {
    'FAMI_ESTRATOVIVIENDA': 'Estrato de Vivienda',
    'FAMI_TIENECOMPUTADOR': 'Tiene Computador',
    'FAMI_TIENEINTERNET': 'Tiene Internet'
}

# Definimos el orden para los estratos
estrato_order = [
    'Sin Estrato', 'Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 'Estrato 5', 'Estrato 6'
]

for col in family_resource_columns:
    avg_scores = df_sample.groupby(col)['PUNT_GLOBAL'].mean().reset_index()

    # Si es el estrato, reordenamos las barras
    if col == 'FAMI_ESTRATOVIVIENDA':
        avg_scores[col] = pd.Categorical(avg_scores[col], categories=estrato_order, ordered=True)
        avg_scores = avg_scores.sort_values(col).dropna()
        
    fig = px.bar(
        avg_scores,
        x=col,
        y='PUNT_GLOBAL',
        title=f'Puntaje Global Promedio por {column_names_spanish[col]}',
        labels={col: column_names_spanish[col], 'PUNT_GLOBAL': 'Puntaje Global Promedio'}
    )
    st.plotly_chart(fig, use_container_width=True)