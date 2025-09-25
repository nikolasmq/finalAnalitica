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

st.title("📊 Dashboard de Resultados Saber Pro en Antioquia años 2017 al 2022")
st.markdown("""
Este dashboard interactivo permite explorar y visualizar los resultados de las pruebas Saber Pro en Antioquia desde 2017 hasta 2022.
Utiliza los menús desplegables y gráficos para analizar diferentes aspectos de los datos.
#### Integrantes del equipo:
    * Licet Colorado
    * Santiago Arias
    * Edwin Sánchez
    * Jayson Mejía
    * Nicolás Quintero
    * Yeny Mejía   

#### Fuente de los datos
[https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe/about_data]

### licencia ![image.png](data:image/png;base64,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)     
Esta información se filtro directamente desde la fuente de datos para seleccionar únicamente los registros correspondientes a Antioquia.
            
            """            
)

# 2. Carga y preprocesamiento de datos
# Asegúrate de que esta ruta sea correcta para tu archivo de datos
st.write("### Para iniciar debemos carga de Datos en formato CSV")
uploaded_file = st.file_uploader("Sube tu archivo CSV de resultados aquí", type=['csv'])

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
if uploaded_file:
    df_sample = load_data(uploaded_file)
else:
    st.info("Por favor, sube un archivo CSV para empezar.")
    st.stop()



# Línea divisoria para separar visualmente secciones del dashboard
st.markdown("---")
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
    width=800,
    height=400,
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

# Línea divisoria para separar visualmente secciones del dashboard
st.markdown("---")

# Análisis de Puntaje Global por Naturaleza del Colegio
st.subheader("Puntaje Global Promedio por Naturaleza del Colegio")
st.write("Analiza la diferencia de rendimiento entre instituciones públicas y privadas.")

# 1. Crea dos columnas: una para el gráfico (8 unidades de ancho) y otra para el texto (4 unidades)
col_grafico, col_explicacion = st.columns([8, 4])

# 2. Generación de datos (Necesitas esta lógica para el gráfico)
average_scores_by_nature = df_sample.groupby('COLE_NATURALEZA')[
    ['PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA', 'PUNT_GLOBAL']
].mean().reset_index()

# 3. Creación del gráfico con Plotly
fig_bar = px.bar(
    average_scores_by_nature,
    x='COLE_NATURALEZA',
    y='PUNT_GLOBAL',
    color='COLE_NATURALEZA',
    title='Puntaje Promedio Global por Naturaleza del Colegio',
    labels={'COLE_NATURALEZA': 'Naturaleza del Colegio', 'PUNT_GLOBAL': 'Puntaje Promedio Global'},
    text='PUNT_GLOBAL'
)

# Ajustes de formato del gráfico
fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_bar.update_layout(
    height=380,  # Ajustado para el diseño de la columna
    yaxis=dict(range=[0, 400]) # Manteniendo el rango del eje Y de tu lógica original
)

# 4. Muestra el gráfico en la primera columna
with col_grafico:
    st.plotly_chart(fig_bar, use_container_width=True)

# 5. Muestra la explicación en la segunda columna
with col_explicacion:
    st.markdown("#### Interpretación Clave 💡")
    st.info(
        "**Los colegios No Oficiales** muestran consistentemente un **puntaje global promedio superior** a los colegios oficiales. "
        "Esta brecha es un indicador clave de las disparidades en el rendimiento académico, que pueden estar influenciadas por "
        "factores socioeconómicos y la disponibilidad de recursos educativos."
    )
    st.markdown(
        """
        - **No Oficial:** Muestra el promedio más alto, lo que sugiere una correlación con mayores recursos.
        - **Oficial:** Indica un promedio menor, destacando la necesidad de fortalecer la inversión en educación pública.
        """
    )


# Línea divisoria para separar visualmente secciones del dashboard
st.markdown("---")
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
grafica_estudiantes, descripcion_estudiantes = st.columns([8, 4])

conteo_df = df_sample.groupby(['year', 'COLE_NATURALEZA']).size().reset_index(name='numero_de_estudiantes')
fig_conteo = px.bar(
    conteo_df,
    x='year',
    y='numero_de_estudiantes',
    color='COLE_NATURALEZA',
    barmode='group',
    title='Conteo de Estudiantes por Año y Naturaleza del Colegio',
    text='numero_de_estudiantes'  # Mostrar valores en las columnas
)
fig_conteo.update_traces(texttemplate='%{text}', textposition='inside')

with grafica_estudiantes:
    st.header("Análisis de Puntajes por Naturaleza y Recursos del Colegio")
    # Gráfico de conteo de estudiantes por naturaleza del colegio
    st.subheader("Número de Estudiantes por Año y Naturaleza del Colegio")
    st.plotly_chart(fig_conteo, use_container_width=True)

with descripcion_estudiantes:
    st.markdown("#### Datos relevantes 💡")
    st.info(
        "La información faltante en los años 2020 y 2021 se debe a la suspensión de las pruebas Saber Pro durante la pandemia de COVID-19. "
        "Los datos muestran importante diferencia entre el número de estudiantes de colegios oficiales y no oficiales."
    )

st.markdown("---")

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
    title='Puntaje Promedio por Naturaleza del Colegio',
    text='Puntaje Promedio'
)
fig_avg_nature.update_traces(texttemplate='%{text:.2f}', textposition='inside')

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
    labels={'FAMI_EDUCACIONMADRE': 'Nivel de Educación de la Madre', 'PUNT_GLOBAL': 'Puntaje Global Promedio'},
    text='PUNT_GLOBAL',
    color='PUNT_GLOBAL'
)
fig_mom_edu.update_xaxes(tickangle=45)
fig_mom_edu.update_traces(texttemplate='%{text:.2f}', textposition='inside')
st.plotly_chart(fig_mom_edu, use_container_width=True)

# Gráfico de Puntaje Global vs. Educación del Padre
avg_score_by_dad_edu = df_sample.groupby('FAMI_EDUCACIONPADRE')['PUNT_GLOBAL'].mean().reindex(education_order).dropna().reset_index()
fig_dad_edu = px.bar(
    avg_score_by_dad_edu,
    x='FAMI_EDUCACIONPADRE',
    y='PUNT_GLOBAL',
    title='Puntaje Global Promedio vs. Educación del Padre',
    labels={'FAMI_EDUCACIONPADRE': 'Nivel de Educación del Padre', 'PUNT_GLOBAL': 'Puntaje Global Promedio'},
    text='PUNT_GLOBAL',
    color='PUNT_GLOBAL'
)
fig_dad_edu.update_xaxes(tickangle=45)
fig_dad_edu.update_traces(texttemplate='%{text:.2f}', textposition='inside')
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
        labels={col: column_names_spanish[col], 'PUNT_GLOBAL': 'Puntaje Global Promedio'},
        text='PUNT_GLOBAL',
        color='PUNT_GLOBAL'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
    st.plotly_chart(fig, use_container_width=True)
