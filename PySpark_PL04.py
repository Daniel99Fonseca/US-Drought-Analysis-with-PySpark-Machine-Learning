# Databricks notebook source
import os
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import plotly.express as px
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

os.makedirs("outputs", exist_ok=True)

# COMMAND ----------
spark = (
    SparkSession.builder
    .appName("Fase02_PL04")
    .getOrCreate()
)

# Carregar a bd em formato long (drought.csv)
drought = spark.read.csv("/BigData/Projecto/data/drought.csv", 
                            header=True, 
                            inferSchema=True)

area_pct_df = spark.read.csv("/BigData/Projecto/data/drought_area_pct.csv", 
                             header=True, inferSchema=True)

area_total_df = spark.read.csv("/BigData/Projecto/data/drought_area_total.csv", 
                               header=True, inferSchema=True)

pop_pct_df = spark.read.csv("/BigData/Projecto/data/drought_pop_pct.csv", 
                            header=True, inferSchema=True)

pop_total_df = spark.read.csv("/BigData/Projecto/data/drought_pop_total.csv",
                              header=True, inferSchema=True)

# COMMAND ----------

# Como cada leitura aparece 6x para cada combinacao de data com state:
unicas_drought = drought.select("map_date", "state_abb").distinct().count()
print(f"drought: {unicas_drought} linhas \n")

# dicionario com os outros dfs
dfs_test = {
    "area_pct": area_pct_df,
    "area_total": area_total_df,
    "pop_pct": pop_pct_df,
    "pop_total": pop_total_df
}

# for loop para testar se numero de linhas é igual
for nome, df in dfs_test.items():
    if unicas_drought == df.count():
        print(f"{nome}: {df.count()} linhas")
    else:
        print(f"ERRO! {nome}: Tem {df.count()} linhas!!")

# COMMAND ----------

drought.show(5)

# COMMAND ----------

# drop em colunas desnecessárias
# stat_fmt e constante
# valid start e valid end sao datas que nao sao uteis
drought = drought.drop("stat_fmt", "valid_start", "valid_end")

# COMMAND ----------

# map_date está como integer, tem de passar para date
drought.printSchema()

# COMMAND ----------

# converter map_date para datetype
drought = drought.withColumn("map_date", F.to_date(F.col("map_date").cast("string"), "yyyyMMdd"))

# confirmar resultados em yyyMMdd
drought.select("map_date").show(5)

# COMMAND ----------

# confirmação que map_date tem tipo date
drought.printSchema()

# COMMAND ----------

# Verificacao de valores nulos
for column in drought.columns:
    print(column, drought.filter(F.col(column).isNull()).count())

# COMMAND ----------

# col numericas
cols_numericas = ["area_pct", "area_total", "pop_pct", "pop_total"]

drought.select(cols_numericas).describe().show()

# pop_pct tem valor max em 101.8%. Isso é impossível, dado que é uma percentagem

# COMMAND ----------

# Cap a 100% em pop_pct
drought = drought.withColumn("pop_pct",
    F.when(F.col("pop_pct") > 100, 100).otherwise(F.col("pop_pct"))
)

# COMMAND ----------

# Verificação de duplicados
total_antes = drought.count()
drought = drought.distinct()
total_depois = drought.count()

print(f"Linhas removidas: {total_antes - total_depois}")

# COMMAND ----------

# Verificar valores únicos em drought_lvl e state_abb
print("Drought levels únicos:")
drought.select("drought_lvl").distinct().show()

print(f"Número de estados únicos: {drought.select('state_abb').distinct().count()}")
drought.select("state_abb").distinct().orderBy("state_abb").show(60)

# COMMAND ----------

# Intervalo temporal dos dados
drought.select(
    F.min("map_date").alias("data_inicio"),
    F.max("map_date").alias("data_fim"),
    F.countDistinct("map_date").alias("semanas_unicas")
).show()

# COMMAND ----------

# DBTITLE 1,EDA
# ----------------------------------------------- 
# --------------------- EDA ---------------------
# ----------------------------------------------- 

# Filtrar fora o "None" (sem seca) para focar nos níveis de seca real
drought_real = drought.filter(F.col("drought_lvl") != "None")

# Média nacional de area_pct em seca, por semana
nacional_tempo = (
    drought_real
    .groupBy("map_date")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .orderBy("map_date")
)

nacional_tempo.show(10)

# COMMAND ----------

# Ordem dos níveis de seca
ordem_lvl = ["D0", "D1", "D2", "D3", "D4"]

# Média nacional de area_pct por semana E por nível de seca
por_nivel_tempo = (
    drought.filter(F.col("drought_lvl").isin(ordem_lvl))
    .groupBy("map_date", "drought_lvl")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .orderBy("map_date", "drought_lvl")
)

por_nivel_tempo_pd = por_nivel_tempo.toPandas()

# Pivotar para o plot
pivot_pd = por_nivel_tempo_pd.pivot(index="map_date", columns="drought_lvl", values="media_area_pct")[ordem_lvl]

# Stacked area chart
fig, ax = plt.subplots(figsize=(14, 5))

cores = ["#fdd835", "#fb8c00", "#e53935", "#880e4f", "#4a148c"]  # amarelo a roxo escuro

pivot_pd.plot.area(ax=ax, color=cores, alpha=0.85)

ax.set_title("Distribuição da Severidade da Seca ao Longo do Tempo (Média Nacional)")
ax.set_xlabel("Data")
ax.set_ylabel("% Área em Seca (média entre estados)")
ax.legend(title="Nível de Seca", loc="upper right")
plt.tight_layout()
plt.savefig("outputs/01_stacked_area.png", dpi=150)
plt.close()

# COMMAND ----------

# Ranking de estados por média de área em seca (excluindo None)
por_estado = (
    drought.filter(F.col("drought_lvl") != "None")
    .groupBy("state_abb")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .orderBy(F.desc("media_area_pct"))
)

por_estado_pd = por_estado.toPandas()

# Bar chart horizontal
fig, ax = plt.subplots(figsize=(10, 14))

ax.barh(
    por_estado_pd["state_abb"][::-1],  # inverter para maior em cima
    por_estado_pd["media_area_pct"][::-1],
    color="steelblue"
)

ax.set_title("Média de Área em Seca por Estado (2001–2021)")
ax.set_xlabel("% Área em Seca (média)")
ax.set_ylabel("Estado")
plt.tight_layout()
plt.savefig("outputs/02_ranking_estados.png", dpi=150)
plt.close()

# COMMAND ----------

print("Top 5 estados com mais seca:")
por_estado.show(5)

print("Top 5 estados com menos seca:")
por_estado.orderBy(F.asc("media_area_pct")).show(5)

# COMMAND ----------

# Correlação entre area_pct e pop_pct, por nível de seca

# Calcular correlação de Pearson por drought_lvl
print("Correlação de Pearson entre area_pct e pop_pct por nível de seca:\n")

for nivel in ["D0", "D1", "D2", "D3", "D4"]:
    df_nivel = drought.filter(F.col("drought_lvl") == nivel)
    
    assembler = VectorAssembler(inputCols=["area_pct", "pop_pct"], outputCol="features")
    df_vec = assembler.transform(df_nivel).select("features")
    
    corr_matrix = Correlation.corr(df_vec, "features").head()
    corr_value = corr_matrix[0][0, 1]  # posição [0,1] da matriz 2x2
    print(f"  {nivel}: r = {corr_value:.4f}")

# COMMAND ----------

# Média de area_pct vs pop_pct por nível de seca
comparacao = (
    drought.filter(F.col("drought_lvl") != "None")
    .groupBy("drought_lvl")
    .agg(
        F.avg("area_pct").alias("media_area_pct"),
        F.avg("pop_pct").alias("media_pop_pct")
    )
    .orderBy("drought_lvl")
)

comp_pd = comparacao.toPandas()

x = np.arange(len(comp_pd))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - width/2, comp_pd["media_area_pct"], width, label="Área (%)", color="steelblue")
ax.bar(x + width/2, comp_pd["media_pop_pct"], width, label="População (%)", color="coral")

ax.set_xticks(x)
ax.set_xticklabels(comp_pd["drought_lvl"])
ax.set_title("Área vs População em Seca por Nível (Média Nacional)")
ax.set_xlabel("Nível de Seca")
ax.set_ylabel("% Média")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/03_area_vs_pop.png", dpi=150)
plt.close()

# COMMAND ----------

# Média de area_pct por estado e nível de seca (excluindo None)
perfil_estados = (
    drought.filter(F.col("drought_lvl") != "None")
    .groupBy("state_abb", "drought_lvl")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .orderBy("state_abb", "drought_lvl")
)

perfil_pd = perfil_estados.toPandas()

# Pivotar para formato de heatmap
heatmap_pd = perfil_pd.pivot(index="state_abb", columns="drought_lvl", values="media_area_pct")
heatmap_pd = heatmap_pd[["D0", "D1", "D2", "D3", "D4"]]

# Ordenar estados pelo total de seca (soma das colunas) para melhor leitura
heatmap_pd = heatmap_pd.loc[heatmap_pd.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(9, 16))
sns.heatmap(
    heatmap_pd,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0.3,
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "% Área em Seca (média)"}
)
ax.set_title("Perfil de Seca por Estado e Nível (2001–2021)")
ax.set_xlabel("Nível de Seca")
ax.set_ylabel("Estado")
plt.tight_layout()
plt.savefig("outputs/04_heatmap_estados.png", dpi=150)
plt.close()

# COMMAND ----------

# Dividir em duas décadas
drought_d1 = drought.filter(
    (F.col("drought_lvl") != "None") & (F.year("map_date") <= 2011)
)
drought_d2 = drought.filter(
    (F.col("drought_lvl") != "None") & (F.year("map_date") > 2011)
)

# Média por estado em cada período
media_d1 = drought_d1.groupBy("state_abb").agg(F.avg("area_pct").alias("media_2001_2011"))
media_d2 = drought_d2.groupBy("state_abb").agg(F.avg("area_pct").alias("media_2012_2021"))

# Join e calcular diferença
tendencia = media_d1.join(media_d2, on="state_abb") \
    .withColumn("variacao", F.col("media_2012_2021") - F.col("media_2001_2011")) \
    .orderBy(F.desc("variacao"))

tendencia.show(10)

# COMMAND ----------

tend_pd = tendencia.toPandas().sort_values("variacao", ascending=True)

cores = ["#e53935" if v > 0 else "steelblue" for v in tend_pd["variacao"]]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(tend_pd["state_abb"], tend_pd["variacao"], color=cores)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Variação de Seca por Estado\n(2012–2021 vs 2001–2011)")
ax.set_xlabel("Variação em % Área em Seca")
ax.set_ylabel("Estado")
plt.tight_layout()
plt.savefig("outputs/05_tendencia_estados.png", dpi=150)
plt.close()

# COMMAND ----------

# O Spark exige que as colunas estejam num vector para fazer a correlação
assembler = VectorAssembler(inputCols=cols_numericas, outputCol="features_corr")
df_vector = assembler.transform(drought).select("features_corr")

# matriz de Pearson
matriz = Correlation.corr(df_vector, "features_corr").collect()[0][0]
correlacao_np = matriz.toArray()

# meter em df
df_corr = pd.DataFrame(correlacao_np, columns=cols_numericas, index=cols_numericas)

# heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação entre Variáveis de Seca")
plt.tight_layout()
plt.savefig("outputs/06_correlacao.png", dpi=150)
plt.close()

# COMMAND ----------

# Tirar uma amostra estratificada
# Vamos focar-nos nos níveis de seca D0-D4
pdf_boxplot_fix = drought.filter(F.col("drought_lvl") != "None") \
                         .select("drought_lvl", "area_pct") \
                         .sample(False, 0.1) \
                         .toPandas()

# Criar o gráfico
plt.figure(figsize=(12, 6))

sns.boxplot(x='drought_lvl', y='area_pct', data=pdf_boxplot_fix, 
            order=['D0', 'D1', 'D2', 'D3', 'D4'])

# zoom no eixo Y
plt.ylim(0, 105)

plt.title("Distribuição da Percentagem de Área por Nível de Seca (D0-D4)", fontsize=14)
plt.xlabel("Nível de Seca", fontsize=12)
plt.ylabel("% de Área Afetada", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_boxplot_niveis.png", dpi=150)
plt.close()

# COMMAND ----------

# Criar índice de severidade ponderado por estado e semana
pesos = {"None": 0, "D0": 1, "D1": 2, "D2": 3, "D3": 4, "D4": 5}

# Adicionar coluna de peso
drought_idx = drought.withColumn("peso", 
    F.when(F.col("drought_lvl") == "None", 0)
     .when(F.col("drought_lvl") == "D0", 1)
     .when(F.col("drought_lvl") == "D1", 2)
     .when(F.col("drought_lvl") == "D2", 3)
     .when(F.col("drought_lvl") == "D3", 4)
     .when(F.col("drought_lvl") == "D4", 5)
)

# Índice ponderado = sum(area_pct * peso) / 100
drought_idx = (
    drought_idx
    .withColumn("ano", F.year("map_date"))
    .withColumn("mes", F.month("map_date"))
    .groupBy("state_abb", "ano", "mes")
    .agg(
        (F.sum(F.col("area_pct") * F.col("peso")) / 100).alias("drought_index")
    )
    .withColumn("ano_mes", 
        F.concat(F.col("ano"), F.lit("-"), F.lpad(F.col("mes"), 2, "0"))
    )
    .orderBy("ano", "mes")
)

drought_idx.show(5)

# COMMAND ----------

mapa_tempo_pd = drought_idx.toPandas()

# Ordenar frames cronologicamente
mapa_tempo_pd = mapa_tempo_pd.sort_values(["ano", "mes"])

fig = px.choropleth(
    mapa_tempo_pd,
    locations="state_abb",
    locationmode="USA-states",
    color="drought_index",
    scope="usa",
    animation_frame="ano_mes",
    color_continuous_scale=[
        [0.0,  "#ffffff"],  
        [0.2,  "#fdd835"],   
        [0.4,  "#fb8c00"],  
        [0.6,  "#e53935"],   
        [0.8,  "#880e4f"],   
        [1.0,  "#4a148c"]   
    ],
    range_color=[0, 5],
    title="Evolução da Severidade de Seca nos EUA (2001–2021)",
    labels={"drought_index": "Índice de Severidade", "ano_mes": "Mês"}
)

fig.update_layout(
    coloraxis_colorbar=dict(
        title="Índice de<br>Severidade",
        tickvals=[0, 1, 2, 3, 4, 5],
        ticktext=["Sem Seca", "D0", "D1", "D2", "D3", "D4"]
    )
)

fig.write_html("outputs/08_mapa_animado.html")

# COMMAND ----------

# DBTITLE 1,CLUSTERING
# -------------------------------------------------
# ------------ CLUSTERING -------------------------
# -------------------------------------------------

# Média de area_pct por estado e nível de seca (excluindo None)
feature_matrix = (
    drought.filter(F.col("drought_lvl") != "None")
    .groupBy("state_abb")
    .pivot("drought_lvl", ["D0", "D1", "D2", "D3", "D4"])
    .agg(F.avg("area_pct"))
)

feature_matrix.orderBy("state_abb").show(5)
print(f"Dimensões: {feature_matrix.count()} linhas x {len(feature_matrix.columns)} colunas")

# COMMAND ----------

# Juntar as 5 features num único vetor
assembler = VectorAssembler(
    inputCols=["D0", "D1", "D2", "D3", "D4"],
    outputCol="features_raw"
)

# Scaling
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

pipeline_prep = Pipeline(stages=[assembler, scaler])
model_prep = pipeline_prep.fit(feature_matrix)
df_scaled = model_prep.transform(feature_matrix)

df_scaled.select("state_abb", "features").show(5, truncate=False)

# COMMAND ----------

# Testar K de 2 a 10
wcss = []
k_values = list(range(2, 11))

for k in k_values:
    kmeans = KMeans(featuresCol="features", k=k, seed=42)
    model = kmeans.fit(df_scaled)
    wcss.append(model.summary.trainingCost)
    print(f"K={k} -> WCSS={model.summary.trainingCost:.4f}")

# Elbow plot
plt.figure(figsize=(9, 4))
plt.plot(k_values, wcss, marker="o", color="steelblue", linewidth=2)
plt.title("Elbow Method — Escolha do K Óptimo")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("WCSS (Within Cluster Sum of Squares)")
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/09_elbow_com_d0.png", dpi=150)
plt.close()

# COMMAND ----------

# Treinar K-Means com K=4
kmeans_final = KMeans(featuresCol="features", k=4, seed=42)
model_final = kmeans_final.fit(df_scaled)

# Adicionar a coluna de cluster ao dataframe
df_clustered = model_final.transform(df_scaled)

# Ver resultado
df_clustered.select("state_abb", "D0", "D1", "D2", "D3", "D4", "prediction") \
    .orderBy("prediction", "state_abb") \
    .show(52, truncate=False)

# COMMAND ----------

# centróides de cada cluster
print("Centróides dos clusters:")
df_clustered.groupBy("prediction") \
    .agg(
        F.avg("D0").alias("media_D0"),
        F.avg("D1").alias("media_D1"),
        F.avg("D2").alias("media_D2"),
        F.avg("D3").alias("media_D3"),
        F.avg("D4").alias("media_D4"),
        F.count("state_abb").alias("num_estados")
    ) \
    .orderBy("prediction") \
    .show()

# COMMAND ----------

# Converter para pandas
mapa_pd = df_clustered.select("state_abb", "prediction").toPandas()

# Nome descritivo dos clusters
mapa_pd["cluster_nome"] = mapa_pd["prediction"].map({
    0: "Cluster 0 - Seca Moderada",
    1: "Cluster 1 - Seca Severa",
    2: "Cluster 2 - Baixa Seca",
    3: "Cluster 3 - Seca Moderada-Alta"
})

fig = px.choropleth(
    mapa_pd,
    locations="state_abb",
    locationmode="USA-states",
    color="cluster_nome",
    scope="usa",
    title="Clustering de Estados por Perfil de Seca (K=4)",
    color_discrete_map={
        "Cluster 0 - Seca Moderada Mista": "#fb8c00",
        "Cluster 1 - Seca Severa": "#e53935",
        "Cluster 2 - Baixa Seca": "steelblue",
        "Cluster 3 - Seca Moderada-Alta": "#8e24aa"
    },
    labels={"cluster_nome": "Cluster"}
)

fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2))
fig.write_html("outputs/10_mapa_k4.html")

# COMMAND ----------

centros_pd = df_clustered.groupBy("prediction") \
    .agg(
        F.avg("D0").alias("D0"),
        F.avg("D1").alias("D1"),
        F.avg("D2").alias("D2"),
        F.avg("D3").alias("D3"),
        F.avg("D4").alias("D4")
    ).orderBy("prediction").toPandas()

# nomes dos clusters baseados na interpretação
nomes_clusters = {
    0: "Cluster 0 - Seca Moderada",
    1: "Cluster 1 - Seca Severa",
    2: "Cluster 2 - Baixa Seca",
    3: "Cluster 3 - Seca Moderada-Alta"
}
cores_clusters = ["#fb8c00", "#e53935", "steelblue", "#8e24aa"]

features = ["D0", "D1", "D2", "D3", "D4"]
n = len(features)
angles = [i * 2 * np.pi / n for i in range(n)] + [0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()

for i, row in centros_pd.iterrows():
    valores = [row[f] for f in features] + [row["D0"]]
    axes[i].plot(angles, valores, color=cores_clusters[i], linewidth=2)
    axes[i].fill(angles, valores, color=cores_clusters[i], alpha=0.25)
    axes[i].set_xticks(angles[:-1])
    axes[i].set_xticklabels(features, fontsize=12)
    axes[i].set_title(nomes_clusters[i], size=11, pad=15)

plt.suptitle("Perfil de Seca por Cluster (Radar Chart)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/11_radar_k4.png", dpi=150, bbox_inches="tight")
plt.close()

# COMMAND ----------

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette"
)

silhouette = evaluator.evaluate(df_clustered)
print(f"Silhouette Score (K=4): {silhouette:.4f}")

# COMMAND ----------

print("Silhouette Score por K:\n")

for k in [2, 3, 4, 5, 6]:
    km = KMeans(featuresCol="features", k=k, seed=42)
    m = km.fit(df_scaled)
    df_temp = m.transform(df_scaled)
    score = evaluator.evaluate(df_temp)
    print(f"  K={k} -> Silhouette = {score:.4f}")

# COMMAND ----------

# Índice de severidade médio por estado e ano
drought_ano = drought.withColumn("peso",
    F.when(F.col("drought_lvl") == "None", 0)
     .when(F.col("drought_lvl") == "D0", 1)
     .when(F.col("drought_lvl") == "D1", 2)
     .when(F.col("drought_lvl") == "D2", 3)
     .when(F.col("drought_lvl") == "D3", 4)
     .when(F.col("drought_lvl") == "D4", 5)
).withColumn("ano", F.year("map_date"))

drought_ano = (
    drought_ano
    .groupBy("state_abb", "ano")
    .agg((F.sum(F.col("area_pct") * F.col("peso")) / 100).alias("drought_index"))
)

# Pivotar para heatmap
heatmap_ano_pd = drought_ano.toPandas().pivot(
    index="state_abb", columns="ano", values="drought_index"
)

# Ordenar estados pelo índice médio total
heatmap_ano_pd = heatmap_ano_pd.loc[
    heatmap_ano_pd.mean(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(
    heatmap_ano_pd,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0.3,
    cbar_kws={"label": "Índice de Severidade de Seca"}
)
ax.set_title("Índice de Severidade de Seca por Estado e Ano (2001–2021)", fontsize=14)
ax.set_xlabel("Ano")
ax.set_ylabel("Estado")
plt.tight_layout()
plt.savefig("outputs/12_heatmap_anual.png", dpi=150)
plt.close()

# COMMAND ----------

# Testar estabilidade com diferentes seeds
print("Estabilidade dos clusters com diferentes seeds:")
for seed in [42, 123, 456, 789, 1000]:
    km = KMeans(featuresCol="features", k=4, seed=seed)
    m = km.fit(df_scaled)
    score = evaluator.evaluate(m.transform(df_scaled))
    print(f"  seed={seed} -> Silhouette = {score:.4f}")

# COMMAND ----------

# DBTITLE 1,SEM D0
##################### SEM D0 ##########################################
#######################################################################
# Clustering apenas com D1-D4 (excluindo D0)
feature_matrix_sem_d0 = (
    drought_real
    .groupBy("state_abb")
    .pivot("drought_lvl", ["D1", "D2", "D3", "D4"])
    .agg(F.avg("area_pct"))
)

assembler_sd0 = VectorAssembler(
    inputCols=["D1", "D2", "D3", "D4"],
    outputCol="features_raw"
)
scaler_sd0 = StandardScaler(
    inputCol="features_raw", outputCol="features",
    withMean=True, withStd=True
)
pipeline_sd0 = Pipeline(stages=[assembler_sd0, scaler_sd0])
df_scaled_sd0 = pipeline_sd0.fit(feature_matrix_sem_d0).transform(feature_matrix_sem_d0)

# Repetir elbow e silhouette para K=2 a K=6
for k in [2, 3, 4, 5, 6]:
    km = KMeans(featuresCol="features", k=k, seed=42)
    m = km.fit(df_scaled_sd0)
    score = evaluator.evaluate(m.transform(df_scaled_sd0))
    print(f"K={k} -> Silhouette = {score:.4f}")

# COMMAND ----------

wcss_sd0 = []
k_values = list(range(2, 11))

print("Elbow sem D0:")
for k in k_values:
    km = KMeans(featuresCol="features", k=k, seed=42)
    model = km.fit(df_scaled_sd0)
    cost = model.summary.trainingCost
    wcss_sd0.append(cost)
    print(f"  K={k} -> WCSS={cost:.4f}")

plt.figure(figsize=(9, 4))
plt.plot(k_values, wcss_sd0, marker="o", color="steelblue", linewidth=2)
plt.title("Elbow Method sem D0")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/13_elbow_sem_d0.png", dpi=150)
plt.close()

# COMMAND ----------

# Modelo final K=3 sem D0
kmeans_final_sd0 = KMeans(featuresCol="features", k=3, seed=42)
model_final_sd0 = kmeans_final_sd0.fit(df_scaled_sd0)
df_clustered_sd0 = model_final_sd0.transform(df_scaled_sd0)

# Centróides em escala original
centros_sd0 = (
    df_clustered_sd0
    .groupBy("prediction")
    .agg(
        F.avg("D1").alias("D1"),
        F.avg("D2").alias("D2"),
        F.avg("D3").alias("D3"),
        F.avg("D4").alias("D4"),
        F.count("state_abb").alias("num_estados")
    )
    .orderBy("prediction")
)
centros_sd0.show()

# COMMAND ----------

# Radar chart K=3 sem D0
centros_pd_sd0 = (
    df_clustered_sd0.groupBy("prediction")
    .agg(
        F.avg("D1").alias("D1"),
        F.avg("D2").alias("D2"),
        F.avg("D3").alias("D3"),
        F.avg("D4").alias("D4")
    )
    .orderBy("prediction")
    .toPandas()
)

features_sd0 = ["D1", "D2", "D3", "D4"]
n = len(features_sd0)
angles = [i * 2 * np.pi / n for i in range(n)] + [0]
max_val = centros_pd_sd0[features_sd0].values.max()

nomes_clusters_sd0 = {
    0: "Cluster 0 - Seca Moderada",
    1: "Cluster 1 - Baixa Seca",
    2: "Cluster 2 - Seca Severa"
}
cores_sd0 = ["#fb8c00", "steelblue", "#e53935"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
for i, row in centros_pd_sd0.iterrows():
    valores = [row[f] for f in features_sd0] + [row["D1"]]
    axes[i].plot(angles, valores, color=cores_sd0[i], linewidth=2)
    axes[i].fill(angles, valores, color=cores_sd0[i], alpha=0.25)
    axes[i].set_xticks(angles[:-1])
    axes[i].set_xticklabels(features_sd0, fontsize=12)
    axes[i].set_title(nomes_clusters_sd0[i], size=10, pad=15)
    
    # Forçar escala partilhada em todos os subplots
    axes[i].set_ylim(0, max_val * 1.1)

plt.suptitle("Perfil de Seca por Cluster K=3 sem D0", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/14_radar_k3_sem_d0.png", dpi=150, bbox_inches="tight")
plt.close()

# COMMAND ----------

mapa_pd_sd0 = df_clustered_sd0.select("state_abb", "prediction").toPandas()
mapa_pd_sd0["cluster_nome"] = mapa_pd_sd0["prediction"].map(nomes_clusters_sd0)

fig = px.choropleth(
    mapa_pd_sd0, locations="state_abb", locationmode="USA-states",
    color="cluster_nome", scope="usa",
    title="Clustering de Estados por Perfil de Seca: K=3 sem D0",
    color_discrete_map={
        nomes_clusters_sd0[0]: "#fb8c00",
        nomes_clusters_sd0[1]: "steelblue",
        nomes_clusters_sd0[2]: "#e53935"
    }
)
fig.write_html("outputs/15_mapa_k3_sem_d0.html")

resultados = []

for seed in [42, 123, 456, 789, 1000]:
    km = KMeans(featuresCol="features", k=3, seed=seed)
    m = km.fit(df_scaled_sd0)
    df_temp = m.transform(df_scaled_sd0)
    score = evaluator.evaluate(df_temp)
    
    # 1. Confirmar seed usada
    print(f"\n--- seed={seed} ---")
    
    # 2. Centróides finais
    print("Centróides:")
    for i, c in enumerate(m.clusterCenters()):
        print(f"  Cluster {i}: {[round(x, 4) for x in c]}")
    
    # 3. Silhouette score
    print(f"Silhouette: {score:.4f}")
    
    # 4. Atribuição de labels por estado
    atribuicao = (
        df_temp.select("state_abb", "prediction")
        .orderBy("state_abb")
        .toPandas()
        .set_index("state_abb")["prediction"]
        .to_dict()
    )
    resultados.append({"seed": seed, "score": score, "atribuicao": atribuicao})

spark.stop()