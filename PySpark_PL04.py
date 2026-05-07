"""
================================================================================
 Projeto de Aprendizagem Não-Supervisionada — US Drought Monitor
================================================================================

 Grupo:        PL04
 Autores:      Daniel Fonseca (125158), Daniil Samsonyuk (130646),
               Guilherme Pires (131658), João Filipe (130665)
 Data:         Maio 2026
 Disciplina:   Processamento de Big Data — ISCTE-IUL — 25/26
 Docentes:     Anabela Costa, Mafalda de Ponte, Maria João Cortinhal

 Descrição:
   Este script realiza a análise completa do dataset US Drought Monitor
   (2001–2021), incluindo limpeza de dados, análise exploratória (EDA) e
   clustering de estados americanos por perfil de seca utilizando K-Means.

   Decisão de feature engineering: o nível D0 foi excluído das features
   de clustering por apresentar baixo poder discriminativo (variação de
   apenas 14–21% entre estados vs. rácio de 14× em D3). A remoção
   melhorou o Silhouette Score em quase todos os valores de K testados.

   Solução final: K=3 sem D0 (Silhouette = 0.7361).

 Dataset esperado:
   /BigData/Projecto/data/drought.csv

 Bibliotecas necessárias:
   pyspark, pandas, numpy, matplotlib, seaborn, plotly

 Como executar:
   spark-submit PySpark_PL04.py
   ou executar em ambiente Databricks

 Notas:
   - Seed fixa (SEED = 42) para reprodutibilidade
   - Outputs guardados em pasta outputs/
================================================================================
"""

# ============================================================================
# 0. IMPORTAÇÕES
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline


# ============================================================================
# 1. INICIALIZAÇÃO DA SPARK SESSION
# ============================================================================
spark = (
    SparkSession.builder
    .appName("US_Drought_Clustering_PL04")
    .getOrCreate()
)

DATA_PATH = "/BigData/Projecto/data/drought.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEED = 42  # seed fixa para reprodutibilidade


# ============================================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================================
# Formato longo: 6 linhas por (map_date, state_abb) — uma por nível de seca
drought = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

print("=== Dataset carregado ===")
print(f"Linhas: {drought.count()} | Colunas: {len(drought.columns)}")
drought.printSchema()


# ============================================================================
# 3. LIMPEZA E PREPARAÇÃO DOS DADOS
# ============================================================================

# 3.1 Remover colunas sem valor analítico
# stat_fmt = constante (sempre 2); valid_start/end = deriváveis de map_date
drought = drought.drop("stat_fmt", "valid_start", "valid_end")

# 3.2 Converter map_date de integer (yyyyMMdd) para DateType
drought = drought.withColumn(
    "map_date",
    F.to_date(F.col("map_date").cast("string"), "yyyyMMdd")
)

# 3.3 Verificação de valores nulos
print("\n=== Valores nulos por coluna ===")
for column in drought.columns:
    print(f"  {column}: {drought.filter(F.col(column).isNull()).count()}")

# 3.4 Estatísticas descritivas
print("\n=== Estatísticas descritivas ===")
drought.select(["area_pct", "area_total", "pop_pct", "pop_total"]).describe().show()

# 3.5 Cap a 100% no pop_pct
# Causa: erros de arredondamento floating point na soma dos 6 níveis.
# Maioritariamente no nível None e em estados IA, GA, VA, SD, WA.
drought = drought.withColumn(
    "pop_pct",
    F.when(F.col("pop_pct") > 100, 100).otherwise(F.col("pop_pct"))
)

# 3.6 Remoção de duplicados
total_antes = drought.count()
drought = drought.distinct()
print(f"\nLinhas duplicadas removidas: {total_antes - drought.count()}")

# 3.7 Verificação de completude
# Cada (map_date, state_abb) deve ter exactamente 6 linhas
incompletos = (
    drought.groupBy("map_date", "state_abb").count()
    .filter(F.col("count") != 6).count()
)
print(f"Combinações incompletas: {incompletos}")  # esperado: 0


# ============================================================================
# 4. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ============================================================================

drought_real = drought.filter(F.col("drought_lvl") != "None")
ordem_lvl = ["D0", "D1", "D2", "D3", "D4"]
cores_lvl = ["#fdd835", "#fb8c00", "#e53935", "#880e4f", "#4a148c"]


# ----------------------------------------------------------------------------
# 4.1 Série temporal nacional
# ----------------------------------------------------------------------------
nacional_tempo = (
    drought_real.groupBy("map_date")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .orderBy("map_date").toPandas()
)

plt.figure(figsize=(14, 4))
plt.plot(nacional_tempo["map_date"], nacional_tempo["media_area_pct"], linewidth=0.8)
plt.title("Média Nacional de Área em Seca ao Longo do Tempo")
plt.xlabel("Data")
plt.ylabel("% Área em Seca (média entre estados)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_serie_nacional.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 4.2 Stacked area chart — severidade ao longo do tempo
# ----------------------------------------------------------------------------
por_nivel_tempo = (
    drought.filter(F.col("drought_lvl").isin(ordem_lvl))
    .groupBy("map_date", "drought_lvl")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .toPandas()
)
pivot_tempo = por_nivel_tempo.pivot(
    index="map_date", columns="drought_lvl", values="media_area_pct"
)[ordem_lvl]

fig, ax = plt.subplots(figsize=(14, 5))
pivot_tempo.plot.area(ax=ax, color=cores_lvl, alpha=0.85)
ax.set_title("Distribuição da Severidade da Seca ao Longo do Tempo (Média Nacional)")
ax.set_xlabel("Data")
ax.set_ylabel("% Área em Seca (média entre estados)")
ax.legend(title="Nível de Seca", loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_severidade_tempo.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 4.3 Ranking de estados
# ----------------------------------------------------------------------------
por_estado = (
    drought_real.groupBy("state_abb")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .orderBy(F.desc("media_area_pct")).toPandas()
)
print("\n=== Top 5 estados com mais seca ===")
print(por_estado.head(5).to_string(index=False))
print("\n=== Top 5 estados com menos seca ===")
print(por_estado.tail(5).to_string(index=False))


# ----------------------------------------------------------------------------
# 4.4 Sazonalidade por mês e nível de seca
# ----------------------------------------------------------------------------
nomes_meses = ["Jan","Fev","Mar","Abr","Mai","Jun",
               "Jul","Ago","Set","Out","Nov","Dez"]

sazon_nivel = (
    drought_real.withColumn("mes", F.month("map_date"))
    .groupBy("mes", "drought_lvl")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .toPandas()
)
pivot_sazon = sazon_nivel.pivot(
    index="mes", columns="drought_lvl", values="media_area_pct"
)[ordem_lvl]
pivot_sazon.index = nomes_meses

fig, ax = plt.subplots(figsize=(11, 4))
pivot_sazon.plot(kind="bar", ax=ax, color=cores_lvl, width=0.75)
ax.set_title("Sazonalidade por Nível de Seca")
ax.set_xlabel("Mês")
ax.set_ylabel("% Área em Seca (média)")
ax.legend(title="Nível de Seca", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_sazonalidade.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 4.5 Correlação Pearson entre area_pct e pop_pct
# ----------------------------------------------------------------------------
print("\n=== Correlação Pearson area_pct vs pop_pct ===")
for nivel in ["None", "D0", "D1", "D2", "D3", "D4"]:
    df_nivel = drought.filter(F.col("drought_lvl") == nivel)
    assembler_corr = VectorAssembler(
        inputCols=["area_pct", "pop_pct"], outputCol="features"
    )
    df_vec = assembler_corr.transform(df_nivel).select("features")
    corr_matrix = Correlation.corr(df_vec, "features").head()
    print(f"  {nivel}: r = {corr_matrix[0][0, 1]:.4f}")


# ----------------------------------------------------------------------------
# 4.6 Heatmap estado x nível de seca
# ----------------------------------------------------------------------------
perfil_estados = (
    drought_real.groupBy("state_abb", "drought_lvl")
    .agg(F.avg("area_pct").alias("media_area_pct"))
    .toPandas()
)
heatmap_pd = perfil_estados.pivot(
    index="state_abb", columns="drought_lvl", values="media_area_pct"
)[ordem_lvl]
heatmap_pd = heatmap_pd.loc[
    heatmap_pd.sum(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(9, 16))
sns.heatmap(heatmap_pd, ax=ax, cmap="YlOrRd", linewidths=0.3,
            annot=True, fmt=".1f",
            cbar_kws={"label": "% Área em Seca (média)"})
ax.set_title("Perfil de Seca por Estado e Nível (2001–2021)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_heatmap_estados.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 4.7 Tendência — variação entre décadas
# ----------------------------------------------------------------------------
media_d1 = (drought_real.filter(F.year("map_date") <= 2011)
    .groupBy("state_abb").agg(F.avg("area_pct").alias("media_2001_2011")))
media_d2 = (drought_real.filter(F.year("map_date") > 2011)
    .groupBy("state_abb").agg(F.avg("area_pct").alias("media_2012_2021")))

tend_pd = (
    media_d1.join(media_d2, on="state_abb")
    .withColumn("variacao", F.col("media_2012_2021") - F.col("media_2001_2011"))
    .toPandas().sort_values("variacao", ascending=True)
)

cores_tend = ["#e53935" if v > 0 else "steelblue" for v in tend_pd["variacao"]]
fig, ax = plt.subplots(figsize=(10, 14))
ax.barh(tend_pd["state_abb"], tend_pd["variacao"], color=cores_tend)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Variação de Seca por Estado (2012–2021 vs 2001–2011)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_tendencia_estados.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 4.8 Heatmap anual — índice de severidade ponderado
# ----------------------------------------------------------------------------
drought_idx = drought.withColumn("peso",
    F.when(F.col("drought_lvl") == "None", 0)
     .when(F.col("drought_lvl") == "D0", 1)
     .when(F.col("drought_lvl") == "D1", 2)
     .when(F.col("drought_lvl") == "D2", 3)
     .when(F.col("drought_lvl") == "D3", 4)
     .when(F.col("drought_lvl") == "D4", 5)
).withColumn("ano", F.year("map_date"))

heatmap_ano = (
    drought_idx.groupBy("state_abb", "ano")
    .agg((F.sum(F.col("area_pct") * F.col("peso")) / 100).alias("drought_index"))
    .toPandas().pivot(index="state_abb", columns="ano", values="drought_index")
)
heatmap_ano = heatmap_ano.loc[
    heatmap_ano.mean(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(heatmap_ano, ax=ax, cmap="YlOrRd", linewidths=0.3,
            cbar_kws={"label": "Índice de Severidade de Seca"})
ax.set_title("Índice de Severidade de Seca por Estado e Ano (2001–2021)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_heatmap_anual.png", dpi=150)
plt.close()


# ============================================================================
# 5. PREPARAÇÃO PARA CLUSTERING
# ============================================================================
# Unidade de análise: estado (52 observações)
# Features: D1, D2, D3, D4 — média histórica de area_pct
#
# Exclusões justificadas:
#   - None: informação inversa (% sem seca), redundante
#   - D0: baixo poder discriminativo — variação 14-21% vs rácio 14x em D3.
#         Validação empírica: Silhouette melhora em quase todos os K sem D0.
#   - area_total/pop_total: size bias
#   - pop_pct: correlação > 0.86 com area_pct (redundante)

FEATURES = ["D1", "D2", "D3", "D4"]

feature_matrix = (
    drought_real
    .groupBy("state_abb")
    .pivot("drought_lvl", FEATURES)
    .agg(F.avg("area_pct"))
)

print(f"\n=== Feature matrix (sem D0) ===")
print(f"Dimensões: {feature_matrix.count()} linhas x {len(feature_matrix.columns)} colunas")
feature_matrix.show(5)


# ============================================================================
# 6. PADRONIZAÇÃO — StandardScaler
# ============================================================================
assembler = VectorAssembler(inputCols=FEATURES, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withMean=True, withStd=True)
df_scaled = Pipeline(stages=[assembler, scaler]).fit(feature_matrix).transform(feature_matrix)


# ============================================================================
# 7. COMPARAÇÃO COM E SEM D0
# ============================================================================
evaluator = ClusteringEvaluator(
    featuresCol="features", predictionCol="prediction", metricName="silhouette"
)

FEATURES_COM_D0 = ["D0", "D1", "D2", "D3", "D4"]
fm_com_d0 = (
    drought_real.groupBy("state_abb")
    .pivot("drought_lvl", FEATURES_COM_D0).agg(F.avg("area_pct"))
)
asm_d0 = VectorAssembler(inputCols=FEATURES_COM_D0, outputCol="features_raw")
scl_d0 = StandardScaler(inputCol="features_raw", outputCol="features",
                         withMean=True, withStd=True)
df_com_d0 = Pipeline(stages=[asm_d0, scl_d0]).fit(fm_com_d0).transform(fm_com_d0)

print("\n=== Comparação Silhouette: com D0 vs sem D0 ===")
print(f"{'K':<5} {'Com D0':<15} {'Sem D0':<15} {'Melhoria'}")
for k in [2, 3, 4, 5, 6]:
    km = KMeans(featuresCol="features", k=k, seed=SEED)
    s_com = evaluator.evaluate(km.fit(df_com_d0).transform(df_com_d0))
    s_sem = evaluator.evaluate(km.fit(df_scaled).transform(df_scaled))
    print(f"{k:<5} {s_com:<15.4f} {s_sem:<15.4f} {s_sem - s_com:+.4f}")


# ============================================================================
# 8. ELBOW METHOD + SILHOUETTE — SEM D0
# ============================================================================
wcss = []
sil_scores = []
k_values = list(range(2, 11))

print("\n=== Elbow + Silhouette (sem D0) ===")
for k in k_values:
    km = KMeans(featuresCol="features", k=k, seed=SEED)
    m = km.fit(df_scaled)
    cost = m.summary.trainingCost
    score = evaluator.evaluate(m.transform(df_scaled))
    wcss.append(cost)
    sil_scores.append(score)
    print(f"  K={k} -> WCSS={cost:.4f} | Silhouette={score:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(k_values, wcss, marker="o", color="steelblue", linewidth=2)
axes[0].set_title("Elbow Method — Sem D0")
axes[0].set_xlabel("K")
axes[0].set_ylabel("WCSS")
axes[0].set_xticks(k_values)
axes[0].grid(alpha=0.3)

axes[1].plot(k_values, sil_scores, marker="o", color="coral", linewidth=2)
axes[1].set_title("Silhouette Score — Sem D0")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette")
axes[1].set_xticks(k_values)
axes[1].grid(alpha=0.3)

plt.suptitle("Escolha de K — Sem D0", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_elbow_silhouette_sem_d0.png", dpi=150)
plt.close()


# ============================================================================
# 9. ANÁLISE DE ESTABILIDADE — K=3
# ============================================================================
K_FINAL = 3

print(f"\n=== Estabilidade do clustering (K={K_FINAL}, sem D0) ===")
for seed in [42, 123, 456, 789, 1000]:
    km = KMeans(featuresCol="features", k=K_FINAL, seed=seed)
    s = evaluator.evaluate(km.fit(df_scaled).transform(df_scaled))
    print(f"  seed={seed} -> Silhouette = {s:.4f}")


# ============================================================================
# 10. MODELO FINAL — K=3, seed=42, sem D0
# ============================================================================
kmeans_final = KMeans(featuresCol="features", k=K_FINAL, seed=SEED)
model_final = kmeans_final.fit(df_scaled)
df_clustered = model_final.transform(df_scaled)

sil_final = evaluator.evaluate(df_clustered)
print(f"\n=== Modelo final: K={K_FINAL}, seed={SEED}, sem D0 ===")
print(f"Silhouette Score: {sil_final:.4f}")

print("\nCentróides (escala original):")
df_clustered.groupBy("prediction").agg(
    F.avg("D1").alias("media_D1"),
    F.avg("D2").alias("media_D2"),
    F.avg("D3").alias("media_D3"),
    F.avg("D4").alias("media_D4"),
    F.count("state_abb").alias("num_estados")
).orderBy("prediction").show()

print("Atribuição de cluster por estado:")
df_clustered.select("state_abb", "prediction") \
    .orderBy("prediction", "state_abb").show(60)


# ============================================================================
# 11. VISUALIZAÇÕES PÓS-CLUSTERING
# ============================================================================

nomes_clusters = {
    0: "Cluster 0 — Seca Moderada (Planícies)",
    1: "Cluster 1 — Baixa Seca (Este)",
    2: "Cluster 2 — Seca Severa (Sudoeste)"
}
cores_clusters = ["#fb8c00", "steelblue", "#e53935"]


# ----------------------------------------------------------------------------
# 11.1 Radar chart com escala partilhada
# Nota: escala partilhada essencial para comparação visual correcta entre clusters
# ----------------------------------------------------------------------------
centros_pd = (
    df_clustered.groupBy("prediction")
    .agg(F.avg("D1").alias("D1"), F.avg("D2").alias("D2"),
         F.avg("D3").alias("D3"), F.avg("D4").alias("D4"))
    .orderBy("prediction").toPandas()
)

n = len(FEATURES)
angles = [i * 2 * np.pi / n for i in range(n)] + [0]
max_val = centros_pd[FEATURES].values.max()  # escala partilhada

fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
for i, row in centros_pd.iterrows():
    valores = [row[f] for f in FEATURES] + [row["D1"]]
    axes[i].plot(angles, valores, color=cores_clusters[i], linewidth=2)
    axes[i].fill(angles, valores, color=cores_clusters[i], alpha=0.25)
    axes[i].set_xticks(angles[:-1])
    axes[i].set_xticklabels(FEATURES, fontsize=12)
    axes[i].set_title(nomes_clusters[i], size=10, pad=15)
    axes[i].set_ylim(0, max_val * 1.1)

plt.suptitle("Perfil de Seca por Cluster — K=3 sem D0 (escala partilhada)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_radar_clusters.png", dpi=150, bbox_inches="tight")
plt.close()


# ----------------------------------------------------------------------------
# 11.2 Boxplots — distribuição interna
# ----------------------------------------------------------------------------
df_box = df_clustered.select("state_abb", "D1", "D2", "D3", "D4", "prediction").toPandas()
df_box["cluster"] = df_box["prediction"].map({
    0: "C0 — Mod. Planícies", 1: "C1 — Baixa Este", 2: "C2 — Severa SO"
})

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for i, nivel in enumerate(FEATURES):
    sns.boxplot(data=df_box, x="cluster", y=nivel,
                ax=axes[i], palette=cores_clusters)
    axes[i].set_title(nivel)
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=30)

fig.suptitle("Distribuição de area_pct por Cluster (D1–D4)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_boxplots_clusters.png", dpi=150)
plt.close()


# ----------------------------------------------------------------------------
# 11.3 Mapa coroplético
# ----------------------------------------------------------------------------
mapa_pd = df_clustered.select("state_abb", "prediction").toPandas()
mapa_pd["cluster_nome"] = mapa_pd["prediction"].map(nomes_clusters)

fig_map = px.choropleth(
    mapa_pd, locations="state_abb", locationmode="USA-states",
    color="cluster_nome", scope="usa",
    title="Clustering de Estados por Perfil de Seca — K=3 sem D0",
    color_discrete_map={nomes_clusters[i]: cores_clusters[i] for i in range(3)}
)
fig_map.write_html(f"{OUTPUT_DIR}/10_mapa_clusters.html")


# ----------------------------------------------------------------------------
# 11.4 Mapa animado — evolução temporal
# ----------------------------------------------------------------------------
drought_idx_mes = drought.withColumn("peso",
    F.when(F.col("drought_lvl") == "None", 0)
     .when(F.col("drought_lvl") == "D0", 1)
     .when(F.col("drought_lvl") == "D1", 2)
     .when(F.col("drought_lvl") == "D2", 3)
     .when(F.col("drought_lvl") == "D3", 4)
     .when(F.col("drought_lvl") == "D4", 5)
).withColumn("ano", F.year("map_date")) \
 .withColumn("mes", F.month("map_date"))

drought_idx_mes = (
    drought_idx_mes
    .groupBy("state_abb", "ano", "mes")
    .agg((F.sum(F.col("area_pct") * F.col("peso")) / 100).alias("drought_index"))
    .withColumn("ano_mes",
        F.concat(F.col("ano"), F.lit("-"), F.lpad(F.col("mes"), 2, "0")))
    .orderBy("ano", "mes").toPandas()
)

fig_anim = px.choropleth(
    drought_idx_mes, locations="state_abb", locationmode="USA-states",
    color="drought_index", scope="usa", animation_frame="ano_mes",
    color_continuous_scale=[
        [0.0, "#ffffff"], [0.2, "#fdd835"], [0.4, "#fb8c00"],
        [0.6, "#e53935"], [0.8, "#880e4f"], [1.0, "#4a148c"]
    ],
    range_color=[0, 5],
    title="Evolução da Severidade de Seca nos EUA (2001–2021)"
)
fig_anim.write_html(f"{OUTPUT_DIR}/11_mapa_animado.html")


# ============================================================================
# 12. FIM
# ============================================================================
print("\n=== Análise concluída ===")
print(f"Outputs em: {OUTPUT_DIR}/")
print(f"Solução final: K={K_FINAL} | seed={SEED} | features=D1-D4 | Silhouette={sil_final:.4f}")

spark.stop()
