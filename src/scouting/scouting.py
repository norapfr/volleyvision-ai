import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go

# ── carga de datos ────────────────────────────────────────────────
with open("data/processed/detections_log.json") as f:
    detections_log = json.load(f)
with open("data/processed/events_log.json") as f:
    events_log = json.load(f)

df_det = pd.DataFrame(detections_log)
df_evt = pd.DataFrame(events_log)

print(f"Detecciones: {len(df_det)}")
print(f"Eventos: {len(df_evt)}")
print(f"\nEventos por clase:\n{df_evt['event'].value_counts()}")

# ── construcción de perfiles por zona ────────────────────────────
# dividimos la pista en 6 zonas (3x2) y construimos un perfil
# estadístico por zona basado en las detecciones y eventos cercanos

persons = df_det[df_det["class"] == "person"].copy()

# cuadrícula 3 columnas x 2 filas = 6 zonas
persons["zone_x"] = pd.cut(persons["cx"],
    bins=[0, 0.33, 0.67, 1.0],
    labels=["izq", "centro", "der"],
    include_lowest=True)

persons["zone_y"] = pd.cut(persons["cy"],
    bins=[0, 0.5, 1.0],
    labels=["arriba", "abajo"],
    include_lowest=True)

persons["zone"] = persons["zone_x"].astype(str) + "_" + persons["zone_y"].astype(str)

# eventos por clase
EVENT_CLASSES = ["attack", "block", "reception", "service",
                 "setting", "Defense-Move"]

profiles = []
for zone, group in persons.groupby("zone"):
    event_counts = {e: 0 for e in EVENT_CLASSES}

    # cuenta eventos que ocurren cerca en el tiempo a detecciones de esta zona
    for _, row in group.iterrows():
        nearby = df_evt[abs(df_evt["frame"] - row["frame"]) < 10]
        for evt in nearby["event"]:
            if evt in event_counts:
                event_counts[evt] += 1

    profile = {
        "zone":        zone,
        "appearances": len(group),
        "avg_cx":      round(group["cx"].mean(), 3),
        "avg_cy":      round(group["cy"].mean(), 3),
        **event_counts
    }
    profiles.append(profile)

df_profiles = pd.DataFrame(profiles)
print(f"\nPerfiles construidos:\n{df_profiles[['zone','appearances'] + EVENT_CLASSES]}")

# ── PCA ──────────────────────────────────────────────────────────
features = ["appearances"] + EVENT_CLASSES
X = df_profiles[features].fillna(0).values
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df_profiles["PC1"] = components[:, 0]
df_profiles["PC2"] = components[:, 1]

print(f"\nVarianza explicada por PCA: {pca.explained_variance_ratio_.sum():.2%}")

# ── K-Means ───────────────────────────────────────────────────────
n_clusters = min(3, len(df_profiles))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_profiles["cluster"] = kmeans.fit_predict(X_scaled).astype(str)

cluster_names = {"0": "Defensivo", "1": "Atacante", "2": "Equilibrado"}
df_profiles["perfil"] = df_profiles["cluster"].map(cluster_names)

print(f"\nClusters asignados:\n{df_profiles[['zone','perfil','appearances']]}")

# ── KNN — jugadores similares ─────────────────────────────────────
nn = NearestNeighbors(n_neighbors=min(3, len(df_profiles)), metric="euclidean")
nn.fit(X_scaled)

print("\n Zonas con perfil similar a cada zona ")
for i, row in df_profiles.iterrows():
    distances, indices = nn.kneighbors([X_scaled[i]])
    similares = []
    for dist, idx in zip(distances[0][1:], indices[0][1:]):
        similares.append(f"{df_profiles.iloc[idx]['zone']} (sim={1/(1+dist):.2f})")
    print(f"  {row['zone']:20s} - similares: {', '.join(similares)}")

# ── visualización PCA + clusters ──────────────────────────────────
fig = px.scatter(
    df_profiles,
    x="PC1", y="PC2",
    color="perfil",
    size="appearances",
    hover_data=["zone", "appearances"] + EVENT_CLASSES,
    title="Mapa de perfiles tácticos por zona (PCA + K-Means)",
    labels={"PC1": "Componente 1 (ataque/defensa)",
            "PC2": "Componente 2 (presencia)"},
    color_discrete_map={
        "Defensivo":   "#19D3F3",
        "Atacante":    "#FF4B4B",
        "Equilibrado": "#AB63FA"
    }
)

# añade etiquetas de zona
for _, row in df_profiles.iterrows():
    fig.add_annotation(
        x=row["PC1"], y=row["PC2"],
        text=row["zone"],
        showarrow=False,
        yshift=12,
        font=dict(size=10, color="white")
    )

fig.update_layout(
    plot_bgcolor="#1E1E1E",
    paper_bgcolor="#1E1E1E",
    font_color="white",
    height=500
)

fig.show()
print("\n Scouting completado")