import streamlit as st
import cv2
import json
import tempfile
from pathlib import Path
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="VolleyVision AI",
    page_icon="🏐",
    layout="wide"
)

st.title("🏐 VolleyVision AI")
st.markdown("*Sistema automático de análisis táctico para voleibol*")

@st.cache_resource
def load_models():
    from huggingface_hub import hf_hub_download
    from pathlib import Path

    Path("models").mkdir(exist_ok=True)

    detection_path = "models/volleyvision_detection.pt"
    events_path    = "models/volleyvision_events.pt"

    if not Path(detection_path).exists():
        with st.spinner("⬇️ Descargando modelo de detección..."):
            detection_path = hf_hub_download(
                repo_id="norapfr/volleyvision-models",
                filename="volleyvision_detection.pt",
                local_dir="models"
            )

    if not Path(events_path).exists():
        with st.spinner("⬇️ Descargando modelo de eventos..."):
            events_path = hf_hub_download(
                repo_id="norapfr/volleyvision-models",
                filename="volleyvision_events.pt",
                local_dir="models"
            )

    detection = YOLO(detection_path)
    events    = YOLO(events_path)
    return detection, events

detection_model, events_model = load_models()

COLORES = {
    "attack":       "#FF4B4B",
    "block":        "#FFA500",
    "reception":    "#00CC96",
    "service":      "#636EFA",
    "setting":      "#AB63FA",
    "Defense-Move": "#19D3F3",
}

CONF_POR_CLASE = {
    "service":      0.75,
    "block":        0.45,
    "attack":       0.45,
    "reception":    0.45,
    "setting":      0.45,
    "Defense-Move": 0.50,
}

EMOJIS = {
    "attack":       "⚡ Attack",
    "block":        "🛡 Block",
    "reception":    "🤲 Reception",
    "service":      "📤 Service",
    "setting":      "🎯 Setting",
    "Defense-Move": "🏃 Defense-Move",
}

PERFIL_COLORES = {
    "Atacante":    "#FF4B4B",
    "Defensivo":   "#19D3F3",
    "Equilibrado": "#AB63FA",
}

PERFIL_BG = {
    "Atacante":    "#FAECE7",
    "Defensivo":   "#E6F1FB",
    "Equilibrado": "#EEEDFE",
}

PERFIL_TEXT = {
    "Atacante":    "#993C1D",
    "Defensivo":   "#185FA5",
    "Equilibrado": "#3C3489",
}

ZONA_NOMBRES = {
    "izq_arriba":    "Izq · primera línea",
    "centro_arriba": "Centro · primera línea",
    "der_arriba":    "Der · primera línea",
    "izq_abajo":     "Izq · fondo de pista",
    "centro_abajo":  "Centro · fondo (libero)",
    "der_abajo":     "Der · fondo de pista",
}

with st.sidebar:
    st.header("⚙️ Configuración")
    conf_detection = st.slider("Confianza detección", 0.1, 0.9, 0.4)
    every_n        = st.slider("Analizar 1 de cada N frames", 1, 15, 3)
    motion_thresh  = st.slider("Umbral de movimiento", 0.5, 10.0, 2.0)
    st.markdown("---")
    st.info(
        "📌 **Mejor rendimiento con:**\n"
        "- Cámara fija y cenital\n"
        "- Pista completa visible\n"
        "- Sin zoom ni cambios de plano\n"
        "- Buena iluminación"
    )
    st.markdown("---")
    st.markdown("**Clases detectadas:**")
    for k, v in EMOJIS.items():
        st.markdown(f"- {v}")

uploaded = st.file_uploader("Sube un vídeo de partido", type=["mp4", "avi", "mov"])

if uploaded:
    if st.session_state.get("last_uploaded") != uploaded.name:
        for key in ["detections_log", "events_log", "heatmap_points", "frames_analyzed"]:
            st.session_state.pop(key, None)
        st.session_state["last_uploaded"] = uploaded.name

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("🚀 Analizar partido", type="primary"):

        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)

        detections_log   = []
        events_log       = []
        heatmap_points   = []
        last_event       = None
        last_event_frame = -15
        prev_gray        = None

        progress = st.progress(0, text="Analizando vídeo...")
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % every_n == 0:
                h, w = frame.shape[:2]

                det_results = detection_model(frame, conf=conf_detection, verbose=False)
                for box in det_results[0].boxes:
                    cls_name = detection_model.names[int(box.cls)]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    detections_log.append({
                        "frame": frame_id,
                        "time":  round(frame_id / fps, 2),
                        "class": cls_name,
                        "conf":  round(float(box.conf), 3),
                        "cx":    round(cx, 3),
                        "cy":    round(cy, 3)
                    })
                    if cls_name == "person":
                        heatmap_points.append((cx, cy))

                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = cv2.absdiff(prev_gray, curr_gray).mean() if prev_gray is not None else 0.0
                prev_gray = curr_gray

                if motion > motion_thresh:
                    evt_results = events_model(frame, conf=0.25, verbose=False)
                    for box in evt_results[0].boxes:
                        cls_name = events_model.names[int(box.cls)]
                        conf_val = float(box.conf)
                        umbral   = CONF_POR_CLASE.get(cls_name, 0.45)

                        if (cls_name != "stand" and
                            conf_val >= umbral and
                            (cls_name != last_event or
                             frame_id - last_event_frame > 15)):

                            events_log.append({
                                "frame":  frame_id,
                                "time":   round(frame_id / fps, 2),
                                "event":  cls_name,
                                "conf":   round(conf_val, 3),
                                "motion": round(motion, 2)
                            })
                            last_event       = cls_name
                            last_event_frame = frame_id

                progress.progress(
                    min(frame_id / total, 1.0),
                    text=f"Analizando frame {frame_id}/{total}..."
                )

            frame_id += 1

        cap.release()
        progress.progress(1.0, text="✅ Análisis completado")

        Path("data/processed").mkdir(exist_ok=True, parents=True)
        with open("data/processed/detections_log.json", "w") as f:
            json.dump(detections_log, f)
        with open("data/processed/events_log.json", "w") as f:
            json.dump(events_log, f)

        st.session_state["detections_log"]  = detections_log
        st.session_state["events_log"]      = events_log
        st.session_state["heatmap_points"]  = heatmap_points
        st.session_state["frames_analyzed"] = frame_id // every_n

# ── Resultados ────────────────────────────────────────────────────
if "detections_log" in st.session_state:

    detections_log  = st.session_state["detections_log"]
    events_log      = st.session_state["events_log"]
    heatmap_points  = st.session_state["heatmap_points"]
    frames_analyzed = st.session_state["frames_analyzed"]

    st.markdown("---")
    st.header("📊 Resultados del análisis")

    df_det = pd.DataFrame(detections_log)
    df_evt = pd.DataFrame(events_log)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames analizados",   frames_analyzed)
    col2.metric("Detecciones totales", len(detections_log))
    col3.metric("Eventos detectados",  len(events_log))
    if not df_evt.empty:
        col4.metric("Evento más frecuente",
                    df_evt["event"].value_counts().index[0])

    if df_evt.empty:
        st.warning("No se detectaron eventos. Prueba a bajar el umbral de movimiento o usa un vídeo con cámara fija cenital.")
    else:
        st.subheader("⏱️ Línea de tiempo de eventos")
        fig_timeline = px.scatter(
            df_evt, x="time", y="event", color="event", size="conf",
            color_discrete_map=COLORES,
            title="Eventos detectados a lo largo del partido",
            labels={"time": "Tiempo (s)", "event": "Evento"}
        )
        fig_timeline.update_traces(marker=dict(sizemin=10))
        fig_timeline.update_layout(
            plot_bgcolor="#1E1E1E", paper_bgcolor="#1E1E1E",
            font_color="white", height=350
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.subheader("📈 Distribución de eventos")
        counts = df_evt["event"].value_counts().reset_index()
        counts.columns = ["event", "count"]
        fig_bar = px.bar(
            counts, x="event", y="count", color="event",
            color_discrete_map=COLORES,
            title="Frecuencia de cada evento",
            labels={"event": "Evento", "count": "Nº detecciones"},
            text="count"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            plot_bgcolor="#1E1E1E", paper_bgcolor="#1E1E1E",
            font_color="white", showlegend=False, height=400,
            yaxis=dict(gridcolor="#333")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🕐 Intensidad táctica por momento del partido")
        max_time = df_evt["time"].max()
        if max_time > 0:
            df_evt["momento"] = pd.cut(
                df_evt["time"],
                bins=[0, max_time / 3, 2 * max_time / 3, max_time],
                labels=["Inicio", "Medio", "Final"],
                include_lowest=True
            )
            pivot = (df_evt.groupby(["momento", "event"], observed=True)
                     .size().reset_index(name="count"))
            fig_momento = px.bar(
                pivot, x="event", y="count", color="momento", barmode="group",
                color_discrete_map={"Inicio": "#4CC9F0", "Medio": "#F72585", "Final": "#7209B7"},
                title="Eventos por tipo y momento del partido",
                labels={"event": "Evento", "count": "Nº eventos", "momento": "Momento"},
                text="count"
            )
            fig_momento.update_traces(textposition="outside")
            fig_momento.update_layout(
                plot_bgcolor="#1E1E1E", paper_bgcolor="#1E1E1E", font_color="white",
                height=420, yaxis=dict(gridcolor="#333"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_momento, use_container_width=True)

            st.subheader("🧠 Resumen táctico")
            total_eventos = len(df_evt)
            col_a, col_b, col_c = st.columns(3)
            for momento, col in zip(["Inicio", "Medio", "Final"], [col_a, col_b, col_c]):
                sub = df_evt[df_evt["momento"] == momento]
                if not sub.empty:
                    top = sub["event"].value_counts().index[0]
                    pct = len(sub) / total_eventos * 100
                    col.metric(
                        f"{'🟦' if momento=='Inicio' else '🟥' if momento=='Medio' else '🟪'} {momento}",
                        f"{len(sub)} eventos",
                        f"Predomina: {top} ({pct:.0f}%)"
                    )

    # ── heatmap ───────────────────────────────────────────────
    if heatmap_points:
        st.subheader("🔥 Heatmap de posiciones de jugadores")
        st.caption("Las zonas más cálidas indican mayor presencia de jugadores durante el partido.")

        xs = [p[0] for p in heatmap_points]
        ys = [p[1] for p in heatmap_points]

        fig_heat = go.Figure()
        fig_heat.add_trace(go.Histogram2dContour(
            x=xs, y=ys, colorscale="Hot", reversescale=True,
            contours=dict(coloring="heatmap", showlabels=False),
            showscale=True, opacity=0.9, ncontours=20
        ))

        # red en la parte superior (y=0.10)
        fig_heat.add_shape(type="rect",
            x0=0.03, y0=0.10, x1=0.97, y1=0.92,
            line=dict(color="white", width=3))
        fig_heat.add_shape(type="line",
            x0=0.03, y0=0.10, x1=0.97, y1=0.10,
            line=dict(color="white", width=5))
        for x in [0.33, 0.67]:
            fig_heat.add_shape(type="line",
                x0=x, y0=0.10, x1=x, y1=0.92,
                line=dict(color="rgba(255,255,255,0.6)", width=2, dash="dot"))

        fig_heat.add_annotation(x=0.5, y=0.06, text="<b>RED</b>",
            showarrow=False, font=dict(size=13, color="white"))

        for txt, x, y in [
            ("Z4\nPrimera línea", 0.17, 0.20),
            ("Z3\nPrimera línea", 0.50, 0.20),
            ("Z2\nPrimera línea", 0.83, 0.20),
            ("Z5\nFondo",         0.17, 0.82),
            ("Z6\nLibero",        0.50, 0.82),
            ("Z1\nFondo",         0.83, 0.82),
        ]:
            fig_heat.add_annotation(x=x, y=y, text=f"<b>{txt}</b>",
                showarrow=False, font=dict(size=11, color="rgba(255,255,255,0.8)"), align="center")

        fig_heat.update_layout(
            plot_bgcolor="#0a0a0a", paper_bgcolor="#1E1E1E",
            xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[1, 0], showgrid=False, zeroline=False, visible=False),
            margin=dict(l=10, r=80, t=10, b=10), height=520
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("⚠️ El heatmap muestra posiciones en coordenadas de imagen. Para una vista cenital precisa se requiere transformación de perspectiva (próxima versión).")

    # ── Scouting ML ───────────────────────────────────────────
    if not df_det.empty:
        st.markdown("---")
        st.header("🧠 Scouting ML — perfiles tácticos por zona")
        st.markdown("> El sistema divide la pista en 6 zonas y analiza qué tipo de acciones ocurren en cada una. Aplica PCA, K-Means y KNN para agrupar zonas por perfil táctico y encontrar zonas con comportamiento similar.")

        with st.expander("ℹ️ Cómo interpretar los resultados"):
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                st.markdown("**🗺️ Mapa de la pista**")
                st.markdown("""
- La red está en la parte **superior** del mapa
- Zonas `_arriba` = primera línea (cerca de la red)
- Zonas `_abajo` = segunda línea (fondo de pista)
- El color indica el perfil táctico asignado por K-Means
                """)
            with col_exp2:
                st.markdown("**🎨 Perfiles (colores)**")
                st.markdown("""
- 🔴 **Atacante** — alta actividad ofensiva (attacks, settings)
- 🔵 **Defensivo** — zona de red, solo bloqueos esporádicos
- 🟣 **Equilibrado** — mezcla de ataque y defensa
                """)
            with col_exp3:
                st.markdown("**🔍 Similitud (KNN)**")
                st.markdown("""
- Va de 0 a 1. Cuanto más alto, más parecido el perfil táctico
- **> 0.7** — muy similares
- **0.4–0.7** — similitud moderada
- **< 0.4** — perfiles distintos
                """)

        EVENT_CLASSES = ["attack", "block", "reception", "service", "setting", "Defense-Move"]

        persons = df_det[df_det["class"] == "person"].copy()
        persons["zone_x"] = pd.cut(persons["cx"],
            bins=[0, 0.33, 0.67, 1.0], labels=["izq", "centro", "der"], include_lowest=True)
        persons["zone_y"] = pd.cut(persons["cy"],
            bins=[0, 0.5, 1.0], labels=["arriba", "abajo"], include_lowest=True)
        persons["zone"] = persons["zone_x"].astype(str) + "_" + persons["zone_y"].astype(str)

        profiles = []
        for zone, group in persons.groupby("zone"):
            event_counts = {e: 0 for e in EVENT_CLASSES}
            for _, row in group.iterrows():
                nearby = df_evt[abs(df_evt["frame"] - row["frame"]) < 10] if not df_evt.empty else pd.DataFrame()
                for evt in nearby.get("event", []):
                    if evt in event_counts:
                        event_counts[evt] += 1
            profiles.append({
                "zone": zone, "appearances": len(group),
                "avg_cx": round(group["cx"].mean(), 3),
                "avg_cy": round(group["cy"].mean(), 3),
                **event_counts
            })

        df_profiles = pd.DataFrame(profiles)
        features  = ["appearances"] + EVENT_CLASSES
        X         = df_profiles[features].fillna(0).values
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X)

        pca   = PCA(n_components=2)
        comps = pca.fit_transform(X_scaled)
        df_profiles["PC1"] = comps[:, 0]
        df_profiles["PC2"] = comps[:, 1]

        n_clusters = min(3, len(df_profiles))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_profiles["cluster"] = kmeans.fit_predict(X_scaled).astype(str)
        cluster_names = {"0": "Defensivo", "1": "Atacante", "2": "Equilibrado"}
        df_profiles["perfil"] = df_profiles["cluster"].map(cluster_names)

        nn = NearestNeighbors(n_neighbors=min(3, len(df_profiles)), metric="euclidean")
        nn.fit(X_scaled)

        # ── mapa de la pista ──────────────────────────────────
        st.subheader("🏐 Mapa de la pista")
        st.caption("La red está arriba. Pulsa 'Ver zona' para explorar el perfil de cada zona.")

        # red
        st.markdown("""
        <div style='text-align:center; padding:6px 12px; background:#2a2a2a;
                    border-radius:6px; color:#aaa; font-size:12px;
                    letter-spacing:3px; margin-bottom:8px;'>
            ━━━━━━━━━━━━  RED  ━━━━━━━━━━━━
        </div>
        """, unsafe_allow_html=True)

        zona_orden = [
            ["izq_arriba", "centro_arriba", "der_arriba"],
            ["izq_abajo",  "centro_abajo",  "der_abajo"],
        ]

        zona_sel = st.session_state.get("zona_sel", "centro_abajo")

        for fila in zona_orden:
            cols = st.columns(3)
            for col, zona in zip(cols, fila):
                row_data = df_profiles[df_profiles["zone"] == zona]
                if not row_data.empty:
                    perf  = row_data.iloc[0]["perfil"]
                    apps  = int(row_data.iloc[0]["appearances"])
                    bg    = PERFIL_BG.get(perf, "#333")
                    clr   = PERFIL_TEXT.get(perf, "#fff")
                    brd   = f"3px solid {PERFIL_COLORES.get(perf,'#888')}" if zona == zona_sel else "2px solid transparent"
                    with col:
                        st.markdown(f"""
                        <div style='background:{bg}; border:{brd}; border-radius:10px;
                                    padding:10px 8px; text-align:center; margin-bottom:4px;'>
                            <div style='font-size:11px; color:#666; margin-bottom:4px;'>
                                {ZONA_NOMBRES.get(zona, zona)}
                            </div>
                            <div style='font-size:15px; font-weight:500; color:{clr};'>
                                {apps:,}
                            </div>
                            <div style='font-size:10px; color:#888;'>apariciones</div>
                            <div style='margin-top:6px; display:inline-block; font-size:11px;
                                        font-weight:500; color:{clr}; background:white;
                                        border-radius:20px; padding:2px 10px;'>{perf}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"Ver {zona}", key=f"btn_{zona}"):
                            st.session_state["zona_sel"] = zona
                            st.rerun()

        st.markdown("<div style='text-align:center; color:#666; font-size:11px; margin-top:2px;'>↑ Fondo de pista</div>", unsafe_allow_html=True)

        # ── detalle zona seleccionada ─────────────────────────
        st.markdown("---")
        zona_sel = st.session_state.get("zona_sel", "centro_abajo")
        row_sel  = df_profiles[df_profiles["zone"] == zona_sel]

        if not row_sel.empty:
            rd   = row_sel.iloc[0]
            perf = rd["perfil"]
            clr  = PERFIL_COLORES.get(perf, "#888")

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.subheader(f"📍 {zona_sel}")
                st.caption(f"{ZONA_NOMBRES.get(zona_sel, '')} · {int(rd['appearances']):,} apariciones · perfil **{perf}**")
                st.markdown("**Acciones detectadas:**")
                max_val = max([rd[e] for e in EVENT_CLASSES] + [1])
                for evt in EVENT_CLASSES:
                    val = int(rd[evt])
                    pct = int(val / max_val * 100)
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"""
                        <div style='margin-bottom:8px;'>
                            <div style='font-size:12px; color:#aaa; margin-bottom:3px;'>{evt}</div>
                            <div style='background:#2a2a2a; border-radius:4px; height:8px;'>
                                <div style='background:{COLORES.get(evt,"#888")}; width:{pct}%;
                                            height:8px; border-radius:4px;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div style='font-size:12px; color:#ccc; padding-top:18px;'>{val}</div>",
                                    unsafe_allow_html=True)

            with col_d2:
                st.subheader("🔍 Zonas similares")
                st.caption("Zonas con perfil táctico parecido según KNN")
                idx_sel = df_profiles[df_profiles["zone"] == zona_sel].index[0]
                distances, indices = nn.kneighbors([X_scaled[idx_sel]])
                for dist, idx in zip(distances[0][1:], indices[0][1:]):
                    sim      = round(1 / (1 + dist), 2)
                    zona_sim = df_profiles.iloc[idx]["zone"]
                    perf_sim = df_profiles.iloc[idx]["perfil"]
                    apps_sim = int(df_profiles.iloc[idx]["appearances"])
                    clr_sim  = PERFIL_COLORES.get(perf_sim, "#888")
                    pct_sim  = int(sim * 100)
                    st.markdown(f"""
                    <div style='background:#1e1e1e; border:1px solid #333; border-radius:10px;
                                padding:12px 14px; margin-bottom:10px;'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <div style='font-size:14px; font-weight:500; color:white;'>{zona_sim}</div>
                                <div style='font-size:11px; color:{clr_sim}; margin-top:2px;'>
                                    {perf_sim} · {apps_sim:,} apariciones
                                </div>
                            </div>
                            <div style='font-size:20px; font-weight:500; color:#378ADD;'>{sim:.2f}</div>
                        </div>
                        <div style='background:#333; border-radius:3px; height:6px; margin-top:10px;'>
                            <div style='background:#378ADD; width:{pct_sim}%;
                                        height:6px; border-radius:3px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── tabla resumen ─────────────────────────────────────
        st.subheader("📋 Perfiles detallados por zona")
        st.dataframe(
            df_profiles[["zone", "perfil", "appearances"] + EVENT_CLASSES]
            .sort_values("appearances", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )

        # ── PCA ───────────────────────────────────────────────
        st.subheader("📉 Mapa PCA de perfiles")
        st.caption(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.1%} — cada punto es una zona. Los puntos cercanos tienen perfil táctico similar.")
        fig_pca = px.scatter(
            df_profiles, x="PC1", y="PC2", color="perfil", size="appearances",
            hover_data=["zone", "appearances"] + EVENT_CLASSES,
            title="Distribución táctica de zonas (PCA + K-Means)",
            labels={"PC1": "Componente 1 (ataque/defensa)", "PC2": "Componente 2 (presencia)"},
            color_discrete_map=PERFIL_COLORES
        )
        for _, row in df_profiles.iterrows():
            fig_pca.add_annotation(
                x=row["PC1"], y=row["PC2"], text=row["zone"],
                showarrow=False, yshift=14, font=dict(size=10, color="white")
            )
        fig_pca.update_layout(
            plot_bgcolor="#1E1E1E", paper_bgcolor="#1E1E1E",
            font_color="white", height=420
        )
        st.plotly_chart(fig_pca, use_container_width=True)