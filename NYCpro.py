
import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import gradio as gr


# ----------------------------
# 1) FILES (your provided names)
# ----------------------------
CSV_TOTAL_POPU = "Total_Popu_data-1766707998522.csv"               # Borough total population
CSV_METRO_ST   = "Metro_ST_data-1766706485612.csv"                # Stations per borough (or by area label)
CSV_METRO_100K = "Metro_100k_data-1766707034774.csv"              # Stations per 100k
CSV_POP_SUP_100K = "pop_sup_100k_data-1766708095860.csv"          # Pop + stations per 100k (combined)
CSV_POP_SUP = "POP_SUPdata-1766713160377.csv"                     # Pop + station count by region
CSV_METRO_MAP = "Metro_Map_data-1766876725818.csv"                # lat/lon points for stations (from your PostGIS query)
CSV_POP_HOTSPOT = "Popuulation_Hotspot_data-1766933668144.csv"    

REQUIRED_FILES = [
    CSV_TOTAL_POPU,
    CSV_METRO_ST,
    CSV_METRO_MAP,
    CSV_POP_HOTSPOT,
    # The rest are optional but recommended for more insights:
    CSV_METRO_100K,
    CSV_POP_SUP_100K,
    CSV_POP_SUP,
]


# ----------------------------
# 2) HELPERS
# ----------------------------
def _file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

def _read_csv_safely(path: str) -> pd.DataFrame:
    # tries common encodings and trims weird unnamed columns
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", regex=True)]
    return df

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")

def _clean_borough_names(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # normalize common forms (optional)
    mapping = {
        "bronx": "The Bronx",
        "the bronx": "The Bronx",
        "staten island": "Staten Island",
        "manhattan": "Manhattan",
        "queens": "Queens",
        "brooklyn": "Brooklyn",
    }
    s2 = s.str.lower().map(mapping).fillna(s)
    return s2

def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

def _map_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=620,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )
    return fig


# ----------------------------
# 3) LOAD & STANDARDIZE DATA
# ----------------------------
def load_all_data() -> tuple[dict, str]:
    """
    Returns:
      data dict with keys:
        total_pop, metro_st, metro_100k, pop_sup_100k, pop_sup, metro_map, pop_hotspot
      info string (status)
    """
    info_lines = []
    missing = [f for f in REQUIRED_FILES if not _file_exists(f)]
    if missing:
        info_lines.append("Missing files (put them next to the script):")
        info_lines.extend([f" - {m}" for m in missing])
        info_lines.append("The app will still run, but some tabs may be limited.")
    else:
        info_lines.append("All expected CSV files found ✅")

    data = {}

    # Required for charts
    if _file_exists(CSV_TOTAL_POPU):
        df = _read_csv_safely(CSV_TOTAL_POPU)

        boro_col = _first_existing_col(df, ["BORONAME", "boro", "borough", "BOROUGH"])
        pop_col  = _first_existing_col(df, ["total_population", "TOTAL_POPULATION", "population", "POPULATION", "pop_total", "POPN_TOTAL"])

        if boro_col and pop_col:
            df = df[[boro_col, pop_col]].copy()
            df.columns = ["BORONAME", "total_population"]
            df["BORONAME"] = _clean_borough_names(df["BORONAME"])
            df["total_population"] = _to_numeric(df, "total_population")
            df = df.dropna(subset=["BORONAME", "total_population"])
            df = df.groupby("BORONAME", as_index=False)["total_population"].sum()
            data["total_pop"] = df.sort_values("total_population", ascending=False)
            info_lines.append(f"Loaded {CSV_TOTAL_POPU}: {len(df):,} rows")
        else:
            info_lines.append(f"Could not detect BORONAME/Population columns in {CSV_TOTAL_POPU}")
    else:
        data["total_pop"] = pd.DataFrame(columns=["BORONAME", "total_population"])

    # Required for charts
    if _file_exists(CSV_METRO_ST):
        df = _read_csv_safely(CSV_METRO_ST)

        # try to find borough + station count columns
        boro_col = _first_existing_col(df, ["BORONAME", "boro", "borough", "BOROUGH"])
        st_col   = _first_existing_col(df, ["total_subway_stations", "total_stations", "stations", "station_count", "count", "TOTAL_SUBWAY_STATIONS"])

        if boro_col and st_col:
            df = df[[boro_col, st_col]].copy()
            df.columns = ["BORONAME", "total_subway_stations"]
            df["BORONAME"] = _clean_borough_names(df["BORONAME"])
            df["total_subway_stations"] = _to_numeric(df, "total_subway_stations")
            df = df.dropna(subset=["BORONAME", "total_subway_stations"])
            df = df.groupby("BORONAME", as_index=False)["total_subway_stations"].sum()
            data["metro_st"] = df.sort_values("total_subway_stations", ascending=False)
            info_lines.append(f"Loaded {CSV_METRO_ST}: {len(df):,} rows")
        else:
            info_lines.append(f"Could not detect BORONAME/Station count columns in {CSV_METRO_ST}")
    else:
        data["metro_st"] = pd.DataFrame(columns=["BORONAME", "total_subway_stations"])

    # Optional insights CSVs
    for key, path in [
        ("metro_100k", CSV_METRO_100K),
        ("pop_sup_100k", CSV_POP_SUP_100K),
        ("pop_sup", CSV_POP_SUP),
    ]:
        if _file_exists(path):
            df = _read_csv_safely(path)
            data[key] = df
            info_lines.append(f"Loaded {path}: {len(df):,} rows")
        else:
            data[key] = pd.DataFrame()

    # Required for maps (stations points)
    if _file_exists(CSV_METRO_MAP):
        df = _read_csv_safely(CSV_METRO_MAP)

        lat = _first_existing_col(df, ["latitude", "lat"])
        lon = _first_existing_col(df, ["longitude", "lon", "lng"])
        name = _first_existing_col(df, ["NAME", "name"])
        boro = _first_existing_col(df, ["BOROUGH", "borough", "BORONAME"])

        if lat and lon:
            out = pd.DataFrame()
            out["latitude"] = _to_numeric(df, lat)
            out["longitude"] = _to_numeric(df, lon)
            out["NAME"] = df[name].astype(str) if name else "Station"
            out["BOROUGH"] = _clean_borough_names(df[boro]) if boro else "Unknown"
            out = out.dropna(subset=["latitude", "longitude"])
            data["metro_map"] = out
            info_lines.append(f"Loaded {CSV_METRO_MAP}: {len(out):,} points")
        else:
            data["metro_map"] = pd.DataFrame(columns=["latitude", "longitude", "NAME", "BOROUGH"])
            info_lines.append(f"Could not detect latitude/longitude in {CSV_METRO_MAP}")
    else:
        data["metro_map"] = pd.DataFrame(columns=["latitude", "longitude", "NAME", "BOROUGH"])

    # Required for maps (population hotspot)
    if _file_exists(CSV_POP_HOTSPOT):
        df = _read_csv_safely(CSV_POP_HOTSPOT)

        lat = _first_existing_col(df, ["lat", "latitude"])
        lon = _first_existing_col(df, ["lon", "longitude"])
        pop = _first_existing_col(df, ["popn_total", "pop_total", "POPN_TOTAL", "population", "POPULATION"])
        boro = _first_existing_col(df, ["BORONAME", "borough", "BOROUGH"])

        if lat and lon and pop:
            out = pd.DataFrame()
            out["lat"] = _to_numeric(df, lat)
            out["lon"] = _to_numeric(df, lon)
            out["popn_total"] = _to_numeric(df, pop)
            out["BORONAME"] = _clean_borough_names(df[boro]) if boro else "Unknown"
            out = out.dropna(subset=["lat", "lon", "popn_total"])
            out = out[out["popn_total"] > 0]
            data["pop_hotspot"] = out
            info_lines.append(f"Loaded {CSV_POP_HOTSPOT}: {len(out):,} hotspot points")
        else:
            data["pop_hotspot"] = pd.DataFrame(columns=["lat", "lon", "popn_total", "BORONAME"])
            info_lines.append(f"Could not detect lat/lon/pop columns in {CSV_POP_HOTSPOT}")
    else:
        data["pop_hotspot"] = pd.DataFrame(columns=["lat", "lon", "popn_total", "BORONAME"])

    return data, "\n".join(info_lines)


# ----------------------------
# 4) CHARTS (Q1, Q2, INSIGHTS)
# ----------------------------
def run_q1_population(data: dict):
    df = data.get("total_pop", pd.DataFrame())
    if df.empty:
        return df, _empty_fig("Q1: Population by Borough"), "Q1: Missing/empty population dataset."

    fig = px.bar(
        df.sort_values("total_population", ascending=True),
        x="total_population",
        y="BORONAME",
        orientation="h",
        title="Q1: Population by Borough",
    )
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=60, b=20))
    return df, fig, f"Q1 ready — {len(df)} borough rows."

def run_q2_stations(data: dict):
    df = data.get("metro_st", pd.DataFrame())
    if df.empty:
        return df, _empty_fig("Q2: Subway Stations by Borough"), "Q2: Missing/empty stations dataset."

    fig = px.bar(
        df.sort_values("total_subway_stations", ascending=True),
        x="total_subway_stations",
        y="BORONAME",
        orientation="h",
        title="Q2: Subway Stations by Borough",
    )
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=60, b=20))
    return df, fig, f"Q2 ready — {len(df)} borough rows."

def run_insights(data: dict):
    """
    Builds a clean, consistent insight table by merging Q1/Q2.
    Adds derived metrics:
      - stations_per_100k (based on borough totals)
      - population_per_station
    """
    pop_df = data.get("total_pop", pd.DataFrame()).copy()
    st_df  = data.get("metro_st", pd.DataFrame()).copy()

    if pop_df.empty or st_df.empty:
        merged = pd.DataFrame(columns=[
            "BORONAME", "total_population", "total_subway_stations",
            "stations_per_100k", "population_per_station"
        ])
        fig1 = _empty_fig("Insight: Population vs Stations (Scatter)")
        fig2 = _empty_fig("Insight: Stations per 100k (Bar)")
        return merged, fig1, fig2, "Insights: need BOTH Q1 and Q2 data."

    merged = pop_df.merge(st_df, on="BORONAME", how="inner")
    merged["stations_per_100k"] = (merged["total_subway_stations"] / merged["total_population"]) * 100000
    merged["population_per_station"] = merged["total_population"] / merged["total_subway_stations"].replace(0, np.nan)
    merged = merged.sort_values("stations_per_100k", ascending=False)

    fig_scatter = px.scatter(
        merged,
        x="total_population",
        y="total_subway_stations",
        hover_name="BORONAME",
        title="Insight: Do more populated boroughs have more subway stations?",
    )
    fig_scatter.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=60, b=20))

    fig_100k = px.bar(
        merged.sort_values("stations_per_100k", ascending=True),
        x="stations_per_100k",
        y="BORONAME",
        orientation="h",
        title="Insight: Subway Stations per 100k People (by Borough)",
    )
    fig_100k.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=60, b=20))

    return merged, fig_scatter, fig_100k, "Insights ready — derived metrics created from Q1 + Q2."


# ----------------------------
# 5) MAPS (Population hotspot, Stations points, Combined overlay)
# ----------------------------
def map_population_hotspot(data: dict):
    df = data.get("pop_hotspot", pd.DataFrame())
    if df.empty:
        fig = _empty_fig("Map 1: Population Hotspot (Heatmap)")
        return fig, "Map 1: Missing/empty hotspot dataset."

    # density heatmap (hotspot) — uses OpenStreetMap tiles (no token)
    fig = px.density_mapbox(
        df,
        lat="lat",
        lon="lon",
        z="popn_total",
        radius=18,
        center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
        zoom=9.3,
        mapbox_style="open-street-map",
        title="Map 1: Population Hotspot (Census Blocks Heatmap)",
    )
    fig = _map_layout(fig, "Map 1: Population Hotspot (Census Blocks Heatmap)")
    return fig, f"Map 1 ready — {len(df):,} hotspot points."

def map_subway_points(data: dict):
    df = data.get("metro_map", pd.DataFrame())
    if df.empty:
        fig = _empty_fig("Map 2: Subway Stations (Points)")
        return fig, "Map 2: Missing/empty subway points dataset."

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="NAME",
        hover_data={"BOROUGH": True, "latitude": True, "longitude": True},
        zoom=9.3,
        center=dict(lat=df["latitude"].mean(), lon=df["longitude"].mean()),
        mapbox_style="open-street-map",
        title="Map 2: Subway Stations (Points)",
    )
    fig.update_traces(marker=dict(size=6))
    fig = _map_layout(fig, "Map 2: Subway Stations (Points)")
    return fig, f"Map 2 ready — {len(df):,} station points."

def map_combined_overlay(data: dict):
    pop = data.get("pop_hotspot", pd.DataFrame())
    st  = data.get("metro_map", pd.DataFrame())

    if pop.empty and st.empty:
        fig = _empty_fig("Map 3: Combined Overlay")
        return fig, "Map 3: Missing both population hotspot + subway points."

    # Start with population heatmap if available
    if not pop.empty:
        fig = px.density_mapbox(
            pop,
            lat="lat",
            lon="lon",
            z="popn_total",
            radius=18,
            center=dict(lat=pop["lat"].mean(), lon=pop["lon"].mean()),
            zoom=9.3,
            mapbox_style="open-street-map",
            title="Map 3: Combined Overlay (Population Hotspot + Subway Points)",
        )
    else:
        # fallback base map
        base_lat = st["latitude"].mean() if not st.empty else 40.7128
        base_lon = st["longitude"].mean() if not st.empty else -74.0060
        fig = go.Figure()
        fig.update_layout(
            mapbox=dict(style="open-street-map", center=dict(lat=base_lat, lon=base_lon), zoom=9.3),
            title="Map 3: Combined Overlay (Population Hotspot + Subway Points)",
        )

    # Add subway points on top
    if not st.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=st["latitude"],
                lon=st["longitude"],
                mode="markers",
                marker=dict(size=6),
                text=st["NAME"],
                name="Subway Stations",
                hovertemplate="<b>%{text}</b><br>Lat:%{lat}<br>Lon:%{lon}<extra></extra>",
            )
        )

    fig = _map_layout(fig, "Map 3: Combined Overlay (Population Hotspot + Subway Points)")
    return fig, "Map 3 ready — overlay rendered."


# ----------------------------
# 6) GRADIO APP (FIXED RETURN)
# ----------------------------
def build_app():
    data, startup_info = load_all_data()

    with gr.Blocks(title="NYC Storyline Dashboard — Population vs Transit") as app:
        gr.Markdown(
            "# NYC Storyline Dashboard — Population vs Transit\n"
            "**Goal:** Does public transportation density align with population distribution in NYC?\n\n"
            "**Order:** Charts (Q1 → Q2 → Insight) → Maps (Population Hotspot → Subway Points → Combined Overlay)\n"
        )

        with gr.Accordion("Data & Status", open=False):
            status_box = gr.Textbox(value=startup_info, lines=10, label="Status (read-only)", interactive=False)

        with gr.Tabs():
            # -------------------- TAB 1: CHARTS --------------------
            with gr.Tab("1) Charts (Q1, Q2, Insight)"):
                with gr.Row():
                    btn_q1 = gr.Button("Run Q1 (Population)", variant="primary")
                    btn_q2 = gr.Button("Run Q2 (Stations)", variant="primary")
                    btn_ins = gr.Button("Run Insight (Population vs Stations)", variant="secondary")

                with gr.Row():
                    q1_table = gr.Dataframe(label="Q1 Table (Population by Borough)", interactive=False)
                    q2_table = gr.Dataframe(label="Q2 Table (Stations by Borough)", interactive=False)

                with gr.Row():
                    q1_plot = gr.Plot(label="Q1 Plot")
                    q2_plot = gr.Plot(label="Q2 Plot")

                with gr.Row():
                    ins_table = gr.Dataframe(label="Insight Table (Derived Metrics)", interactive=False)

                with gr.Row():
                    ins_plot1 = gr.Plot(label="Insight Plot 1 (Scatter)")
                    ins_plot2 = gr.Plot(label="Insight Plot 2 (Stations per 100k)")

                info_charts = gr.Textbox(label="Info", lines=3, interactive=False)

                def _ui_q1():
                    df, fig, msg = run_q1_population(data)
                    return df, fig, msg

                def _ui_q2():
                    df, fig, msg = run_q2_stations(data)
                    return df, fig, msg

                def _ui_ins():
                    df, fig1, fig2, msg = run_insights(data)
                    return df, fig1, fig2, msg

                btn_q1.click(_ui_q1, inputs=None, outputs=[q1_table, q1_plot, info_charts])
                btn_q2.click(_ui_q2, inputs=None, outputs=[q2_table, q2_plot, info_charts])
                btn_ins.click(_ui_ins, inputs=None, outputs=[ins_table, ins_plot1, ins_plot2, info_charts])

            # -------------------- TAB 2: MAPS --------------------
            with gr.Tab("2) Maps (Hotspot, Stations, Combined)"):
                with gr.Row():
                    btn_m1 = gr.Button("Show Population Hotspot (Map 1)", variant="primary")
                    btn_m2 = gr.Button("Show Subway Stations (Points) (Map 2)", variant="primary")
                    btn_m3 = gr.Button("Show Combined Overlay (Map 3)", variant="secondary")

                map1 = gr.Plot(label="Map 1")
                map2 = gr.Plot(label="Map 2")
                map3 = gr.Plot(label="Map 3")

                info_maps = gr.Textbox(label="Info", lines=3, interactive=False)

                def _ui_m1():
                    fig, msg = map_population_hotspot(data)
                    return fig, msg

                def _ui_m2():
                    fig, msg = map_subway_points(data)
                    return fig, msg

                def _ui_m3():
                    fig, msg = map_combined_overlay(data)
                    return fig, msg

                btn_m1.click(_ui_m1, inputs=None, outputs=[map1, info_maps])
                btn_m2.click(_ui_m2, inputs=None, outputs=[map2, info_maps])
                btn_m3.click(_ui_m3, inputs=None, outputs=[map3, info_maps])

    return app  # ✅ IMPORTANT: return the Blocks object (NOT demo)


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)





