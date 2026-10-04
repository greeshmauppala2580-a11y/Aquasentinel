import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import alerts
import engine
from simulator import PondTwin

st.set_page_config(page_title="AquaSentinel", page_icon="🐟", layout="wide")
st.title("🐟 AquaSentinel - Pond Early-Warning Dashboard")


def init():
    """Create the virtual pond, run 46 h of normal history, and train the AI on it."""
    twin = PondTwin()
    hist = [engine.enrich(twin.step()) for _ in range(46 * 12)]
    detector = engine.Detector(pd.DataFrame(hist[144:]))     # skip the first 12 h of start-up
    st.session_state.update(twin=twin, hist=hist, detector=detector, run=True,
                            alert_log=[], sent={}, last=None)

if "twin" not in st.session_state:
    init()

# ---------------- Sidebar controls ----------------
sb = st.sidebar
species = sb.selectbox("Species", list(engine.PROFILES))
sb.subheader("Inject a scenario")
heat = sb.checkbox("🔥 Heatwave")
feed = sb.checkbox("🍽️ Overfeeding")
bloom = sb.checkbox("🟢 Algal bloom")
fault = sb.checkbox("🛠️ DO sensor fault")
st.session_state.twin.set_scenarios(heatwave=heat, overfeed=feed, bloom=bloom, sensor_fault=fault)
aer_mode = sb.radio("Aerator", ["Auto (AI)", "Force ON", "Force OFF"])
speed = sb.slider("Speed (simulated min per refresh)", 5, 60, 30, step=5)
st.session_state.run = sb.toggle("▶ Run simulation", value=st.session_state.run)
if sb.button("Reset pond"):
    init(); st.rerun()
sb.caption("Telegram alerts: " + ("ON ✅" if alerts.telegram_ready() else "off (set env vars to enable)"))


def tick():
    s = st.session_state
    tw, prof = s.twin, engine.PROFILES[species]
    for _ in range(speed // 5):
        s.hist.append(engine.enrich(tw.step()))
    s.hist = s.hist[-900:]                                  # keep memory small
    df = pd.DataFrame(s.hist)
    r = s.hist[-1]
    fc = engine.forecast_do(df)
    risk = engine.risk_from_forecast(fc, prof)
    fl = engine.sensor_fault(df)
    anomaly = s.detector.is_anomaly(r)
    recs = engine.advise(r, prof, risk, fl, anomaly, r["time"])
    if aer_mode == "Auto (AI)":
        tw.aerator = engine.aerator_decision(r, prof, risk, tw.aerator)
    else:
        tw.aerator = aer_mode == "Force ON"
    for level, why, action in recs:                         # send each alert at most every 3 sim-hours
        if level in ("critical", "warning") and tw.t_min - s.sent.get(why[:30], -999) > 180:
            s.sent[why[:30]] = tw.t_min
            msg = f"[{level.upper()}] {r['time']:%d %b %H:%M} - {why}. ACTION: {action}"
            s.alert_log.insert(0, msg)
            alerts.send_telegram("AquaSentinel " + msg)
    s.last = dict(df=df, r=r, fc=fc, risk=risk, recs=recs, prof=prof)


@st.fragment(run_every=2)
def live():
    if st.session_state.run:
        tick()
    elif st.session_state.last is None:
        tick()
    L = st.session_state.last
    df, r, fc, risk, recs, prof = (L[k] for k in ("df", "r", "fc", "risk", "recs", "prof"))
    score, status, subs = engine.health_score(r, prof)

    c1, c2 = st.columns([1, 2])
    with c1:
        g = go.Figure(go.Indicator(mode="gauge+number", value=score, title={"text": f"Pond Health: {status}"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                   "steps": [{"range": [0, 50], "color": "#f8b4b4"}, {"range": [50, 75], "color": "#fde68a"},
                             {"range": [75, 100], "color": "#bbf7d0"}]}))
        g.update_layout(height=260, margin=dict(t=60, b=0, l=20, r=20))
        st.plotly_chart(g, use_container_width=True)
        st.caption(f"🕒 Simulated time: {r['time']:%d %b %H:%M}   |   💨 Aerator: {'ON' if r['aerator'] else 'OFF'}")
    with c2:
        m = st.columns(5)
        m[0].metric("Temp °C", f"{r['temp']:.1f}")
        m[1].metric("pH", f"{r['ph']:.2f}")
        m[2].metric("DO mg/L", f"{r['do']:.2f}")
        m[3].metric("Turbidity NTU", f"{r['turb']:.0f}")
        m[4].metric("NH₃ mg/L*", f"{r['nh3']:.3f}")
        st.caption("*NH₃ is a virtual (soft) sensor computed from TAN, pH and temperature.")
        st.subheader("Recommendations")
        for level, why, action in recs:
            box = {"critical": st.error, "warning": st.warning, "ok": st.success}[level]
            box(f"**{why}**\n\n➡️ {action}")

    # Dissolved-oxygen history + forecast
    h = df.tail(288)
    fig = go.Figure()
    fig.add_scatter(x=h["time"], y=h["do"], name="DO (measured)", line=dict(color="#0ea5e9"))
    if fc is not None:
        ft = [r["time"] + pd.Timedelta(minutes=5 * (i + 1)) for i in range(len(fc))]
        fig.add_scatter(x=ft, y=fc, name="DO forecast (6 h)", line=dict(color="#f97316", dash="dash"))
    fig.add_hline(y=prof["do_crit"], line_color="red", annotation_text="critical")
    fig.add_hline(y=prof["do_ok"], line_color="green", annotation_text="safe")
    fig.update_layout(height=320, title="Dissolved oxygen: last 24 h + 6 h forecast", yaxis_title="mg/L",
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Temperature", "pH", "Ammonia (NH₃)", "Turbidity"])
    for tab, col in zip(tabs, ["temp", "ph", "nh3", "turb"]):
        tab.line_chart(h.set_index("time")[col], height=200)

    st.subheader("Alert log (also sent to Telegram if enabled)")
    st.write("\n\n".join(st.session_state.alert_log[:6]) or "No alerts yet.")

live()
