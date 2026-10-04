import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Approximate tolerance ranges (tune with local guidance / CIFRI / NFDB advice)
PROFILES = {
    "Rohu (carp)":     dict(temp=(25, 32), ph=(6.8, 8.5), do_ok=5.0, do_crit=3.0, nh3_ok=0.02, nh3_crit=0.05),
    "Tilapia":         dict(temp=(27, 32), ph=(6.5, 8.5), do_ok=5.0, do_crit=3.0, nh3_ok=0.02, nh3_crit=0.05),
    "Vannamei shrimp": dict(temp=(26, 32), ph=(7.5, 8.5), do_ok=5.0, do_crit=3.0, nh3_ok=0.01, nh3_crit=0.03),
}
STEP_MIN, DAY = 5, 288   # 288 five-minute steps = 24 hours


def nh3_from(tan, temp, ph):
    """Virtual ammonia sensor: toxic NH3 fraction of TAN (Emerson et al. formula)."""
    pka = 0.09018 + 2729.92 / (temp + 273.15)
    return tan / (1 + 10 ** (pka - ph))


def enrich(r):
    r["nh3"] = nh3_from(r["tan"], r["temp"], r["ph"])
    return r


# ---------- Health score ----------
def _low(x, ok, crit):   # higher is better
    return 100 if x >= ok else 0 if x <= crit else 100 * (x - crit) / (ok - crit)

def _high(x, ok, crit):  # lower is better
    return 100 if x <= ok else 0 if x >= crit else 100 * (crit - x) / (crit - ok)

def _band(x, lo, hi, margin):
    d = 0 if lo <= x <= hi else (lo - x if x < lo else x - hi)
    return max(0, 100 * (1 - d / margin))

def health_score(r, p):
    subs = {"DO": _low(r["do"], p["do_ok"], p["do_crit"]),
            "NH3": _high(r["nh3"], p["nh3_ok"], p["nh3_crit"]),
            "Temp": _band(r["temp"], *p["temp"], 4),
            "pH": _band(r["ph"], *p["ph"], 1.5),
            "Turbidity": _high(r["turb"], 50, 120)}
    w = {"DO": .3, "NH3": .25, "Temp": .2, "pH": .15, "Turbidity": .1}
    avg = sum(subs[k] * w[k] for k in subs)
    score = 0.6 * avg + 0.4 * min(subs.values())   # one bad parameter drags the score down
    status = "GREEN" if score >= 75 else "AMBER" if score >= 50 else "RED"
    return round(score), status, subs


# ---------- Anomaly + sensor-fault detection ----------
class Detector:
    COLS = ["temp", "do", "ph", "turb", "tan"]

    def __init__(self, normal_df):
        self.model = IsolationForest(n_estimators=100, random_state=0).fit(normal_df[self.COLS])
        self.thr = np.percentile(self.model.score_samples(normal_df[self.COLS]), 0.5) - 0.02

    def is_anomaly(self, r):
        return self.model.score_samples(pd.DataFrame([r])[self.COLS])[0] < self.thr


def sensor_fault(df):
    """Stuck probe (no change for 1 hour) or impossible jump."""
    d = df["do"].to_numpy()
    if len(d) < 13:
        return None
    if d[-12:].max() - d[-12:].min() < 0.02:
        return "Dissolved-oxygen probe looks STUCK (value frozen for 1 hour)"
    if abs(d[-1] - d[-2]) > 1.5:
        return "Dissolved-oxygen reading jumped impossibly fast"
    return None


# ---------- Forecast ----------
def forecast_do(df, horizon=72):
    """Predict DO for the next 6 h: 'same time yesterday' + how far today is drifting from yesterday."""
    d = df["do"].to_numpy()
    n = len(d)
    if n < DAY + 20:
        return None
    off_now = np.mean(d[-3:] - d[-DAY - 3:-DAY])
    off_1h = np.mean(d[-15:-12] - d[-DAY - 15:-DAY - 12])
    trend = (off_now - off_1h) / 12                       # drift per step
    k = np.arange(1, horizon + 1)
    base = d[n - 1 + k - DAY]
    offs = (off_now + trend * np.minimum(k, 36)) * np.exp(-k * STEP_MIN / 60 / 8)
    return np.clip(base + offs, 0, 14)


def risk_from_forecast(fc, p):
    warn = (p["do_ok"] + p["do_crit"]) / 2
    if fc is None:
        return None
    idx = np.where(fc < warn)[0]
    return {"min_do": float(fc.min()), "warn_level": warn,
            "lead_h": (idx[0] + 1) * STEP_MIN / 60 if len(idx) else None}


# ---------- Advice + auto-action ----------
def advise(r, p, risk, fault, anomaly, clock):
    out = []
    if fault:
        out.append(("warning", fault, "Clean/check the probe and verify with a hand-held kit. Do NOT trust this sensor yet."))
    if r["do"] < p["do_crit"] and not fault:
        out.append(("critical", f"Dissolved oxygen {r['do']:.1f} mg/L is below the critical limit {p['do_crit']}",
                    "Switch ON aerators now and stop feeding."))
    if risk and risk["lead_h"] is not None:
        out.append(("warning", f"AI expects oxygen below {risk['warn_level']:.1f} mg/L in ~{risk['lead_h']:.1f} h (lowest ~{risk['min_do']:.1f})",
                    "Start the aerator before then and cut today's feed by ~30%."))
    if r["nh3"] > p["nh3_crit"]:
        out.append(("critical", f"Toxic ammonia (NH3) {r['nh3']:.3f} mg/L - pH {r['ph']:.1f} and {r['temp']:.1f} C make it worse",
                    "Stop feeding, exchange 20-30% water, run the aerator."))
    elif r["nh3"] > p["nh3_ok"]:
        out.append(("warning", f"Ammonia (NH3) rising: {r['nh3']:.3f} mg/L", "Reduce feed and watch pH."))
    lo, hi = p["temp"]
    if not lo <= r["temp"] <= hi:
        out.append(("warning", f"Temperature {r['temp']:.1f} C outside {lo}-{hi} C", "Add shade/fresh water; feed at cooler hours."))
    lo, hi = p["ph"]
    if not lo <= r["ph"] <= hi:
        out.append(("warning", f"pH {r['ph']:.2f} outside {lo}-{hi}", "Check for algal bloom; consider partial water exchange."))
    if r["turb"] > 70:
        out.append(("warning", f"Turbidity {r['turb']:.0f} NTU is high - possible algal bloom",
                    "Expect a night-time oxygen crash; keep the aerator ready."))
    if anomaly and not out:
        out.append(("warning", "AI flagged an unusual combination of readings", "Inspect the pond."))
    return out or [("ok", "All parameters are in the safe range", "No action needed.")]


def aerator_decision(r, p, risk, is_on):
    warn = (p["do_ok"] + p["do_crit"]) / 2
    lead = risk["lead_h"] if risk else None
    if r["do"] < warn or (lead is not None and lead < 3):
        return True
    if is_on and r["do"] > p["do_ok"] + 1.5 and lead is None:
        return False
    return is_on
