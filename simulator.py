import math
import numpy as np
import pandas as pd

START = pd.Timestamp("2026-01-01 00:00")


class PondTwin:
    DT_H = 5 / 60  # one step = 5 minutes, expressed in hours

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.t_min = 0                                   # simulated minutes since start
        self.temp, self.do, self.ph = 28.0, 6.5, 7.8     # true state of the pond
        self.tan, self.turb = 0.3, 30.0                  # TAN = total ammonia nitrogen (mg/L)
        self.scn = {"heatwave": False, "overfeed": False, "bloom": False, "sensor_fault": False}
        self.aerator = False
        self._stuck = None                               # frozen DO value when the sensor "fails"

    def set_scenarios(self, **flags):
        """Turn scenarios on/off, e.g. set_scenarios(heatwave=True)."""
        self.scn.update(flags)
        if not self.scn["sensor_fault"]:
            self._stuck = None

    def step(self):
        h = (self.t_min / 60) % 24
        dt = self.DT_H
        s = self.scn
        sun = max(0.0, math.sin(math.pi * (h - 6) / 12))   # 0 at night, 1 at noon
        bloom = 1.0 if s["bloom"] else 0.0
        feed = 3.0 if s["overfeed"] else 1.0

        # 1) Temperature drifts toward a daily wave (+5 C in a heatwave)
        target_t = 28 + 2 * math.sin(2 * math.pi * (h - 9) / 24) + (5 if s["heatwave"] else 0)
        self.temp += 0.08 * (target_t - self.temp)

        # 2) Dissolved oxygen = photosynthesis - respiration + exchange with air (+ aerator)
        sat = 14.6 - 0.39 * self.temp + 0.0049 * self.temp ** 2        # max DO the water can hold
        photo = (1.6 + 2.5 * bloom) * sun
        resp = (0.55 + 0.12 * feed + 0.9 * bloom) * (1 + 0.06 * (self.temp - 28))
        k = 0.30 + (1.5 if self.aerator else 0.0)
        self.do += dt * (photo - resp + k * (sat - self.do))
        self.do = min(max(self.do, 0.2), 14)

        # 3) Ammonia builds up from feed and is slowly removed
        self.tan += dt * (0.03 * feed - 0.09 * self.tan)

        # 4) pH rises in daylight (photosynthesis), more so in an algal bloom
        target_ph = 7.5 + 0.45 * sun + bloom * (0.9 * sun - 0.2)
        self.ph += 0.15 * (target_ph - self.ph)

        # 5) Turbidity rises with blooms and extra feed
        self.turb += 0.1 * (25 + 90 * bloom + 12 * (feed - 1) - self.turb)

        # Sensor readings = true value + noise
        n = self.rng.normal
        do_read = self.do + n(0, 0.05)
        if s["sensor_fault"]:                      # a broken probe repeats the same number forever
            if self._stuck is None:
                self._stuck = do_read
            do_read = self._stuck
        reading = {"time": START + pd.Timedelta(minutes=self.t_min),
                   "temp": self.temp + n(0, 0.05), "ph": self.ph + n(0, 0.02),
                   "do": do_read, "turb": self.turb + n(0, 1.5),
                   "tan": max(0, self.tan + n(0, 0.01)), "aerator": self.aerator}
        self.t_min += 5
        return reading
