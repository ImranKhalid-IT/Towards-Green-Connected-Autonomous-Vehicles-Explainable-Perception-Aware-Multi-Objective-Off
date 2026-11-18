"""OPTIONAL CARLA 0.9.15 + SUMO 1.19 bridge (not needed by the simulator).

This module connects the GreenCAV decision loop to a running CARLA server and a
SUMO/TraCI traffic simulation, so that the simulated exogenous processes
(weather, traffic density, camera frames) can be replaced by co-simulation
outputs.  It was written against the public CARLA 0.9.15 and SUMO TraCI APIs
but could not be executed in the environment where this package was built
(no CARLA/SUMO installation); test it on your setup before use.

Typical use (CARLA server on localhost:2000, SUMO config from CARLA's
Co-Simulation/Sumo examples):

    bridge = CarlaSumoBridge(sumo_cfg="Town04.sumocfg")
    bridge.set_weather("heavy_rain")
    frame = bridge.next_frame()            # HxWx3 uint8 camera image
    density = bridge.vehicle_density()     # vehicles per km of monitored road
"""
from __future__ import annotations

import queue
from typing import Optional

import numpy as np

# CARLA weather presets for the four weather states of Eq. (2)
WEATHER_PRESETS = {
    "clear": dict(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0, fog_density=0.0, wetness=0.0),
    "light_rain": dict(cloudiness=60.0, precipitation=30.0, precipitation_deposits=20.0, fog_density=5.0, wetness=30.0),
    "heavy_rain": dict(cloudiness=90.0, precipitation=90.0, precipitation_deposits=80.0, fog_density=15.0, wetness=90.0),
    "dense_smog": dict(cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, fog_density=85.0,
                       fog_distance=5.0, wetness=10.0),
}


class CarlaSumoBridge:
    def __init__(self, host: str = "localhost", port: int = 2000, sumo_cfg: Optional[str] = None,
                 image_size=(640, 640), fixed_delta: float = 0.1, road_length_km: float = 1.0):
        import carla  # type: ignore

        self.carla = carla
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta          # 10 Hz = f_s
        self.world.apply_settings(settings)
        bp_lib = self.world.get_blueprint_library()
        spawn = self.world.get_map().get_spawn_points()[0]
        self.ego = self.world.spawn_actor(bp_lib.filter("vehicle.tesla.model3")[0], spawn)
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(image_size[0]))
        cam_bp.set_attribute("image_size_y", str(image_size[1]))
        self.cam = self.world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=self.ego)
        self._frames: "queue.Queue" = queue.Queue()
        self.cam.listen(self._frames.put)
        self.road_length_km = road_length_km
        self.traci = None
        if sumo_cfg:
            import traci  # type: ignore
            traci.start(["sumo", "-c", sumo_cfg, "--step-length", str(fixed_delta)])
            self.traci = traci

    def set_weather(self, name: str) -> None:
        self.world.set_weather(self.carla.WeatherParameters(**WEATHER_PRESETS[name]))

    def next_frame(self) -> np.ndarray:
        self.world.tick()
        if self.traci is not None:
            self.traci.simulationStep()
        img = self._frames.get(timeout=5.0)
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        return arr[:, :, :3][:, :, ::-1].copy()            # BGRA -> RGB

    def vehicle_density(self) -> float:
        """Vehicles per km (Supplementary Table S4 regimes: 10-20, 30-50, 70-100, 120+)."""
        if self.traci is None:
            n = len(self.world.get_actors().filter("vehicle.*"))
        else:
            n = self.traci.vehicle.getIDCount()
        return n / self.road_length_km

    def close(self) -> None:
        try:
            self.cam.stop(); self.cam.destroy(); self.ego.destroy()
        finally:
            if self.traci is not None:
                self.traci.close()


def density_to_regime(veh_per_km: float) -> str:
    if veh_per_km < 25:
        return "sparse"
    if veh_per_km < 60:
        return "moderate"
    if veh_per_km < 110:
        return "dense"
    return "ultra_dense"
