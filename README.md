# GPS-denied swarm: ArUco localization, leapfrog coverage, mine mapping

Three drones search an arena for mines with GPS off. Each drone carries an ArUco marker on top
(no markers on the ground).

- **Joey (mother):** hovers at 6 m and looks down. She computes every drone's position from the
  markers and sends it to their autopilots. She runs the mission and the mine map.
- **Marky and DeeDee (daughters):** take turns. One sits landed as the position reference; the
  other scans at 3 m with the mine detector running. They then leapfrog forward.

## Run

```bash
./run_swarm.sh                                  # Gazebo, 3 SITL, 3 MAVROS, localizer, mine detection
python3 robofest/robofest/swarm_mission.py      # second terminal, when it says "Ready"
```

- The mother can need 1–2 minutes before her autopilot lets her arm.
- Ctrl-C in the `run_swarm.sh` terminal stops everything and saves the map to `mine_map/`.
- Options: `HEADLESS=1` (no windows, logs in `sitl/logs/`), `GPS=1` (simulated GPS instead),
  `MINES=0` (no mine detection), `DETECTOR=hybrid` (YOLO + colour detector),
  `python3 robofest/robofest/swarm_mission.py --ros-args -p stages:=1`.

## Contents

| Path | What |
|---|---|
| `run_swarm.sh` | starts the whole stack |
| `swarm.sdf`, `mines_truth.json` | the world (3 drones, 12 mines) and the true mine positions |
| `Joey/`, `DeeDee/`, `Marky/`, `rpi_cam_v2/` | drone and camera models (markers 10, 11, 12) |
| `gps_denied.parm`, `gps_denied_mother.parm`, `gps_denied_scanner.parm` | autopilot setup: no GPS, ArUco position + optical flow |
| `sim_flow.parm` | simulated optical flow and rangefinder (simulation only) |
| `config/` | MAVROS launch file with only the plugins the swarm uses |
| `robofest/robofest/aruco_localizer.py` | the mother's ArUco localizer |
| `robofest/robofest/swarm_mission.py` | the leapfrog mission |
| `minedet/` | mine detector (YOLO26n LiteRT) and mine map nodes |
