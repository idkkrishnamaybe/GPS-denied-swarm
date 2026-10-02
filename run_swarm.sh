#!/bin/bash
# Swarm simulation with the ArUco "GPS" (see ARUCO.md). All markers are on the drones:
#   Joey   - mother  (ID 10): SITL instance 0, MAVROS /Joey, runs the localizer + mission
#   Marky  - daughter (ID 12): SITL instance 2, MAVROS /Marky; first landed reference
#   DeeDee - daughter (ID 11): SITL instance 1, MAVROS /DeeDee; first scanner
# The daughters take turns: one scans while the other is the landed reference (leapfrog).
#
#   ./run_swarm.sh          GPS off: the autopilots navigate on the ArUco poses fused with
#                           optical flow (gps_denied.parm + sim_flow.parm); then start
#                           robofest/robofest/swarm_mission.py
#   GPS=1 ./run_swarm.sh    simulated GPS; ArUco poses are only published alongside
#
# Mine detection (MINES=0 turns it off): both daughters' cameras run the YOLO mine
# detector from ./minedet (MINEDET=...), every detection is
# projected onto the ground in the arena frame, and the mine map (1 m exclusion zones,
# occupancy grid) is shown in a second rqt_image_view and saved to ./mine_map/.
# DETECTOR=hybrid adds the classical colour/shape detector; DETECTOR=color uses only it.
#
# WORLD=other.sdf TRUTH=other_mines.json: another world
#
# HEADLESS=1: no windows (Gazebo server only, no rqt); SITL, MAVROS and the nodes run
# in the background with their output in sitl/logs/. For remote machines and tests.
set -e
cd "$(dirname "$0")"
HEADLESS=${HEADLESS:-0}
WORLD=${WORLD:-swarm.sdf}
TRUTH=${TRUTH:-mines_truth.json}
LOGS=$PWD/sitl/logs
mkdir -p "$LOGS"

# term TITLE COMMAND: a terminal window, or (HEADLESS=1) a background job logging
# to sitl/logs/TITLE.log. MAVProxy runs non-interactive there (no console to read).
term() {
  local title=$1 cmd=$2
  if [ "$HEADLESS" = "1" ]; then
    local log="$LOGS/$(echo "$title" | tr -c 'A-Za-z0-9\n' '_' | tr -s '_' | sed 's/_$//').log"
    bash -c "${cmd//sim_vehicle.py /sim_vehicle.py --mavproxy-args=--non-interactive }" > "$log" 2>&1 < /dev/null &
  else
    gnome-terminal --title="$title" -- bash -c "$cmd; exec bash" &
  fi
}

# window COMMAND...: a GUI tool, skipped when HEADLESS=1
window() {
  [ "$HEADLESS" = "1" ] || "$@" &
}

cleanup() {
  echo ""
  echo "==== Stopping swarm... ===="
  # the xterms this script opened (their shells would otherwise stay open)
  pkill -f "xterm -T (SITL|MAVROS) - " || true
  pkill -INT -f "ros2 launch .*mavros" || true
  pkill -f "aruco_localizer" || true
  pkill -INT -f "mine_map_node" || true          # SIGINT: it saves the map on the way out
  pkill -f "mine_detector_node" || true
  pkill -f "rqt_image_view" || true
  pkill -f "parameter_bridge" || true
  pkill -f "sim_vehicle.py" || true
  pkill -f "arducopter" || true
  pkill -f "mavproxy" || true
  kill $GZ_PID 2>/dev/null || true
  sleep 2
  # whatever ignored the polite signals (rqt_image_view ignores SIGTERM)
  pkill -9 -f "mavros_node|rqt_image_view|aruco_localizer|mine_detector_node|mine_map_node|parameter_bridge|arducopter|mavproxy" || true
  echo "Done!"
  exit 0
}
trap cleanup SIGINT SIGTERM

export GZ_IP=127.0.0.1
# the drone and camera models in this folder (model://Joey, model://rpi_cam_v2, ...)
export GZ_SIM_RESOURCE_PATH="$PWD${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
CAM=/world/iris_runway/model/Joey/model/gimbal/link/pitch_link/sensor/camera
if [ "${GPS:-0}" = "1" ]; then
  echo "== simulated GPS mode =="
  SITL_PARAMS_JOEY=""; SITL_PARAMS_DEEDEE=""; SITL_PARAMS_MARKY=""; PUBLISH_MAVROS=false
else
  echo "== GPS-denied mode: autopilots navigate on the ArUco poses =="
  # sim_flow.parm: simulated optical flow + rangefinder (fused with the ArUco position)
  SITL_PARAMS_JOEY="--add-param-file=$PWD/gps_denied.parm --add-param-file=$PWD/gps_denied_mother.parm --add-param-file=$PWD/sim_flow.parm"
  SITL_PARAMS_DEEDEE="--add-param-file=$PWD/gps_denied.parm --add-param-file=$PWD/gps_denied_scanner.parm --add-param-file=$PWD/sim_flow.parm"
  SITL_PARAMS_MARKY="$SITL_PARAMS_DEEDEE"
  PUBLISH_MAVROS=true
fi

# NVIDIA OpenGL (PRIME offload) only when the GPU actually works; otherwise Mesa on
# /dev/dri (e.g. the AMD iGPU). Forcing the NVIDIA GLX vendor without a working NVIDIA
# GPU makes the Gazebo GUI abort ("Failed to create OpenGL context").
GL_ENV=()
if nvidia-smi -L > /dev/null 2>&1; then
  GL_ENV=(__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia)
  echo "-> Gazebo ($WORLD) on the NVIDIA GPU"
else
  unset __NV_PRIME_RENDER_OFFLOAD __GLX_VENDOR_LIBRARY_NAME
  echo "-> Gazebo ($WORLD) with Mesa (no working NVIDIA GPU)"
fi
if [ "$HEADLESS" = "1" ]; then
  env "${GL_ENV[@]}" gz sim -s -r --headless-rendering "$WORLD" > "$LOGS/gazebo.log" 2>&1 &
else
  env "${GL_ENV[@]}" gz sim -r "$WORLD" &
fi
GZ_PID=$!
sleep 10

echo "-> ArduCopter SITL: Joey (instance 0), DeeDee (instance 1), Marky (instance 2)"
# each SITL keeps its logs in its own folder; -w resets saved parameters so the
# parameter files above always apply
mkdir -p sitl/Joey sitl/DeeDee sitl/Marky
term "SITL - Joey (mother)" \
  "cd sitl/Joey && sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON -I0 --sysid 1 -N -w $SITL_PARAMS_JOEY --out=udp:127.0.0.1:14555"
term "SITL - DeeDee (scanner)" \
  "cd sitl/DeeDee && sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON -I1 --sysid 2 -N -w $SITL_PARAMS_DEEDEE --out=udp:127.0.0.1:14565"
term "SITL - Marky (daughter)" \
  "cd sitl/Marky && sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON -I2 --sysid 3 -N -w $SITL_PARAMS_MARKY --out=udp:127.0.0.1:14575"
echo "   waiting 18 s for SITL..."
sleep 18

echo "-> MAVROS: /Joey/mavros, /DeeDee/mavros, /Marky/mavros"
# apm.launch's settings, but only the plugins the swarm uses (config/mavros_pluginlists.yaml);
# MAVROS_PLUGINS=all loads the stock set
MAVROS="ros2 launch $PWD/config/mavros_swarm.launch"
[ "${MAVROS_PLUGINS:-}" = "all" ] && \
  MAVROS="$MAVROS pluginlists_yaml:=$(ros2 pkg prefix mavros)/share/mavros/launch/apm_pluginlists.yaml"
term "MAVROS - Joey" "$MAVROS fcu_url:=udp://127.0.0.1:14555@ tgt_system:=1 namespace:=Joey"
term "MAVROS - DeeDee" "$MAVROS fcu_url:=udp://127.0.0.1:14565@ tgt_system:=2 namespace:=DeeDee"
term "MAVROS - Marky" "$MAVROS fcu_url:=udp://127.0.0.1:14575@ tgt_system:=3 namespace:=Marky"
sleep 5

echo "-> ArUco localizer on Joey's camera"
# /clock: the mission runs on sim time (the simulator is often slower than real time)
ros2 run ros_gz_bridge parameter_bridge \
  "$CAM/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "$CAM/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
  "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock" > "$LOGS/bridge.log" 2>&1 &
/usr/bin/python3 -u robofest/robofest/aruco_localizer.py --ros-args -p publish_mavros:=$PUBLISH_MAVROS \
  2>&1 | tee "$LOGS/aruco_localizer.log" &
sleep 3
window ros2 run rqt_image_view rqt_image_view /aruco/debug_image

MINEDET=${MINEDET:-$PWD/minedet}
if [ "${MINES:-1}" = "1" ] && [ -f "$MINEDET/ros2/mine_detector_node.py" ]; then
  echo "-> Mine detection on DeeDee's and Marky's cameras, mine map -> $PWD/mine_map"
  DCAM=/world/iris_runway/model/NAME/model/gimbal/link/pitch_link/sensor/camera
  BRIDGE=()
  for n in DeeDee Marky; do
    BRIDGE+=("${DCAM/NAME/$n}/image@sensor_msgs/msg/Image[gz.msgs.Image"
             "${DCAM/NAME/$n}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo")
  done
  ros2 run ros_gz_bridge parameter_bridge "${BRIDGE[@]}" --ros-args -r __node:=mine_camera_bridge \
    > "$LOGS/bridge_mines.log" 2>&1 &
  for n in DeeDee Marky; do
    python3 -u "$MINEDET/ros2/mine_detector_node.py" --ros-args -r __node:=mine_detector_$n \
      -p drone:=$n -p detector:=${DETECTOR:-yolo} > "$LOGS/mine_detector_$n.log" 2>&1 &
  done
  TRUTH_ARG=(); [ -f "$TRUTH" ] && TRUTH_ARG=(-p truth_file:=$PWD/$TRUTH)
  python3 -u "$MINEDET/ros2/mine_map_node.py" --ros-args -p drones:="['DeeDee','Marky']" \
    -p out_dir:=$PWD/mine_map "${TRUTH_ARG[@]}" 2>&1 | tee "$LOGS/mine_map.log" &
  sleep 2
  window ros2 run rqt_image_view rqt_image_view /mines/map_image
elif [ "${MINES:-1}" = "1" ]; then
  echo "   (no mine detection: $MINEDET/ros2/mine_detector_node.py not found; set MINEDET=...)"
fi

echo "========================================================"
if [ "${GPS:-0}" = "1" ]; then
  echo " Ready.  Simulated GPS: fly the drones over MAVROS yourself"
else
  echo " Ready.  Fly:    python3 robofest/robofest/swarm_mission.py   (leapfrog coverage)"
  echo "         (the mother may need 1-2 min before the autopilot lets her arm)"
fi
echo "         Poses:  ros2 topic echo /DeeDee/aruco/pose"
echo "         Mines:  mine_map/mines.csv (live: rqt /mines/map_image, /DeeDee/mines/debug_image)"
echo "         Logs:   $LOGS"
echo "         Ctrl-C here stops everything."
echo "========================================================"
wait
