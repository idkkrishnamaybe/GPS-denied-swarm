#!/usr/bin/env python3
"""GPS-denied leapfrog coverage, using only the ArUco markers on the drones.

Runs on the mother, like aruco_localizer: she is the only drone that estimates
position and does all the communication. The daughters' autopilots are reached
over her link (their MAVROS instances) and fly only on what she sends.

Needs the autopilots set up by gps_denied.parm (+ gps_denied_mother.parm /
gps_denied_scanner.parm, + sim_flow.parm in simulation) and aruco_localizer
running with publish_mavros:=true.

The mother hovers still at `mother_alt` while one daughter scans the ground she
sees and the other sits landed as the position reference. Then they leapfrog:

  1. The scanning daughter takes off at the trailing edge of the mother's view,
     climbs inwards to `scan_alt` and flies two scan lines, then lands at the
     leading edge. The localizer freezes its landed pose: a new landmark.
  2. The other daughter (the old reference) is released, takes off and lands
     next to it.
  3. The mother moves forward by `step`, so both landed daughters end up at the
     trailing edge of her view, and the daughter that just moved scans the new
     area. Repeat for `stages` stages.

Every path is planned in the mother's frame (forward, left) around the point
straight below her, and keeps a daughter
  - off her nadir (at least 1 m away): there her downward lidar and flow sensor
    would see the daughter instead of the ground, and her downwash hits it;
  - inside her camera view: at 3 m below, about +-1.3 m forward x +-1.8 m
    sideways; a landed marker within +-2.6 m x +-3.5 m;
  - from covering a landed reference in her image.

Positions: with EK3_SRC1_POSXY/YAW = ExternalNav, every drone's local frame is
the arena frame (x/y). Height: the mother uses her barometer (0 = where she
booted); the daughters use the height the mother measures (arena z).

Also:
  - The mother climbs level (attitude target, no position target) until her first
    ArUco fix: that fix moves her EKF into the arena frame, and a position target
    set before it would be metres off.
  - Landing spots keep `mine_clearance` from the mines in /mines/map (mine_map_node);
    before touching down a daughter hovers `landing_check_alt` above the spot for
    `landing_check_time` so her own camera can check it, and moves if it sees a mine.
  - In simulation (a /clock publisher exists) the mission runs on sim time: the
    simulator is often slower than real time.
"""
import json
import math
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from geographic_msgs.msg import GeoPointStamped
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import AttitudeTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, MessageInterval, SetMode
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Drone:
    """MAVROS handles, latest data and the current setpoint for one vehicle."""

    def __init__(self, node, name, marker_id=None):
        self.node, self.name, self.id = node, name, marker_id
        self.state = State()
        self.local = None          # (time, PoseStamped) from the EKF
        self.aruco = None          # (time, PoseStamped) from the localizer
        self.yaw = None
        self.sp = None             # (x, y, z, yaw) streamed while armed in GUIDED
        self.climb_to = None       # instead of sp: climb to this height, level, not turning
        ns = f'/{name}/mavros'
        node.create_subscription(State, f'{ns}/state', self.on_state, 10)
        node.create_subscription(PoseStamped, f'{ns}/local_position/pose', self.on_local, qos_profile_sensor_data)
        node.create_subscription(PoseStamped, f'/{name}/aruco/pose', self.on_aruco, 10)
        node.create_subscription(Imu, f'{ns}/imu/data', self.on_imu, qos_profile_sensor_data)
        self.origin_pub = node.create_publisher(GeoPointStamped, f'{ns}/global_position/set_gp_origin', 10)
        self.pos_pub = node.create_publisher(PoseStamped, f'{ns}/setpoint_position/local', 10)
        self.att_pub = node.create_publisher(AttitudeTarget, f'{ns}/setpoint_raw/attitude', 10)
        self.mode_cli = node.create_client(SetMode, f'{ns}/set_mode')
        self.arm_cli = node.create_client(CommandBool, f'{ns}/cmd/arming')
        self.tko_cli = node.create_client(CommandTOL, f'{ns}/cmd/takeoff')
        self.rate_cli = node.create_client(MessageInterval, f'{ns}/set_message_interval')
        self.raw_param_cli = node.create_client(SetParameters, f'{ns}/setpoint_raw/set_parameters')

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def on_state(self, m):
        self.state = m

    def on_local(self, m):
        self.local = (self.now(), m)

    def on_aruco(self, m):
        self.aruco = (self.now(), m)

    def on_imu(self, m):
        self.yaw = yaw_of(m.orientation)

    def fresh(self, item, max_age=0.6):
        return item is not None and self.now() - item[0] < max_age

    def ekf_uses_aruco(self):
        """EKF position is live and agrees (x/y) with the latest ArUco fix."""
        if not (self.fresh(self.local, 1.5) and self.fresh(self.aruco)):
            return False
        a, l = self.aruco[1].pose.position, self.local[1].pose.position
        return math.hypot(a.x - l.x, a.y - l.y) < 0.3

    def position(self):
        p = self.local[1].pose.position
        return p.x, p.y, p.z

    def request(self, client, req):
        if client.service_is_ready():
            client.call_async(req)

    def keep_rates(self):
        """LOCAL_POSITION_NED at 10 Hz, ATTITUDE_QUATERNION at 20 Hz (default telemetry
        is ~2 Hz and MAVProxy / ground stations can reset it, so this is re-sent)."""
        for msg_id, hz in ((32, 10.0), (31, 20.0)):
            self.request(self.rate_cli, MessageInterval.Request(message_id=msg_id, message_rate=hz))

    def send_origin(self, lat, lon, alt):
        m = GeoPointStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.position.latitude, m.position.longitude, m.position.altitude = lat, lon, alt
        self.origin_pub.publish(m)

    def set_mode(self, mode):
        self.request(self.mode_cli, SetMode.Request(custom_mode=mode))

    def arm(self):
        self.request(self.arm_cli, CommandBool.Request(value=True))

    def takeoff(self, alt):
        self.request(self.tko_cli, CommandTOL.Request(altitude=float(alt)))

    def enable_thrust(self):
        """MAVROS drops attitude setpoints that carry thrust until setpoint_raw's
        thrust_scaling is set (not in apm_config.yaml). 1.0 passes our thrust through
        unchanged; ArduCopter reads it as a climb rate. -> future (None if not ready)."""
        if not self.raw_param_cli.service_is_ready():
            return None
        value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=1.0)
        req = SetParameters.Request(parameters=[ParameterMsg(name='thrust_scaling', value=value)])
        return self.raw_param_cli.call_async(req)

    def send_climb(self, max_up=0.8, max_down=0.4, wp_speed_up=2.5):
        """GUIDED attitude target: level, heading = the latest measured one, thrust = climb
        rate (0.5 = hold; ArduCopter scales by WPNAV_SPEED_UP/DN, default 2.5 m/s up).
        No position or velocity target, and the heading target follows the measured heading
        every tick, so when the first ArUco fix moves the EKF into the arena frame (metres
        and degrees) there is nothing stale for the autopilot to chase. ArduCopter 4.7 wants
        all three body rates given or all ignored (else it just holds position)."""
        z = self.position()[2] if self.fresh(self.local, 1.5) else self.climb_to
        vz = min(max_up, max(-max_down, 1.0 * (self.climb_to - z)))
        yaw = self.yaw or 0.0
        m = AttitudeTarget()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | AttitudeTarget.IGNORE_PITCH_RATE | \
            AttitudeTarget.IGNORE_YAW_RATE
        m.orientation.z, m.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)   # level, ENU heading
        m.thrust = float(0.5 + 0.5 * vz / wp_speed_up)
        self.att_pub.publish(m)

    def send_setpoint(self):
        x, y, z, yaw = self.sp
        m = PoseStamped()
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.header.frame_id = 'map'
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pos_pub.publish(m)


class LeapfrogMission(Node):
    def __init__(self):
        super().__init__('swarm_mission')
        self.use_sim_clock()
        p = self.declare_parameter
        self.mother = Drone(self, p('mother', 'Joey').value)
        names = p('daughters', ['Marky', 'DeeDee']).value     # first one starts as the reference
        ids = p('daughter_ids', [12, 11]).value
        self.daughters = [Drone(self, n, i) for n, i in zip(names, ids)]
        self.mother_alt = p('mother_alt', 6.0).value
        self.scan_alt = p('scan_alt', 3.0).value
        self.stages = p('stages', 2).value
        # Layout in the mother's frame (metres from the point below her).
        self.edge = p('edge', 2.7).value              # landing spots: this far to the side
        self.spot_fwd = p('spot_forward', 0.6).value  # the two spots at +-this forward
        self.line_fwd = p('scan_line_forward', 1.0).value   # scan lines at +-this forward
        self.line_side = p('scan_side', 1.3).value    # scan lines span +-this sideways
        self.step = p('step', 5.4).value              # mother's move per stage (to her right)
        self.takeoff_alt = p('daughter_takeoff_alt', 1.0).value
        self.mother_takeoff_alt = p('mother_takeoff_alt', 1.5).value   # below any marker view
        # Landing: hover over the spot at check_alt while the daughter's own camera looks at the
        # touchdown area (the scan before only reaches the spot's inner half), then land.
        self.check_alt = p('landing_check_alt', 1.0).value      # still inside the mother's view
        self.check_time = p('landing_check_time', 2.0).value
        self.landing_clearance = p('landing_clearance', 0.5).value   # drone + mine radius + margin
        self.scan_speed = p('scan_speed', 0.4).value
        self.transit_speed = p('transit_speed', 0.5).value
        self.mother_speed = p('mother_speed', 0.4).value
        self.origin = p('ekf_origin', [-35.363262, 149.165237, 584.0]).value

        self.all = [self.mother] + self.daughters
        self.landmarks = [self.daughters[0].id]       # what the localizer should use
        self.active = []                              # what it reports it uses
        self.lm_pub = self.create_publisher(String, '/aruco/landmarks', 10)
        self.create_subscription(String, '/aruco/landmarks_active', self.on_active, 10)
        # Mines found so far (mine_map_node): a daughter never lands on or next to one.
        self.mine_clearance = p('mine_clearance', 1.45).value   # 1 m exclusion + mine + drone radius
        self.max_spot_shift = p('max_spot_shift', 1.2).value    # how far a landing spot may move
        self.mines = []
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, p('mine_map_topic', '/mines/map').value, self.on_mines, latched)
        self.station = None                           # mother's hover point (arena x, y)
        self.psi = None                               # mother's heading, held
        self.last = {}
        self.task = self.run()
        self.create_timer(0.05, self.tick)
        self.log(f'leapfrog: mother {self.mother.name} at {self.mother_alt} m, daughters '
                 f'{[d.name for d in self.daughters]} scanning at {self.scan_alt} m, {self.stages} stages')

    # ------------------------------------------------------------------ plumbing
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def use_sim_clock(self, wait=3.0):
        """In simulation, run on Gazebo's clock (bridged to /clock by run_swarm.sh). The
        simulator often runs slower than real time; on the wall clock the setpoint ramps
        would then be 2-3x too fast for the drones and every timeout too short."""
        if self.get_parameter('use_sim_time').value:
            return
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if self.count_publishers('/clock') > 0:
                self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
                self.get_logger().info('simulation clock (/clock) found: the mission runs on sim time')
                return
            time.sleep(0.1)

    def log(self, text):
        self.get_logger().info(text)

    def every(self, key, period):
        if self.now() - self.last.get(key, -1e9) >= period:
            self.last[key] = self.now()
            return True
        return False

    def on_mines(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        # confirmed mines, plus tentative ones seen twice: landing on either would be bad
        self.mines = [(m['x'], m['y']) for m in data.get('mines', [])] + \
                     [(m['x'], m['y']) for m in data.get('tentative', []) if m.get('hits', 0) >= 2]

    def on_active(self, msg):
        self.active = [int(v) for v in msg.data.split(',') if v]

    def tick(self):
        if self.every('rates', 2.0):
            for d in self.all:
                if d.state.connected:
                    d.keep_rates()
        if self.every('landmarks', 0.5):
            self.lm_pub.publish(String(data=','.join(str(i) for i in self.landmarks)))
        for d in self.all:
            if d.state.armed and d.state.mode == 'GUIDED':
                if d.climb_to is not None:
                    d.send_climb()
                elif d.sp is not None:
                    d.send_setpoint()
        if self.task is not None:
            try:
                next(self.task)
            except StopIteration:
                self.task = None
            except RuntimeError as e:        # a step failed: stop here, airborne drones keep holding
                self.get_logger().error(f'mission stopped: {e}')
                self.task = None

    # --------------------------------------------------------------- geometry
    def pt(self, fwd, left):
        """Arena x, y of a point given in the mother's frame around her station."""
        c, s = math.cos(self.psi), math.sin(self.psi)
        return self.station[0] + fwd * c - left * s, self.station[1] + fwd * s + left * c

    def landing_spot(self, side, keep_from=None):
        """Landing spot at the leading edge (forward = side * spot_fwd, left = -edge), moved if
        needed to keep `mine_clearance` from every mine found so far and 1 m from keep_from
        (the other landed daughter). It stays on its side of the mother and inside her view
        after she moves on. Returns the arena (x, y)."""
        nominal = (side * self.spot_fwd, -self.edge)
        cands = []
        for df in np.round(np.arange(-0.6, self.max_spot_shift + 1e-6, 0.15), 2):
            for dl in (0.0, 0.15, 0.3):       # only inwards: further out, the mother loses
                fwd, left = side * (self.spot_fwd + df), -self.edge + dl   # it during the 1 m check
                if side * fwd < 0.3 or abs(fwd) > 2.0:        # own side, inside the view
                    continue
                cands.append((math.hypot(df, dl), fwd, left))
        best = None
        for shift, fwd, left in sorted(cands):
            x, y = self.pt(fwd, left)
            if keep_from is not None and math.dist((x, y), keep_from) < 1.0:
                continue
            clear = min((math.dist((x, y), mn) for mn in self.mines), default=float('inf'))
            if clear >= self.mine_clearance:
                best = (shift, fwd, left, clear)
                break
            if best is None or clear > best[3]:
                best = (shift, fwd, left, clear)
        shift, fwd, left, clear = best
        spot = tuple(float(v) for v in self.pt(fwd, left))
        if shift > 0.01:
            self.log(f'landing spot moved {shift:.2f} m to ({spot[0]:+.2f}, {spot[1]:+.2f}): '
                     f'{clear:.2f} m from the nearest mine')
        if clear < self.mine_clearance:
            self.get_logger().warn(f'no landing spot keeps {self.mine_clearance} m from all mines; '
                                   f'using ({spot[0]:+.2f}, {spot[1]:+.2f}), {clear:.2f} m from one')
        return spot

    def enable_thrust(self, d, timeout=30.0):
        t0, fut = self.now(), None
        while fut is None or not fut.done():
            if self.now() - t0 > timeout:
                raise RuntimeError(f'could not set thrust_scaling on {d.name}/mavros/setpoint_raw')
            if fut is None:
                fut = d.enable_thrust()
            yield
        res = fut.result()
        if res is None or not all(r.successful for r in res.results):
            raise RuntimeError(f'{d.name}/mavros/setpoint_raw refused thrust_scaling')
        self.log(f'{d.name}: MAVROS passes attitude thrust through (thrust_scaling 1.0)')

    def land_safely(self, d, side, keep_from=None, tries=3):
        """Fly to a landing spot, look at it from check_alt and land only if the daughter's
        camera saw no mine within landing_clearance of it; otherwise pick another spot."""
        spot = None
        for attempt in range(tries):
            spot = self.landing_spot(side, keep_from)
            yield from self.move(d, *spot, self.check_alt, self.transit_speed)
            yield from self.settle(d, *spot)
            yield from self.sleep(self.check_time)
            near = min((math.dist(spot, mn) for mn in self.mines), default=float('inf'))
            if near >= self.landing_clearance:
                break
            self.get_logger().warn(f'{d.name}: mine {near:.2f} m from the landing spot, choosing another')
        else:
            self.get_logger().warn(f'{d.name}: no mine-free spot after {tries} tries, landing at the last one')
        yield from self.land(d, spot)

    # ------------------------------------------------------------ small tasks
    def wait(self, cond, what, timeout=None):
        t0 = self.now()
        while not cond():
            if timeout is not None and self.now() - t0 > timeout:
                raise RuntimeError(f'timed out waiting for {what}')
            yield

    def sleep(self, seconds):
        t0 = self.now()
        while self.now() - t0 < seconds:
            yield

    def set_landmarks(self, ids):
        """Tell the localizer which landed drones to use, and wait until it does."""
        self.landmarks = list(ids)
        names = {d.id: d.name for d in self.daughters}
        self.log(f'landmarks -> {[names[i] for i in ids]}')
        yield from self.wait(lambda: self.active == self.landmarks, 'the localizer to switch landmarks', 60)

    def arm_in(self, d, mode, timeout=180.0):
        t0 = self.now()
        while not (d.state.armed and d.state.mode == mode):
            if self.now() - t0 > timeout:
                raise RuntimeError(f'{d.name} did not arm in {mode} within {timeout:.0f} s (see its autopilot '
                                   f'messages, e.g. "VisOdom: not healthy" = the mother cannot see its marker)')
            if self.every(f'{d.name}_arm', 2.0):     # IMUs can need a minute or two after boot
                if d.state.mode != mode:
                    d.set_mode(mode)
                else:
                    d.arm()
            yield

    def move(self, d, x, y, z, speed, tol=0.2):
        """Ramp the setpoint in a straight line to (x, y, z), then wait to get there."""
        yaw = d.sp[3]
        while True:
            cx, cy, cz, _ = d.sp
            dx, dy, dz = x - cx, y - cy, z - cz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            step = speed * 0.05
            if dist <= step:
                d.sp = (x, y, z, yaw)
                break
            d.sp = (cx + dx / dist * step, cy + dy / dist * step, cz + dz / dist * step, yaw)
            yield
        yield from self.wait(lambda: d.fresh(d.local, 1.5) and
                             math.dist(d.position()[:2], (x, y)) < tol and abs(d.position()[2] - z) < 0.3,
                             f'{d.name} to reach its waypoint', 60)

    def settle(self, d, x, y, tol=0.1, hold=1.5, timeout=30.0):
        """Wait until d has stayed within tol of (x, y) for `hold` seconds, i.e. it has stopped
        there. LAND holds the position where it is engaged, so a drone still drifting
        towards its spot would otherwise land short of it or slide past it."""
        t0, since = self.now(), None
        while True:
            here = d.fresh(d.local, 1.5) and math.dist(d.position()[:2], (x, y)) < tol
            since = (since or self.now()) if here else None
            if since is not None and self.now() - since >= hold:
                return
            if self.now() - t0 > timeout:
                self.log(f'{d.name} did not settle within {tol} m of ({x:+.2f}, {y:+.2f}); landing anyway')
                return
            yield

    def daughter_takeoff(self, d):
        yield from self.wait(d.ekf_uses_aruco, f'{d.name} to have an ArUco fix', 120)
        yield from self.arm_in(d, 'GUIDED')
        d.sp = None                  # no position targets yet: they would cancel the takeoff on the pad
        d.takeoff(self.takeoff_alt)
        yield from self.wait(lambda: d.fresh(d.local, 1.5) and d.position()[2] > self.takeoff_alt - 0.2,
                             f'{d.name} takeoff', 60)
        x, y, _ = d.position()
        d.sp = (x, y, self.takeoff_alt, d.yaw)
        self.log(f'{d.name} airborne')

    def land(self, d, spot=None):
        while d.state.armed:
            if d.state.mode != 'LAND' and self.every(f'{d.name}_land', 1.0):
                d.set_mode('LAND')
            yield
        d.sp = None
        if spot is not None and d.fresh(d.local, 1.5):
            x, y, _ = d.position()
            self.log(f'{d.name} landed at ({x:+.2f}, {y:+.2f}), {math.dist((x, y), spot) * 100:.0f} cm from its spot')
        else:
            self.log(f'{d.name} landed')

    # ------------------------------------------------------------ the mission
    def launch_mother(self):
        """EKF origins, then the mother up to mother_alt on the ArUco fix; sets station / psi."""
        m = self.mother
        yield from self.wait(lambda: all(d.state.connected for d in self.all), 'all autopilots')
        for _ in range(3):                           # EKF origin (no GPS); ignored once set
            for d in self.all:
                d.send_origin(*self.origin)
            yield from self.sleep(1.0)

        # A short GUIDED takeoff, then a level climb (attitude target, no position target,
        # heading target = measured heading) until the reference is in view. Not one takeoff
        # to full height, and not a velocity climb on flow: when the first ArUco fix moves
        # her EKF into the arena frame mid-climb, the takeoff's (or the velocity
        # controller's) position target is suddenly metres away and she flies off to it.
        yield from self.enable_thrust(m)
        yield from self.arm_in(m, 'GUIDED')
        m.sp = m.climb_to = None
        m.takeoff(self.mother_takeoff_alt)
        self.log('mother taking off on optical flow')
        yield from self.wait(lambda: m.fresh(m.local, 1.5) and m.position()[2] > self.mother_takeoff_alt - 0.3,
                             'the mother to take off', 60)
        m.climb_to = self.mother_alt
        self.log('mother climbing level until the reference is in view')
        yield from self.wait(lambda: m.fresh(m.aruco) and m.aruco[1].pose.position.z >= self.mother_alt - 0.4
                             and m.ekf_uses_aruco(), 'the mother to reach altitude on the ArUco fix', 180)
        x, y, _ = m.position()
        self.station, self.psi = (x, y), m.yaw
        m.sp, m.climb_to = (x, y, self.mother_alt, self.psi), None
        self.log(f'mother on station at ({x:+.2f}, {y:+.2f}), heading {math.degrees(self.psi):+.0f} deg')

    def run(self):
        m = self.mother
        yield from self.launch_mother()
        ref, scanner = self.daughters
        s = -1.0      # scanner's side: its landing spot is at forward s * spot_fwd
        for stage in range(1, self.stages + 1):
            self.log(f'=== stage {stage}/{self.stages}: {scanner.name} scans, {ref.name} is the reference ===')
            yield from self.set_landmarks([ref.id])
            yield from self.daughter_takeoff(scanner)
            fw, sd, a = self.line_fwd, self.line_side, self.scan_alt
            # climb inwards, two scan lines (1 m in front of / behind the mother's nadir),
            # crossing over only on the leading side, then down to the leading edge
            for f, l, spd in ((s * fw, +sd, self.transit_speed), (s * fw, -sd, self.scan_speed),
                              (-s * fw, -sd, self.scan_speed), (-s * fw, +sd, self.scan_speed),
                              (-s * fw, -sd, self.transit_speed)):
                yield from self.move(scanner, *self.pt(f, l), a, spd)
            yield from self.land_safely(scanner, -s)
            yield from self.sleep(3.0)
            yield from self.set_landmarks([ref.id, scanner.id])        # freeze the new one
            if stage == self.stages:
                break

            # The old reference moves over next to it (crossing on the trailing side, whose
            # spot is now empty, so it never covers the new reference in the mother's image).
            yield from self.set_landmarks([scanner.id])
            yield from self.daughter_takeoff(ref)
            for f, l in ((-s * fw, +sd), (s * fw, +sd), (s * fw, -sd)):
                yield from self.move(ref, *self.pt(f, l), a, self.transit_speed)
            yield from self.land_safely(ref, s, keep_from=scanner.position()[:2])
            yield from self.sleep(3.0)
            yield from self.set_landmarks([scanner.id, ref.id])

            # Mother moves on: both landed daughters end up at the trailing edge of her view.
            self.station = self.pt(0.0, -self.step)
            self.log(f'mother moving {self.step} m to station ({self.station[0]:+.2f}, {self.station[1]:+.2f})')
            yield from self.move(m, *self.station, self.mother_alt, self.mother_speed, tol=0.25)
            ref, scanner = scanner, ref
        self.log('coverage complete: both daughters landed, mother holding position')


def main(args=None):
    rclpy.init(args=args)
    node = LeapfrogMission()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
