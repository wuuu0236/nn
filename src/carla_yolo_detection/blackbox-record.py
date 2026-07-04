import carla
import random
import time
import numpy as np
import cv2
import torch
import warnings
import traceback
import os
import threading
import csv

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("\n[错误] 缺少 scipy 库！请在 Anaconda 终端运行: pip install scipy\n")
    exit(1)

warnings.filterwarnings("ignore")

print("正在加载 YOLOv5 神经网络模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.classes = [0, 1, 2, 3, 5, 7, 9]
print(f"模型加载完毕！当前使用的计算设备是: {device.upper()}")

# ==============================================================================
# ── 全局变量与状态追踪 ───────────────────────────────────────────────────────
# ==============================================================================
score = 100
stats = {
    'collision': 0, 
    'aeb': 0, 
    'ldw': 0, 
    'red_light': 0
}
state_flags = {
    'aeb_active': False, 
    'ldw_active': False, 
    'red_active': False
}
speed_history = []  

sensor_lock = threading.Lock()
latest_image = None
latest_depth = None

aeb_state = "NORMAL"
LDW_THRESHOLD = 60
last_collision_time = 0.0
current_weather_name = "CLEAR"

is_recording = False
is_replaying = False
recorder_filename = os.path.abspath("scenario.log")
# ==============================================================================

def box_iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    a_area = (a[2]-a[0]) * (a[3]-a[1])
    b_area = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (a_area + b_area - inter + 1e-6)

class Track:
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]
        ], np.float32)
        
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
        
        self.cx, self.cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        self.w, self.h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        
        self.kf.statePre = np.array([[self.cx],[self.cy],[self.w],[self.h],[0],[0],[0],[0]], np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        
        self.time_since_update = 0
        self.history = [(int(self.cx), int(self.cy))]  
        self.hits = 0
        self.info = None
        
    def predict(self):
        pred = self.kf.predict()
        self.cx, self.cy, self.w, self.h = pred[0,0], pred[1,0], pred[2,0], pred[3,0]
        self.time_since_update += 1
        return pred

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        
        self.history.append((int(cx), int(cy)))
        if len(self.history) > 30:
            self.history.pop(0)
            
        self.kf.correct(np.array([[cx],[cy],[w],[h]], np.float32))
        post = self.kf.statePost
        self.cx, self.cy, self.w, self.h = post[0,0], post[1,0], post[2,0], post[3,0]

    def get_bbox(self):
        bbox = [self.cx - self.w/2, self.cy - self.h/2, self.cx + self.w/2, self.cy + self.h/2]
        if np.isnan(bbox).any() or np.isinf(bbox).any():
            return [0, 0, 0, 0]
        return bbox

class SORTTracker:
    def __init__(self, max_age=3, iou_threshold=0.3):
        self.max_age = max_age  
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, dets):
        trks = np.zeros((len(self.tracks), 4))
        to_del = []
        for t, trk in enumerate(self.tracks):
            trk.predict()
            trks[t] = trk.get_bbox()
            if np.isnan(trks[t]).any() or trks[t].tolist() == [0,0,0,0]:
                to_del.append(t)
        for t in reversed(to_del):
            self.tracks.pop(t)

        matched, unmatched_dets, unmatched_trks = self._match(dets, trks)

        for m in matched:
            trk_idx, det_idx = m[1], m[0]
            self.tracks[trk_idx].update(dets[det_idx, :4])
            self.tracks[trk_idx].info = dets[det_idx, 4:].copy() 

        for i in unmatched_dets:
            trk = Track(dets[i, :4], self.next_id)
            trk.info = dets[i, 4:].copy()
            self.tracks.append(trk)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks

    def _match(self, dets, trks):
        if len(trks) == 0 or len(dets) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(dets))), list(range(len(trks)))
        
        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = box_iou(det, trk)
                
        if iou_matrix.size > 0 and iou_matrix.max() > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(row_ind, col_ind)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
            
        if len(matched_indices) > 0:
            unmatched_dets = [d for d in range(len(dets)) if d not in matched_indices[:, 0]]
            unmatched_trks = [t for t in range(len(trks)) if t not in matched_indices[:, 1]]
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(trks)))
                
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0, 2), dtype=int)
        return matches, unmatched_dets, unmatched_trks

def get_track_color(track_id):
    np.random.seed(track_id * 100)
    color = np.random.randint(50, 255, size=3).tolist()
    return (color[0], color[1], color[2])

tracker = SORTTracker(max_age=3, iou_threshold=0.3)

def camera_callback(image): 
    global latest_image
    with sensor_lock: latest_image = image

def depth_callback(image): 
    global latest_depth
    with sensor_lock: latest_depth = image

def decode_depth(depth_image):
    raw = np.frombuffer(depth_image.raw_data, dtype=np.uint8).reshape((depth_image.height, depth_image.width, 4))
    return (raw[:, :, 2].astype(np.float32) + raw[:, :, 1].astype(np.float32) * 256.0 + raw[:, :, 0].astype(np.float32) * 65536.0) / 16777215.0 * 1000.0

def get_box_depth(depth_map, x1, y1, x2, y2):
    h, w = depth_map.shape
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = max(1, (x2 - x1) // 5), max(1, (y2 - y1) // 5)
    px1, px2 = max(0, cx - bw), min(w - 1, cx + bw)
    py1, py2 = max(0, cy - bh), min(h - 1, cy + bh)
    patch = depth_map[py1:py2, px1:px2]
    return float(np.median(patch)) if patch.size > 0 else -1.0

def apply_aeb(vehicle, min_dist, dist_warn, dist_brake):
    global aeb_state, is_replaying
    if is_replaying: 
        return aeb_state  
    
    ctrl = vehicle.get_control()
    if min_dist > dist_warn:
        new_state = "NORMAL"
        if aeb_state == "BRAKE":
            ctrl.brake = 0.0
            vehicle.apply_control(ctrl)
            vehicle.set_autopilot(True, 8000)  
    elif min_dist > dist_brake:
        new_state = "WARN"
        if aeb_state == "BRAKE":
            ctrl.brake = 0.0
            vehicle.apply_control(ctrl)
            vehicle.set_autopilot(True, 8000)
    else:
        new_state = "BRAKE"
        if aeb_state != "BRAKE":
            vehicle.set_autopilot(False)       
        ctrl.throttle = 0.0
        brake_force = min(1.0, (dist_brake - min_dist) / dist_brake + 0.3)
        ctrl.brake = float(max(0.3, brake_force)) 
        vehicle.apply_control(ctrl)

    if new_state != aeb_state:
        aeb_state = new_state
        if new_state == "WARN": print(f"\n[⚠ AEB预警] 前方目标 {min_dist:.1f}m，注意！")
        elif new_state == "BRAKE": print(f"\n[🚨 紧急制动] 前方目标 {min_dist:.1f}m，当前制动力度: {ctrl.brake:.2f}")
        else: print(f"\n[✅ AEB] 解除制动，恢复 autopilot")
    return aeb_state

def detect_lanes(img_bgr):
    h, w = img_bgr.shape[:2]
    img_lane = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    roi_vertices = np.array([[(0, h), (w * 0.1, h * 0.55), (w * 0.9, h * 0.55), (w, h)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=100)

    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3: continue
            if slope < 0: left_lines.append(line[0])
            else: right_lines.append(line[0])

    def average_line(line_group, img_h):
        if not line_group: return None
        xs, ys = [], []
        for x1, y1, x2, y2 in line_group:
            xs += [x1, x2]; ys += [y1, y2]
        try: fit = np.polyfit(ys, xs, 1)
        except: return None
        return (int(np.polyval(fit, img_h)), img_h, int(np.polyval(fit, int(img_h * 0.55))), int(img_h * 0.55), fit)

    left, right = average_line(left_lines, h), average_line(right_lines, h)
    img_cx, offset, ldw_state = w // 2, 0, "NORMAL"
    lx_b, rx_b = left[0] if left else None, right[0] if right else None

    if lx_b is not None and rx_b is not None: offset = (lx_b + rx_b) // 2 - img_cx
    elif lx_b is not None: offset = lx_b - (img_cx - 160)
    elif rx_b is not None: offset = (img_cx + 160) - rx_b

    if offset > LDW_THRESHOLD: ldw_state = "RIGHT"
    elif offset < -LDW_THRESHOLD: ldw_state = "LEFT"

    if left and right:
        pts = np.array([[left[0], left[1]], [left[2], left[3]], [right[2], right[3]], [right[0], right[1]]], dtype=np.int32)
        overlay = img_lane.copy()
        cv2.fillPoly(overlay, [pts], (255, 180, 0))
        cv2.addWeighted(overlay, 0.25, img_lane, 0.75, 0, img_lane)

    lane_color = (0, 80, 255) if ldw_state != "NORMAL" else (255, 200, 0)
    if left: cv2.line(img_lane, (left[0], left[1]), (left[2], left[3]), lane_color, 4)
    if right: cv2.line(img_lane, (right[0], right[1]), (right[2], right[3]), lane_color, 4)

    return img_lane, offset, ldw_state

def analyze_traffic_light_color(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0: return "UNKNOWN"
    h, w = roi_bgr.shape[:2]
    if h < 6 or w < 3: return "UNKNOWN"
    
    third = h // 3
    seg_top, seg_mid, seg_bot = roi_bgr[0:third, :], roi_bgr[third:2*third, :], roi_bgr[2*third:, :]
    
    def brightness(seg):
        return float(np.mean(cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)[:, :, 2])) if seg.size > 0 else 0
    
    b_top, b_mid, b_bot = brightness(seg_top), brightness(seg_mid), brightness(seg_bot)
    b_max = max(b_top, b_mid, b_bot)
    if b_max < 80: return "UNKNOWN"
    
    if b_top >= b_mid and b_top >= b_bot: return "RED"
    elif b_mid >= b_top and b_mid >= b_bot: return "YELLOW"
    else: return "GREEN"

def spawn_traffic(client, world, number_of_vehicles=30):
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(False)
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    temp_actors = []
    for i in range(min(number_of_vehicles, len(spawn_points))):
        npc = world.try_spawn_actor(random.choice(bp_lib.filter('vehicle.*')), spawn_points[i])
        if npc:
            npc.set_autopilot(True, tm.get_port())
            temp_actors.append(npc)
    return temp_actors

def collision_handler(event): 
    global score, stats, last_collision_time
    current_time = time.time()
    if current_time - last_collision_time < 3.0:
        return
    last_collision_time = current_time
    
    print(f"\n[💥碰撞预警] 发生碰撞! 撞到了: {event.other_actor.type_id}")
    stats['collision'] += 1
    score -= 20

def generate_report():
    avg_speed = np.mean(speed_history) if speed_history else 0.0
    final_score = max(0, score)
    report = f"""
==================================================
           CARLA 自动驾驶行为评测报告
==================================================
【最终综合评分】: {final_score} / 100 分

【行为扣分明细】:
- 碰撞发生次数 : {stats['collision']} 次  (扣除 {stats['collision']*20} 分)
- AEB 紧急制动 : {stats['aeb']} 次  (扣除 {stats['aeb']*3} 分)
- LDW 车道偏离 : {stats['ldw']} 次  (扣除 {stats['ldw']*3} 分)
- 违规闯红灯   : {stats['red_light']} 次  (扣除 {stats['red_light']*10} 分)

【行驶参考数据】:
- 全程平均车速 : {avg_speed:.1f} km/h
==================================================
"""
    print(report)
    try:
        report_path = os.path.join(os.getcwd(), "driving_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[INFO] 报告已自动保存至: {report_path}\n")
    except Exception as e:
        print(f"[错误] 报告保存失败: {e}")

def set_weather(world, weather_type):
    global current_weather_name
    weather = carla.WeatherParameters()
    if weather_type == 'CLEAR':
        weather.cloudiness, weather.precipitation, weather.fog_density = 0.0, 0.0, 0.0
        weather.sun_altitude_angle = 45.0
        current_weather_name = "CLEAR"
    elif weather_type == 'RAIN':
        weather.cloudiness, weather.precipitation, weather.fog_density = 80.0, 80.0, 10.0
        weather.sun_altitude_angle = 45.0
        current_weather_name = "RAIN"
    elif weather_type == 'FOG':
        weather.cloudiness, weather.precipitation, weather.fog_density = 20.0, 0.0, 80.0
        weather.sun_altitude_angle = 45.0
        current_weather_name = "FOG"
    elif weather_type == 'NIGHT':
        weather.cloudiness, weather.precipitation, weather.fog_density = 0.0, 0.0, 0.0
        weather.sun_altitude_angle = -10.0 
        current_weather_name = "NIGHT"
    world.set_weather(weather)
    print(f"\n[天气系统] 切换至: {current_weather_name}")


def main():
    global latest_image, latest_depth, tracker, score, stats, state_flags, speed_history
    global current_weather_name, is_recording, is_replaying, recorder_filename
    actor_list = []
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'}
    
    system_start_time = time.time()
    
    blackbox_path = os.path.join(os.getcwd(), 'blackbox.csv')
    blackbox_file = open(blackbox_path, mode='w', newline='', encoding='utf-8')
    blackbox_writer = csv.writer(blackbox_file)
    blackbox_writer.writerow(['timestamp', 'x', 'y', 'z', 'speed_kmh', 'yaw', 'aeb_state', 'ldw_state', 'score'])
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        
        print("[INFO] 正在清理地图上的残留 actor...")
        for a in list(world.get_actors().filter('sensor.*')) + list(world.get_actors().filter('vehicle.*')): a.destroy()
        
        set_weather(world, 'CLEAR')
        
        vehicle = None
        for idx, sp in enumerate(world.get_map().get_spawn_points()):
            vehicle = world.try_spawn_actor(bp_lib.filter('vehicle.tesla.model3')[0], sp)
            if vehicle: break
        if not vehicle: raise RuntimeError("无可用出生点")
        
        actor_list.append(vehicle)
        vehicle.set_autopilot(True, 8000)
        
        world.get_spectator().set_transform(carla.Transform(vehicle.get_transform().location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        actor_list.extend(spawn_traffic(client, world, 30))
        
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(bp_lib.find('sensor.camera.rgb'), cam_transform, attach_to=vehicle)
        actor_list.append(camera); camera.listen(camera_callback)
        depth_cam = world.spawn_actor(bp_lib.find('sensor.camera.depth'), cam_transform, attach_to=vehicle)
        actor_list.append(depth_cam); depth_cam.listen(depth_callback)
        col_sens = world.spawn_actor(bp_lib.find('sensor.other.collision'), carla.Transform(), attach_to=vehicle)
        actor_list.append(col_sens); col_sens.listen(collision_handler)
        
        print("\n✅ 系统启动完毕！")
        print("====== 录像与控制台 ======")
        print("  [C/R/F/N] : 切换天气")
        print("  [W] : 开始/停止 录像 (Toggle Record)")
        print("  [P] : 回放录像 (Play Replay)")
        print("  [S] : 停止回放/保存录像 (Stop current action)")
        print("  [q] : 退出系统")
        print("==========================\n")
        
        while True:
            with sensor_lock:
                img_data, depth_data = latest_image, latest_depth
                latest_image = None
                latest_depth = None
                
            if img_data is not None:
                start_time = time.time()
                
                img_bgr = np.array(img_data.raw_data).reshape((img_data.height, img_data.width, 4))[:, :, :3]
                depth_map = decode_depth(depth_data) if depth_data is not None else None
                
                spd = vehicle.get_velocity()
                speed_ms = (spd.x**2 + spd.y**2 + spd.z**2) ** 0.5
                speed_kmh = speed_ms * 3.6
                transform = vehicle.get_transform()
                
                dynamic_dist_brake = max(5.0, speed_ms * 1.5)
                dynamic_dist_warn  = max(15.0, speed_ms * 3.0 + 5.0)
                
                if speed_kmh > 1.0:
                    speed_history.append(speed_kmh)
                    if len(speed_history) > 10000:
                        speed_history.pop(0)

                img_display, lane_offset, ldw_state = detect_lanes(img_bgr) if speed_ms > 1.5 else (img_bgr.copy(), 0, "NORMAL")
                    
                if ldw_state != "NORMAL":
                    if not state_flags['ldw_active']:
                        stats['ldw'] += 1
                        score -= 3
                        state_flags['ldw_active'] = True
                else:
                    state_flags['ldw_active'] = False

                results = model(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                detections = results.xyxy[0]
                
                current_tl_state = "UNKNOWN"
                tl_candidates = []
                raw_dets_for_tracker = []  
                
                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = results.names[int(cls)]
                    
                    if label == 'traffic light' and float(conf) > 0.5:
                        cx = (x1 + x2) // 2
                        if abs(cx - img_data.width // 2) < img_data.width * 0.3:
                            tl_color = analyze_traffic_light_color(img_bgr[y1:y2, x1:x2])
                            area = (x2 - x1) * (y2 - y1)
                            tl_candidates.append((area, tl_color, (x1, y1, x2, y2)))
                        continue
                        
                    if label in vehicle_classes:
                        dist = get_box_depth(depth_map, x1, y1, x2, y2) if depth_map is not None else -1.0
                        raw_dets_for_tracker.append([x1, y1, x2, y2, float(conf), int(cls), dist])
                
                if tl_candidates:
                    tl_candidates.sort(key=lambda x: x[0], reverse=True)
                    current_tl_state = tl_candidates[0][1]
                    tx1, ty1, tx2, ty2 = tl_candidates[0][2]
                    tc = (0,0,255) if current_tl_state=="RED" else (0,255,255) if current_tl_state=="YELLOW" else (0,255,0)
                    cv2.rectangle(img_display, (tx1, ty1), (tx2, ty2), tc, 2)
                    cv2.putText(img_display, f"{current_tl_state}", (tx1, ty1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 2)

                if current_tl_state == "RED":
                    if speed_kmh > 5.0 and (time.time() - system_start_time) > 5.0 and not state_flags['red_active']:
                        stats['red_light'] += 1
                        score -= 10
                        state_flags['red_active'] = True
                else:
                    state_flags['red_active'] = False

                min_dist = float('inf')
                dets_array = np.array(raw_dets_for_tracker) if len(raw_dets_for_tracker) > 0 else np.empty((0, 7))
                tracked_objects = tracker.update(dets_array)
                
                for trk in tracked_objects:
                    bbox = trk.get_bbox()
                    if trk.time_since_update <= 1 and trk.info is not None and bbox != [0,0,0,0]:
                        x1, y1, x2, y2 = map(int, bbox)
                        conf, cls, dist = trk.info
                        label = results.names[int(cls)]
                        track_id = trk.id
                        
                        if trk.time_since_update > 0 and depth_map is not None:
                            dist = get_box_depth(depth_map, max(0, x1), max(0, y1), min(img_data.width-1, x2), min(img_data.height-1, y2))
                            trk.info[2] = dist
                        
                        cx = (x1 + x2) // 2
                        in_path = abs(cx - (img_data.width // 2)) < (img_data.width * 0.20)
                        
                        if 0 < dist < 100.0 and in_path:
                            min_dist = min(min_dist, dist)
                            
                        color = get_track_color(track_id)
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                        text = f"{label} #{track_id}: {dist:.1f}m" if dist > 0 else f"{label} #{track_id}"
                        cv2.putText(img_display, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        if len(trk.history) > 1:
                            for i in range(1, len(trk.history)):
                                cv2.line(img_display, trk.history[i-1], trk.history[i], color, max(1, int(3 * (i/len(trk.history))))) 

                state = apply_aeb(vehicle, min_dist, dynamic_dist_warn, dynamic_dist_brake)
                
                if state == "BRAKE":
                    if not state_flags['aeb_active']:
                        stats['aeb'] += 1
                        score -= 3
                        state_flags['aeb_active'] = True
                else:
                    state_flags['aeb_active'] = False

                # 确保 ldw_state 和 aeb_state 安全计算后再写入黑匣子
                blackbox_writer.writerow([
                    f"{time.time():.3f}", f"{transform.location.x:.2f}", f"{transform.location.y:.2f}", f"{transform.location.z:.2f}",
                    f"{speed_kmh:.2f}", f"{transform.rotation.yaw:.2f}", aeb_state, ldw_state, score
                ])

                if state == "WARN": cv2.putText(img_display, f"WARNING: {min_dist:.1f}m", (10, img_data.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                elif state == "BRAKE":
                    overlay = img_display.copy()
                    cv2.rectangle(overlay, (0, 0), (img_data.width, img_data.height), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.15, img_display, 0.85, 0, img_display)
                    cv2.putText(img_display, f"EMERGENCY BRAKE! {min_dist:.1f}m", (10, img_data.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                ix, iy = img_data.width - 40, 90
                cv2.circle(img_display, (ix, iy), 15, (50, 50, 50), -1)
                if current_tl_state == "RED":
                    cv2.circle(img_display, (ix, iy), 15, (0, 0, 255), -1)
                    if speed_ms > 0.5: cv2.putText(img_display, "! RED LIGHT AHEAD !", (img_data.width // 2 - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                elif current_tl_state == "YELLOW": cv2.circle(img_display, (ix, iy), 15, (0, 255, 255), -1)
                elif current_tl_state == "GREEN": cv2.circle(img_display, (ix, iy), 15, (0, 255, 0), -1)

                display_score = max(0, score)
                score_color = (0, 255, 0) if display_score >= 80 else (0, 165, 255) if display_score >= 60 else (0, 0, 255)
                cv2.putText(img_display, f"SCORE: {display_score}", (img_data.width - 180, img_data.height - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 3)

                cv2.putText(img_display, f"FPS: {1.0/(time.time()-start_time):.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                weather_color = (255, 255, 255) if current_weather_name in ["CLEAR", "RAIN"] else (150, 150, 150)
                cv2.putText(img_display, f"WEATHER: {current_weather_name}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, weather_color, 2)
                
                if is_recording:
                    if int(time.time() * 2) % 2 == 0: cv2.circle(img_display, (20, 95), 8, (0, 0, 255), -1)
                    cv2.putText(img_display, "REC [W:Stop/Save]", (40, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif is_replaying:
                    cv2.putText(img_display, "REPLAYING... [S:Stop]", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                else:
                    cv2.putText(img_display, "REC: OFF [W:Start P:Play S:Stop]", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                cv2.imshow("CARLA YOLO + Depth AEB + LDW + Traffic Light + SORT Tracking", img_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] 检测到键盘 'q' 退出指令...")
                    break
                elif key == ord('c'): set_weather(world, 'CLEAR')
                elif key == ord('r'): set_weather(world, 'RAIN')
                elif key == ord('f'): set_weather(world, 'FOG')
                elif key == ord('n'): set_weather(world, 'NIGHT')
                elif key == ord('w'):
                    if not is_recording and not is_replaying:
                        print(f"\n[录像机] 开始录制场景，日志将保存至: {recorder_filename}")
                        client.start_recorder(recorder_filename)
                        is_recording = True
                    elif is_recording: 
                        print("\n[录像机] 录制已停止并保存。")
                        client.stop_recorder()
                        is_recording = False
                elif key == ord('s'):
                    if is_recording:
                        print("\n[录像机] 录制已停止并保存。")
                        client.stop_recorder()
                        is_recording = False
                    elif is_replaying:
                        print("\n[录像机] 回放已手动停止。")
                        try: client.stop_replayer(True)
                        except: pass
                        is_replaying = False
                elif key == ord('p'):
                    if is_recording:
                        print("\n[录像机] 停止当前录制，准备回放...")
                        client.stop_recorder()
                        is_recording = False
                    if os.path.exists(recorder_filename):
                        print(f"\n[录像机] 开始回放录像文件: {recorder_filename}")
                        is_replaying = True
                        client.replay_file(recorder_filename, 0, 0, 0)
                    else:
                        print("\n[错误] 没有找到录制文件！请先按 'W' 录制一段数据。")
                
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[INFO] 检测到 Ctrl+C 中断指令...")
    except Exception as e:
        print(f"\n[程序运行发生崩溃] 错误原因: {e}")
        traceback.print_exc()
    finally:
        generate_report()
        if 'blackbox_file' in locals() and not blackbox_file.closed:
            blackbox_file.close()
            print(f"[INFO] 黑匣子数据已保存至: {blackbox_path}")
            
        for actor in actor_list:
            if actor: actor.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()