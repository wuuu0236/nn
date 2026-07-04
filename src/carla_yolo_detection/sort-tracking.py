import carla
import random
import time
import numpy as np
import cv2
import torch
import warnings
import traceback

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("\n[错误] 缺少 scipy 库！请在 Anaconda 终端运行: pip install scipy\n")
    exit(1)

warnings.filterwarnings("ignore")

print("正在加载 YOLOv5 神经网络模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
print(f"模型加载完毕！当前使用的计算设备是: {device.upper()}")

# ==============================================================================
# ── 第 6 次提交修复版：鲁棒性强化 SORT 多目标跟踪器 ───────────────────────────
# ==============================================================================

def box_iou(a, b):
    """计算两个边界框的交并比 (IoU)"""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    a_area = (a[2]-a[0]) * (a[3]-a[1])
    b_area = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (a_area + b_area - inter + 1e-6)

class Track:
    """单个目标的卡尔曼滤波器跟踪实例（强化修复版）"""
    def __init__(self, bbox, track_id):
        self.id = track_id
        # 状态向量: [cx, cy, w, h, dx, dy, dw, dh]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]
        ], np.float32)
        
        # ── CRITICAL FIX: 显式初始化噪声和误差协方差矩阵，防止底层 C++ 报错引发闪退 ──
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
        self.info = None  # 存储 [conf, cls, dist]
        
    def predict(self):
        """卡尔曼预测下一帧位置"""
        pred = self.kf.predict()
        self.cx, self.cy, self.w, self.h = pred[0,0], pred[1,0], pred[2,0], pred[3,0]
        self.time_since_update += 1
        return pred

    def update(self, bbox):
        """使用观测值校正卡尔曼状态"""
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
        """获取当前跟踪框，加入极端值防御，防止 NaN 传递引发格式转换闪退"""
        bbox = [self.cx - self.w/2, self.cy - self.h/2, self.cx + self.w/2, self.cy + self.h/2]
        if np.isnan(bbox).any() or np.isinf(bbox).any():
            return [0, 0, 0, 0]
        return bbox

class SORTTracker:
    """SORT 多目标跟踪管理器（鲁棒强化版）"""
    def __init__(self, max_age=3, iou_threshold=0.3):
        self.max_age = max_age  
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1

    def update(self, dets):
        # 1. 预测已有目标
        trks = np.zeros((len(self.tracks), 4))
        to_del = []
        for t, trk in enumerate(self.tracks):
            trk.predict()
            trks[t] = trk.get_bbox()
            if np.isnan(trks[t]).any() or trks[t].tolist() == [0,0,0,0]:
                to_del.append(t)
        for t in reversed(to_del):
            self.tracks.pop(t)

        # 2. 匈牙利算法进行 IoU 安全匹配
        matched, unmatched_dets, unmatched_trks = self._match(dets, trks)

        # 3. 更新匹配成功的跟踪器
        for m in matched:
            trk_idx, det_idx = m[1], m[0]
            self.tracks[trk_idx].update(dets[det_idx, :4])
            self.tracks[trk_idx].info = dets[det_idx, 4:] 

        # 4. 创建新跟踪器
        for i in unmatched_dets:
            trk = Track(dets[i, :4], self.next_id)
            trk.info = dets[i, 4:]
            self.tracks.append(trk)
            self.next_id += 1

        # 5. 移除失联长久的跟踪器
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks

    def _match(self, dets, trks):
        """强化版边界安全匹配算法，杜绝一切空值越界崩溃"""
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

# ==============================================================================

# 全局变量
latest_image = None
latest_depth = None
DIST_WARN   = 15.0
DIST_BRAKE  = 5.0
aeb_state = "NORMAL"
LDW_THRESHOLD = 60

# 实例化全局跟踪器
tracker = SORTTracker(max_age=3, iou_threshold=0.3)

def camera_callback(image): global latest_image; latest_image = image
def depth_callback(image): global latest_depth; latest_depth = image

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

def apply_aeb(vehicle, min_dist):
    global aeb_state
    ctrl = carla.VehicleControl()
    if min_dist > DIST_WARN: new_state = "NORMAL"
    elif min_dist > DIST_BRAKE: new_state = "WARN"
    else:
        new_state = "BRAKE"
        ctrl.throttle, ctrl.brake = 0.0, 1.0
        vehicle.apply_control(ctrl)

    if new_state != aeb_state:
        aeb_state = new_state
        if new_state == "WARN": print(f"\n[⚠ AEB预警] 前方目标 {min_dist:.1f}m，注意！")
        elif new_state == "BRAKE": print(f"\n[🚨 紧急制动] 前方目标 {min_dist:.1f}m，已刹车！")
        else:
            print(f"\n[✅ AEB] 解除制动，恢复 autopilot")
            vehicle.set_autopilot(True, 8000)
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
    
    if b_top >= b_mid and b_top >= b_bot: brightest_seg, pos_col = seg_top, "RED"
    elif b_mid >= b_top and b_mid >= b_bot: brightest_seg, pos_col = seg_mid, "YELLOW"
    else: brightest_seg, pos_col = seg_bot, "GREEN"
        
    hsv = cv2.cvtColor(brightest_seg, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, (0,30,80), (10,255,255)), cv2.inRange(hsv, (160,30,80), (180,255,255)))
    s_red = cv2.countNonZero(mask_red)
    s_yellow = cv2.countNonZero(cv2.inRange(hsv, (15,30,80), (35,255,255)))
    s_green = cv2.countNonZero(cv2.inRange(hsv, (35,30,80), (100,255,255)))
    s_max = max(s_red, s_yellow, s_green)
    
    if s_max < 3: return pos_col
    return pos_col

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

def collision_handler(event): print(f"\n[💥碰撞预警] 发生碰撞! 撞到了: {event.other_actor.type_id}")

def main():
    global latest_image, latest_depth, tracker
    actor_list = []
    vehicle_classes = {'car', 'truck', 'bus', 'motorbike', 'bicycle', 'person'}
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        
        print("[INFO] 正在清理地图上的残留 actor...")
        for a in list(world.get_actors().filter('sensor.*')) + list(world.get_actors().filter('vehicle.*')): a.destroy()
        
        vehicle = None
        for idx, sp in enumerate(world.get_map().get_spawn_points()):
            vehicle = world.try_spawn_actor(bp_lib.filter('vehicle.tesla.model3')[0], sp)
            if vehicle: break
        if not vehicle: raise RuntimeError("无可用出生点")
        
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)
        
        world.get_spectator().set_transform(carla.Transform(vehicle.get_transform().location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        actor_list.extend(spawn_traffic(client, world, 30))
        
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(bp_lib.find('sensor.camera.rgb'), cam_transform, attach_to=vehicle)
        actor_list.append(camera); camera.listen(camera_callback)
        depth_cam = world.spawn_actor(bp_lib.find('sensor.camera.depth'), cam_transform, attach_to=vehicle)
        actor_list.append(depth_cam); depth_cam.listen(depth_callback)
        col_sens = world.spawn_actor(bp_lib.find('sensor.other.collision'), carla.Transform(), attach_to=vehicle)
        actor_list.append(col_sens); col_sens.listen(collision_handler)
        
        print("\n✅ 系统启动！按 Ctrl+C 退出...")
        
        while True:
            if latest_image is not None:
                start_time = time.time()
                img_data, depth_data = latest_image, latest_depth
                latest_image = None
                
                img_bgr = np.array(img_data.raw_data).reshape((img_data.height, img_data.width, 4))[:, :, :3]
                depth_map = decode_depth(depth_data) if depth_data is not None else None
                
                spd = vehicle.get_velocity()
                speed_ms = (spd.x**2 + spd.y**2 + spd.z**2) ** 0.5
                img_display, lane_offset, ldw_state = detect_lanes(img_bgr) if speed_ms > 1.5 else (img_bgr.copy(), 0, "NORMAL")
                    
                results = model(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                detections = results.xyxy[0]
                
                current_tl_state = "UNKNOWN"
                raw_dets_for_tracker = []  
                
                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = results.names[int(cls)]
                    
                    if label == 'traffic light' and float(conf) > 0.5:
                        tl_color = analyze_traffic_light_color(img_bgr[y1:y2, x1:x2])
                        if tl_color == "RED": current_tl_state = "RED"
                        elif tl_color == "YELLOW" and current_tl_state != "RED": current_tl_state = "YELLOW"
                        elif tl_color == "GREEN" and current_tl_state == "UNKNOWN": current_tl_state = "GREEN"
                        c = (0,0,255) if tl_color=="RED" else (0,255,255) if tl_color=="YELLOW" else (0,255,0)
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), c, 2)
                        cv2.putText(img_display, f"{tl_color}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                        continue
                        
                    if label in vehicle_classes:
                        dist = get_box_depth(depth_map, x1, y1, x2, y2) if depth_map is not None else -1.0
                        raw_dets_for_tracker.append([x1, y1, x2, y2, float(conf), int(cls), dist])
                
                # ── 执行 SORT 跟踪 ──
                min_dist = float('inf')
                dets_array = np.array(raw_dets_for_tracker) if len(raw_dets_for_tracker) > 0 else np.empty((0, 7))
                tracked_objects = tracker.update(dets_array)
                
                # ── 绘制跟踪结果与轨迹 ──
                for trk in tracked_objects:
                    bbox = trk.get_bbox()
                    if trk.time_since_update == 0 and trk.info is not None and bbox != [0,0,0,0]:
                        x1, y1, x2, y2 = map(int, bbox)
                        conf, cls, dist = trk.info
                        label = results.names[int(cls)]
                        track_id = trk.id
                        
                        if dist > 0: min_dist = min(min_dist, dist)
                        color = get_track_color(track_id)
                        
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                        text = f"{label} #{track_id}: {dist:.1f}m" if dist > 0 else f"{label} #{track_id}"
                        cv2.putText(img_display, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        if len(trk.history) > 1:
                            for i in range(1, len(trk.history)):
                                cv2.line(img_display, trk.history[i-1], trk.history[i], color, max(1, int(3 * (i/len(trk.history))))) 

                # AEB
                state = apply_aeb(vehicle, min_dist)
                if state == "WARN": cv2.putText(img_display, f"WARNING: {min_dist:.1f}m", (10, img_data.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                elif state == "BRAKE":
                    overlay = img_display.copy()
                    cv2.rectangle(overlay, (0, 0), (img_data.width, img_data.height), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.15, img_display, 0.85, 0, img_display)
                    cv2.putText(img_display, f"EMERGENCY BRAKE! {min_dist:.1f}m", (10, img_data.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # 红绿灯 UI
                ix, iy = img_data.width - 40, 90
                cv2.circle(img_display, (ix, iy), 15, (50, 50, 50), -1)
                if current_tl_state == "RED":
                    cv2.circle(img_display, (ix, iy), 15, (0, 0, 255), -1)
                    if speed_ms > 0.5: cv2.putText(img_display, "! RED LIGHT AHEAD !", (img_data.width // 2 - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                elif current_tl_state == "YELLOW": cv2.circle(img_display, (ix, iy), 15, (0, 255, 255), -1)
                elif current_tl_state == "GREEN": cv2.circle(img_display, (ix, iy), 15, (0, 255, 0), -1)

                cv2.putText(img_display, f"FPS: {1.0/(time.time()-start_time):.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("CARLA YOLO + Depth AEB + LDW + Traffic Light + SORT Tracking", img_display)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n正在关闭系统...")
    except Exception as e:
        # ── CRITICAL FIX: 异常捕获机制，保护终端打印具体错因，拒绝盲目闪退 ──
        print(f"\n[程序运行发生崩溃] 错误原因: {e}")
        traceback.print_exc()
    finally:
        for actor in actor_list:
            if actor: actor.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()