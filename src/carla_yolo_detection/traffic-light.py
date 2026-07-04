import carla
import random
import time
import numpy as np
import cv2
import torch
import warnings

warnings.filterwarnings("ignore")

print("正在加载 YOLOv5 神经网络模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
print(f"模型加载完毕！当前使用的计算设备是: {device.upper()}")

# 全局变量
latest_image = None
latest_depth = None

# AEB 距离阈值（米）
DIST_WARN   = 15.0
DIST_BRAKE  = 5.0
aeb_state = "NORMAL"

# LDW 偏移阈值（像素）
LDW_THRESHOLD = 60

def camera_callback(image):
    global latest_image
    latest_image = image

def depth_callback(image):
    global latest_depth
    latest_depth = image

def decode_depth(depth_image):
    raw = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    raw = raw.reshape((depth_image.height, depth_image.width, 4))
    R = raw[:, :, 2].astype(np.float32)
    G = raw[:, :, 1].astype(np.float32)
    B = raw[:, :, 0].astype(np.float32)
    depth_m = (R + G * 256.0 + B * 65536.0) / 16777215.0 * 1000.0
    return depth_m

def get_box_depth(depth_map, x1, y1, x2, y2):
    h, w = depth_map.shape
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw = max(1, (x2 - x1) // 5)
    bh = max(1, (y2 - y1) // 5)
    px1, px2 = max(0, cx - bw), min(w - 1, cx + bw)
    py1, py2 = max(0, cy - bh), min(h - 1, cy + bh)
    patch = depth_map[py1:py2, px1:px2]
    if patch.size == 0:
        return -1.0
    return float(np.median(patch))

def apply_aeb(vehicle, min_dist):
    global aeb_state
    ctrl = carla.VehicleControl()

    if min_dist > DIST_WARN:
        new_state = "NORMAL"
    elif min_dist > DIST_BRAKE:
        new_state = "WARN"
    else:
        new_state = "BRAKE"
        ctrl.throttle = 0.0
        ctrl.brake    = 1.0
        vehicle.apply_control(ctrl)

    if new_state != aeb_state:
        aeb_state = new_state
        if new_state == "WARN":
            print(f"\n[⚠ AEB预警] 前方目标 {min_dist:.1f}m，注意！")
        elif new_state == "BRAKE":
            print(f"\n[🚨 紧急制动] 前方目标 {min_dist:.1f}m，已刹车！")
        else:
            print(f"\n[✅ AEB] 解除制动，恢复 autopilot")
            vehicle.set_autopilot(True, 8000)
    return aeb_state

# ── LDW 车道线检测 ────────────────────────────────────────────────────────
def detect_lanes(img_bgr):
    h, w = img_bgr.shape[:2]
    img_lane = img_bgr.copy()
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    roi_vertices = np.array([[
        (0,          h),
        (w * 0.1,    h * 0.55),
        (w * 0.9,    h * 0.55),
        (w,          h),
    ]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1, theta=np.pi/180,
        threshold=30, minLineLength=30, maxLineGap=100
    )

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
            xs += [x1, x2]
            ys += [y1, y2]
        try:
            fit = np.polyfit(ys, xs, 1)
        except Exception:
            return None
        y_bottom = img_h
        y_top    = int(img_h * 0.55)
        x_bottom = int(np.polyval(fit, y_bottom))
        x_top    = int(np.polyval(fit, y_top))
        return (x_bottom, y_bottom, x_top, y_top, fit)

    left  = average_line(left_lines,  h)
    right = average_line(right_lines, h)

    img_cx = w // 2
    offset = 0
    ldw_state = "NORMAL"

    left_x_bottom  = left[0]  if left  else None
    right_x_bottom = right[0] if right else None

    if left_x_bottom is not None and right_x_bottom is not None:
        lane_cx = (left_x_bottom + right_x_bottom) // 2
        offset  = lane_cx - img_cx
    elif left_x_bottom is not None:
        offset = left_x_bottom - (img_cx - 160)
    elif right_x_bottom is not None:
        offset = (img_cx + 160) - right_x_bottom

    if offset > LDW_THRESHOLD: ldw_state = "RIGHT"
    elif offset < -LDW_THRESHOLD: ldw_state = "LEFT"

    if left is not None and right is not None:
        lx_b, ly_b, lx_t, ly_t, _ = left
        rx_b, ry_b, rx_t, ry_t, _ = right
        pts = np.array([[lx_b, ly_b], [lx_t, ly_t], [rx_t, ry_t], [rx_b, ry_b]], dtype=np.int32)
        overlay = img_lane.copy()
        cv2.fillPoly(overlay, [pts], (255, 180, 0))
        cv2.addWeighted(overlay, 0.25, img_lane, 0.75, 0, img_lane)

    lane_color = (255, 200, 0)
    if ldw_state != "NORMAL": lane_color = (0, 80, 255)

    if left is not None:
        cv2.line(img_lane, (left[0], left[1]), (left[2], left[3]), lane_color, 4)
    if right is not None:
        cv2.line(img_lane, (right[0], right[1]), (right[2], right[3]), lane_color, 4)

    if left is not None and right is not None:
        mid_b = ((left[0] + right[0]) // 2, h)
        mid_t = ((left[2] + right[2]) // 2, int(h * 0.55))
        for i in range(0, 10):
            t0, t1 = i / 10, (i + 0.5) / 10
            p0 = (int(mid_b[0] + (mid_t[0] - mid_b[0]) * t0), int(mid_b[1] + (mid_t[1] - mid_b[1]) * t0))
            p1 = (int(mid_b[0] + (mid_t[0] - mid_b[0]) * t1), int(mid_b[1] + (mid_t[1] - mid_b[1]) * t1))
            cv2.line(img_lane, p0, p1, (0, 255, 255), 2)

    return img_lane, offset, ldw_state

# ── 第5次提交：信号灯颜色分析（针对CARLA仿真优化）────────────────────────────
def analyze_traffic_light_color(roi_bgr):
    """
    CARLA仿真信号灯特点：
    1. 只有1/3的灯在亮，其余是暗的
    2. 亮灯是自发光的，亮度极高，颜色可能过曝
    3. 红灯在最上方，绿灯在最下方
    策略：把框分成上/中/下三段，分别判断哪段最亮，再根据位置+颜色综合判断
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN"
    
    h, w = roi_bgr.shape[:2]
    if h < 6 or w < 3:   # 框太小，直接跳过
        return "UNKNOWN"
    
    # 把框切成上、中、下三段
    third = h // 3
    seg_top = roi_bgr[0:third, :]
    seg_mid = roi_bgr[third:2*third, :]
    seg_bot = roi_bgr[2*third:, :]
    
    def brightness(seg):
        """返回区域的平均亮度（V通道）"""
        if seg.size == 0:
            return 0
        hsv = cv2.cvtColor(seg, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))
    
    b_top = brightness(seg_top)
    b_mid = brightness(seg_mid)
    b_bot = brightness(seg_bot)
    b_max = max(b_top, b_mid, b_bot)
    
    # 亮度太低说明灯太远或背光，无法判断
    if b_max < 80:
        return "UNKNOWN"
    
    # 找出最亮的那段，根据位置判断灯色
    # 红灯=最上段最亮，黄灯=中段最亮，绿灯=最下段最亮
    if b_top >= b_mid and b_top >= b_bot:
        brightest_seg = seg_top
        position_color = "RED"
    elif b_mid >= b_top and b_mid >= b_bot:
        brightest_seg = seg_mid
        position_color = "YELLOW"
    else:
        brightest_seg = seg_bot
        position_color = "GREEN"
        
    # 再对最亮的那段做HSV颜色验证，双重确认
    hsv = cv2.cvtColor(brightest_seg, cv2.COLOR_BGR2HSV)
    mask_red1   = cv2.inRange(hsv, (0,   30, 80), (10,  255, 255))
    mask_red2   = cv2.inRange(hsv, (160, 30, 80), (180, 255, 255))
    mask_red    = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, (15,  30, 80), (35,  255, 255))
    mask_green  = cv2.inRange(hsv, (35,  30, 80), (100, 255, 255))
    
    s_red    = cv2.countNonZero(mask_red)
    s_yellow = cv2.countNonZero(mask_yellow)
    s_green  = cv2.countNonZero(mask_green)
    s_max    = max(s_red, s_yellow, s_green)
    
    # 如果颜色验证和位置判断一致，直接返回
    # 如果颜色验证没有明显结果（过曝发白），就信任位置判断
    if s_max < 3:
        return position_color   # 过曝情况下只看位置
        
    if s_max == s_red:
        hsv_color = "RED"
    elif s_max == s_yellow:
        hsv_color = "YELLOW"
    else:
        hsv_color = "GREEN"
        
    # 位置和颜色都说是同一个，最可信
    if hsv_color == position_color:
        return position_color
        
    # 两者不一致时，优先信任位置（因为仿真里颜色过曝更常见）
    return position_color
# ─────────────────────────────────────────────────────────────────────────

def spawn_traffic(client, world, number_of_vehicles=30):
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager(8000)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.set_synchronous_mode(False)
    vehicle_bps = bp_lib.filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    temp_actors = []
    for i in range(min(number_of_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        npc = world.try_spawn_actor(bp, spawn_points[i])
        if npc:
            npc.set_autopilot(True, tm.get_port())
            temp_actors.append(npc)
    return temp_actors

def collision_handler(event):
    print(f"\n[💥碰撞预警] 发生碰撞! 撞到了: {event.other_actor.type_id}")

def main():
    global latest_image, latest_depth
    actor_list = []
    vehicle_classes = {'car', 'truck', 'bus', 'motorbike', 'bicycle', 'person'}
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        bp_lib = world.get_blueprint_library()
        
        print("[INFO] 正在清理地图上的残留 actor...")
        all_actors = world.get_actors()
        vehicles = all_actors.filter('vehicle.*')
        sensors  = all_actors.filter('sensor.*')
        for a in list(sensors) + list(vehicles):
            a.destroy()
        print(f"[INFO] 清理完成：{len(list(vehicles))} 辆车，{len(list(sensors))} 个传感器")
        
        vehicle_bp   = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        vehicle = None
        used_index = 0
        for idx, sp in enumerate(spawn_points):
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle:
                used_index = idx
                break
        if vehicle is None: raise RuntimeError("无可用出生点")
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)
        print(f"[INFO] 自车出生点索引: {used_index}, 位置: {spawn_points[used_index].location}")
        
        spectator = world.get_spectator()
        t0 = vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            t0.location + carla.Location(z=50),
            carla.Rotation(pitch=-90)
        ))
        print("[INFO] CARLA视角已对准自车，可自由移动视角")
        
        traffic_actors = spawn_traffic(client, world, 30)
        actor_list.extend(traffic_actors)
        
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '480')
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(camera_callback)
        
        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '640')
        depth_bp.set_attribute('image_size_y', '480')
        depth_bp.set_attribute('fov', '90')
        depth_cam = world.spawn_actor(depth_bp, cam_transform, attach_to=vehicle)
        actor_list.append(depth_cam)
        depth_cam.listen(depth_callback)
        
        col_bp = bp_lib.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=vehicle)
        actor_list.append(collision_sensor)
        collision_sensor.listen(collision_handler)
        
        print("\n✅ 系统启动！按 Ctrl+C 退出...")
        
        while True:
            if latest_image is not None:
                start_time = time.time()
                img_data   = latest_image
                latest_image = None
                depth_data = latest_depth
                
                i  = np.array(img_data.raw_data)
                i2 = i.reshape((img_data.height, img_data.width, 4))
                img_bgr = i2[:, :, :3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                depth_map = decode_depth(depth_data) if depth_data is not None else None
                
                spd = vehicle.get_velocity()
                speed_ms = (spd.x**2 + spd.y**2 + spd.z**2) ** 0.5
                if speed_ms > 1.5:
                    img_display, lane_offset, ldw_state = detect_lanes(img_bgr)
                else:
                    img_display = img_bgr.copy()
                    lane_offset, ldw_state = 0, "NORMAL"
                    
                results    = model(img_rgb)
                detections = results.xyxy[0]
                min_dist = float('inf')
                current_tl_state = "UNKNOWN"
                
                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label    = results.names[int(cls)]
                    conf_val = float(conf)
                    
                    if label == 'traffic light' and conf_val > 0.5:
                        roi = img_bgr[y1:y2, x1:x2]
                        tl_color = analyze_traffic_light_color(roi)
                        
                        if tl_color == "RED":
                            current_tl_state = "RED"
                        elif tl_color == "YELLOW" and current_tl_state != "RED":
                            current_tl_state = "YELLOW"
                        elif tl_color == "GREEN" and current_tl_state == "UNKNOWN":
                            current_tl_state = "GREEN"
                            
                        color_rgb = (255, 255, 255)
                        if tl_color == "RED":    color_rgb = (0, 0, 255)
                        elif tl_color == "YELLOW": color_rgb = (0, 255, 255)
                        elif tl_color == "GREEN":  color_rgb = (0, 255, 0)
                        
                        cv2.rectangle(img_display, (x1, y1), (x2, y2), color_rgb, 2)
                        cv2.putText(img_display, f"{tl_color} {conf_val:.0%}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_rgb, 2)
                        continue
                        
                    dist = -1.0
                    if depth_map is not None and label in vehicle_classes:
                        dist = get_box_depth(depth_map, x1, y1, x2, y2)
                        if dist > 0: min_dist = min(min_dist, dist)
                        
                    if dist <= 0:          color = (0, 255, 0)
                    elif dist > DIST_WARN: color = (0, 255, 0)
                    elif dist > DIST_BRAKE: color = (0, 200, 255)
                    else:                  color = (0, 0, 255)
                    
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                    text = f"{label}: {dist:.1f}m" if dist > 0 else f"{label} {conf_val:.0%}"
                    cv2.putText(img_display, text, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                                
                state = apply_aeb(vehicle, min_dist)
                if state == "WARN":
                    cv2.putText(img_display, f"WARNING: {min_dist:.1f}m",
                                (10, img_data.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                elif state == "BRAKE":
                    overlay = img_display.copy()
                    cv2.rectangle(overlay, (0, 0), (img_data.width, img_data.height), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.15, img_display, 0.85, 0, img_display)
                    cv2.putText(img_display, f"EMERGENCY BRAKE! {min_dist:.1f}m",
                                (10, img_data.height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # 信号灯图标（右侧圆点）
                icon_x, icon_y = img_data.width - 40, 90
                cv2.circle(img_display, (icon_x, icon_y), 15, (50, 50, 50), -1)
                
                if current_tl_state == "RED":
                    cv2.circle(img_display, (icon_x, icon_y), 15, (0, 0, 255), -1)
                    cv2.putText(img_display, "RED LIGHT",
                                (img_data.width // 2 - 80, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if speed_ms > 0.5:
                        cv2.putText(img_display, "! RED LIGHT AHEAD - BRAKE!",
                                    (img_data.width // 2 - 160, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                elif current_tl_state == "YELLOW":
                    cv2.circle(img_display, (icon_x, icon_y), 15, (0, 255, 255), -1)
                elif current_tl_state == "GREEN":
                    cv2.circle(img_display, (icon_x, icon_y), 15, (0, 255, 0), -1)

                # LDW
                if ldw_state == "LEFT":
                    cv2.putText(img_display, "! LDW: 偏右！",
                                (img_data.width // 2 - 80, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
                elif ldw_state == "RIGHT":
                    cv2.putText(img_display, "! LDW: 偏左！",
                                (img_data.width // 2 - 80, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

                offset_color = (0, 255, 0) if ldw_state == "NORMAL" else (0, 80, 255)
                cv2.putText(img_display, f"Lane offset: {lane_offset:+d}px",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)

                fps = 1.0 / (time.time() - start_time)
                cv2.putText(img_display, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                loc = vehicle.get_transform()
                cv2.putText(img_display,
                            f"X:{loc.location.x:.1f} Y:{loc.location.y:.1f} Yaw:{loc.rotation.yaw:.0f}deg",
                            (10, img_data.height - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                state_color = {'NORMAL': (0,255,0), 'WARN': (0,200,255), 'BRAKE': (0,0,255)}
                cv2.putText(img_display, f"AEB: {state}",
                            (img_data.width - 180, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            state_color.get(state, (255,255,255)), 2)

                cv2.imshow("CARLA YOLO + Depth AEB + LDW + Traffic Light", img_display)
                cv2.waitKey(1)
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n正在关闭系统...")
    finally:
        for actor in actor_list:
            if actor is not None:
                actor.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()