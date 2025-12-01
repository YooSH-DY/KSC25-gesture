import json
import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import websocket

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# 멀티스레딩 카메라 리더 클래스
class ThreadedCamera:
    """별도 스레드에서 카메라 프레임을 읽어 성능 향상"""

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.frame = None
        self.ret = False
        self.running = False

    def start(self):
        self.running = True
        self.thread.start()
        return self

    def update(self):
        while self.running:
            if self.capture.isOpened():
                self.ret, self.frame = self.capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()

    def release(self):
        self.stop()
        self.capture.release()


# 전역 변수들
MODE_CONFIRMATION_THRESHOLD = 3
MODE5_CONFIRM_FRAMES = 100
ANGLE_THRESHOLD = 150

# 검지-중지 거리 임계값 (정규화된 거리 기준)
INDEX_MIDDLE_DISTANCE_THRESHOLD = 120.0

# 제스처 안정화를 위한 전역 변수들
GESTURE_STABILIZATION_TIME = 0.3  # 0.5초
current_gesture_candidate = None
gesture_start_time = None
stable_gesture = None

# Unity 웹소켓 통신 설정
UNITY_WEBSOCKET_URL = "ws://localhost:8765/gesture"
unity_websocket = None
last_sent_gesture = None

# 제스처 숫자 매핑
GESTURE_TO_NUMBER = {
    "L": 1,
    "3": 2,
    "B": 3,
    "G": 4,
    "1": 5,
    "L-I": 6,
    "1-I": 7,
    "8": 8,
    "Open N": 9,
    "Bent 3": 10,
    "Baby O": 11,
    # 무기 제스처
    "3_Fire": 4,  # 소총 발사 -> G 번호 사용
    "3_Reload": 5,  # 소총 재장전 -> 1 번호 사용
    "SG": 4,  # 샷건 발사 [0,1,1,1,1] -> G 번호 사용
    "S1": 5,  # 샷건 재장전 [-1,1,1,1,1] -> 1 번호 사용
    "M1": 5,  # 소총 재장전2 [-1,1,1,-1,-1] -> 1 번호 사용
}

# 엄지-다른손가락 접촉 임계값 (픽셀 거리 기준)
THUMB_TOUCH_THRESHOLD = 24

# 엄지와 Ring DIP 근접 판정용 임계값 (정규화된 palm_width 기준)
RING_DIP_THUMB_THRESHOLD = 0.06


# 각 카메라별 상태를 저장할 클래스
class CameraState:
    def __init__(self, camera_id, camera_type="top"):
        self.camera_id = camera_id
        self.camera_type = camera_type  # "top" 또는 "bottom"
        self.mode_confirmation_count = 0
        self.last_detected_mode = None
        self.last_confirmed_mode = None
        self.last_sent_mode = None
        self.last_sent_is = None
        self.mode5_counter = 0
        self.prev_mode = None
        self.mode = None

        # 손가락 각도 저장 (다른 카메라와 공유용)
        self.finger_angles = {}

        # 손가락 상태 저장 (1: straight, 0: between, -1: bent)
        self.finger_states_numeric = {}

        # 스무딩 객체들
        self.distance_smoother = ExponentialMovingAverage(alpha=0.2)
        self.angle_smoother = ExponentialMovingAverage(alpha=0.1)
        self.thumb_angle_smoother = ExponentialMovingAverage(alpha=0.1)

        # 캘리브레이션 시스템
        self.calibration = CalibrationSystem()

        # 웹소켓 전송 타이밍
        self.last_send_time = time.time()

        # 손날(측면) 방향 감지
        self.is_side_facing = False
        self.palm_normal_z = 0.0
        self.side_facing_confidence = 0.0


# 지수이동평균 스무딩 클래스
class ExponentialMovingAverage:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.last_value = None

    def smooth(self, value):
        if self.last_value is None:
            self.last_value = value
            return value
        smoothed_value = self.alpha * value + (1 - self.alpha) * self.last_value
        self.last_value = smoothed_value
        return smoothed_value

    def reset(self):
        self.last_value = None


# 캘리브레이션 시스템
class CalibrationSystem:
    def __init__(self):
        self.state = "ready"
        self.mode1_values = []
        self.mode2_values = []
        self.mode1_min = None
        self.mode1_max = None
        self.mode2_min = None
        self.mode2_max = None
        self.collection_count = 0
        self.target_samples = 60

        self.mode1_range_10_90 = None
        self.mode1_offset_10 = None
        self.mode2_range_10_90 = None
        self.mode2_offset_10 = None

    def get_remaining_time(self):
        remaining_frames = self.target_samples - self.collection_count
        remaining_seconds = remaining_frames / 30.0
        return max(0, remaining_seconds)

    def start_mode1_calibration(self):
        self.state = "mode1_collect"
        self.mode1_values = []
        self.collection_count = 0

    def start_mode2_calibration(self):
        self.state = "mode2_collect"
        self.mode2_values = []
        self.collection_count = 0

    def collect_sample(self, mode, distance_value):
        if self.state == "mode1_collect" and mode == "mode1":
            self.mode1_values.append(distance_value)
            self.collection_count += 1
            if self.collection_count >= self.target_samples:
                self.mode1_min = min(self.mode1_values)
                self.mode1_max = max(self.mode1_values)
                self._update_mode1_cache()
                self.state = "ready"

        elif self.state == "mode2_collect" and mode == "mode2":
            self.mode2_values.append(distance_value)
            self.collection_count += 1
            if self.collection_count >= self.target_samples:
                self.mode2_min = min(self.mode2_values)
                self.mode2_max = max(self.mode2_values)
                self._update_mode2_cache()
                self.state = "ready"

    def _update_mode1_cache(self):
        if self.mode1_min is not None and self.mode1_max is not None:
            range_val = self.mode1_max - self.mode1_min
            if range_val > 1e-6:
                self.mode1_range_10_90 = range_val * 0.7
                self.mode1_offset_10 = self.mode1_min + range_val * 0.15
            else:
                self.mode1_range_10_90 = None
                self.mode1_offset_10 = None

    def _update_mode2_cache(self):
        if self.mode2_min is not None and self.mode2_max is not None:
            range_val = self.mode2_max - self.mode2_min
            if range_val > 1e-6:
                self.mode2_range_10_90 = range_val * 0.7
                self.mode2_offset_10 = self.mode2_min + range_val * 0.15
            else:
                self.mode2_range_10_90 = None
                self.mode2_offset_10 = None

    def is_calibrated(self):
        return (
            self.mode1_min is not None
            and self.mode1_max is not None
            and self.mode2_min is not None
            and self.mode2_max is not None
        )

    def get_percentage(self, mode, distance_value):
        if (
            mode == "mode1"
            and self.mode1_range_10_90 is not None
            and self.mode1_offset_10 is not None
        ):
            normalized = (
                distance_value - self.mode1_offset_10
            ) / self.mode1_range_10_90
            return max(0, min(120, normalized * 100))
        elif (
            mode == "mode2"
            and self.mode2_range_10_90 is not None
            and self.mode2_offset_10 is not None
        ):
            normalized = (
                distance_value - self.mode2_offset_10
            ) / self.mode2_range_10_90
            return max(0, min(120, normalized * 100))
        return None

    def set_defaults(self):
        self.mode1_min = 0.3
        self.mode1_max = 1.5
        self.mode2_min = 0.1
        self.mode2_max = 0.8
        self._update_mode1_cache()
        self._update_mode2_cache()
        self.state = "ready"

    def reset(self):
        self.state = "ready"
        self.mode1_values = []
        self.mode2_values = []
        self.mode1_min = None
        self.mode1_max = None
        self.mode2_min = None
        self.mode2_max = None
        self.collection_count = 0
        self.mode1_range_10_90 = None
        self.mode1_offset_10 = None
        self.mode2_range_10_90 = None
        self.mode2_offset_10 = None


def check_hand_orientation(hand_landmarks):
    """손목의 방향을 확인하여 팔이 수직인지 판단"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    hand_vector_y = middle_mcp.y - wrist.y
    is_arm_raised = hand_vector_y < -0.05
    return is_arm_raised


def check_hand_side_orientation(hand_landmarks, camera_type="side"):
    """
    손이 측면(손날)을 향하는지 판단합니다.
    손바닥의 법선 벡터를 계산하여 Z축 방향 성분을 확인합니다.

    Args:
        hand_landmarks: MediaPipe 손 랜드마크
        camera_type: "side" 또는 "bottom" - 카메라별로 다른 임계값 적용

    Returns:
        tuple: (is_side_facing: bool, palm_normal_z: float, confidence: float)
            - is_side_facing: 손이 측면을 향하면 True
            - palm_normal_z: 손바닥 법선의 Z 성분
            - confidence: 판정 신뢰도 (0.0 ~ 1.0)
    """
    try:
        # 손바닥 평면을 정의하는 3개의 점
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        # 두 벡터 계산
        # v1: wrist -> index_mcp
        v1 = (index_mcp.x - wrist.x, index_mcp.y - wrist.y, index_mcp.z - wrist.z)

        # v2: wrist -> pinky_mcp
        v2 = (pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z)

        # 외적(cross product)으로 손바닥 평면의 법선 벡터 계산
        # normal = v1 × v2
        normal_x = v1[1] * v2[2] - v1[2] * v2[1]
        normal_y = v1[2] * v2[0] - v1[0] * v2[2]
        normal_z = v1[0] * v2[1] - v1[1] * v2[0]

        # 법선 벡터 정규화
        magnitude = math.sqrt(normal_x**2 + normal_y**2 + normal_z**2)

        if magnitude < 1e-6:
            return False, 0.0, 0.0

        norm_x = normal_x / magnitude
        norm_y = normal_y / magnitude
        norm_z = normal_z / magnitude

        # 카메라별 임계값 설정
        # 측면(상단) 카메라: norm_z <= threshold 이면 손날 (사용자 요청으로 -0.1로 조정)
        # 하단 카메라: norm_z >= -0.5 이면 손날 (하단은 반대로 적용)

        if camera_type == "side":
            # 측면 카메라: norm_z <= 0.3 이면 손날 (사용자 요청으로 상단 기준 임계값을 0.3으로 설정)
            threshold = 0.3
            is_side = norm_z <= threshold

            # 신뢰도 계산: threshold보다 작아질수록 신뢰도 증가
            if norm_z <= threshold:
                confidence = min(1.0, max(0.0, (threshold - norm_z) / 0.7))
            else:
                confidence = 0.0
        else:
            # 하단 카메라: norm_z >= -0.5 이면 손날 (반대로 적용)
            threshold = -0.5
            is_side = norm_z >= threshold

            # 신뢰도 계산: threshold보다 클수록 신뢰도 증가
            if norm_z >= threshold:
                confidence = min(1.0, max(0.0, (norm_z - threshold) / 0.7))
            else:
                confidence = 0.0

        return is_side, norm_z, confidence

    except Exception:
        return False, 0.0, 0.0


def is_thumb_extended(hand_landmarks, handedness):
    mcp = hand_landmarks.landmark[2]
    tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    ring_mcp = hand_landmarks.landmark[13]
    pinky_mcp = hand_landmarks.landmark[17]

    palm_cx = (index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 4
    palm_cy = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4

    dist_tip_palm = math.hypot(tip.x - palm_cx, tip.y - palm_cy)
    dist_mcp_tip = math.hypot(tip.x - mcp.x, tip.y - mcp.y)

    angle = calculate_angle(
        (mcp.x, mcp.y),
        (hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y),
        (tip.x, tip.y),
    )

    if dist_tip_palm < dist_mcp_tip * 0.8:
        return False

    index_pip = hand_landmarks.landmark[6]
    thumb_index_distance = math.hypot(tip.x - index_pip.x, tip.y - index_pip.y)
    palm_width = math.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y)

    if thumb_index_distance < palm_width * 0.8:
        return False

    if angle > 145 and angle < 180:
        if handedness == "Right":
            return tip.x > mcp.x and tip.x > hand_landmarks.landmark[1].x
        else:
            return tip.x < mcp.x and tip.x < hand_landmarks.landmark[1].x
    return False


def calculate_angle(a, b, c):
    """a, b, c는 (x, y) 튜플. b는 각도의 꼭짓점"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-6
    )
    angle = math.acos(cosine_angle)
    return math.degrees(angle)


def calculate_angle_3d(a, b, c):
    """a, b, c는 (x, y, z) 튜플. b는 각도의 꼭짓점 - 3D 벡터 계산"""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc + 1e-6)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle = math.acos(cosine_angle)
    return math.degrees(angle)


def calculate_thumb_spread_angle(hand_landmarks, handedness):
    """엄지 관절 각도 계산"""
    mcp = hand_landmarks.landmark[2]
    ip = hand_landmarks.landmark[3]
    tip = hand_landmarks.landmark[4]

    joint_angle = calculate_angle_3d(
        (mcp.x, mcp.y, mcp.z), (ip.x, ip.y, ip.z), (tip.x, tip.y, tip.z)
    )

    index_mcp = hand_landmarks.landmark[5]
    thumb_index_distance = math.sqrt(
        (tip.x - index_mcp.x) ** 2
        + (tip.y - index_mcp.y) ** 2
        + (tip.z - index_mcp.z) ** 2
    )

    wrist = hand_landmarks.landmark[0]
    hand_size = math.sqrt(
        (index_mcp.x - wrist.x) ** 2
        + (index_mcp.y - wrist.y) ** 2
        + (index_mcp.z - wrist.z) ** 2
    )

    normalized_spread = thumb_index_distance / (hand_size + 1e-6)

    if normalized_spread > 1.2:
        return +joint_angle
    elif normalized_spread < 0.8:
        return -(joint_angle * 1.5)
    else:
        return +(joint_angle * 0.7)


def finger_angle(hand_landmarks, mcp_id, pip_id, tip_id):
    mcp = hand_landmarks.landmark[mcp_id]
    pip = hand_landmarks.landmark[pip_id]
    tip = hand_landmarks.landmark[tip_id]
    return calculate_angle((mcp.x, mcp.y), (pip.x, pip.y), (tip.x, tip.y))


def classify_thumb_state_side(hand_landmarks, camera_type="side", handedness="Right"):
    """
    측면/하단 카메라에서 엄지 상태 분류: 1(straight) / 0(between) / -1(bent)

    Args:
        hand_landmarks: MediaPipe 손 랜드마크
        camera_type: "side" 또는 "bottom"

    Returns:
        1: straight (펴짐), 0: between (중간), -1: bent (굽힘)
    """
    # 엄지 랜드마크
    thumb_tip = hand_landmarks.landmark[4]  # 끝

    # 검지 MCP (비교 기준)
    index_mcp = hand_landmarks.landmark[5]

    if camera_type == "side":
        # 측면 카메라: X축 거리로 판단 (엄지가 펴지면 X가 커짐)
        # 측면 카메라: 간단히 펴짐으로 간주
        return 1, None  # straight (무조건 펴짐), normalized_y는 None
    else:
        # 하단 카메라: tracking.py와 동일한 normalized 좌표 기반 zone 판정 + In1/In2/In3 세분화

        # 손 크기 기준 계산
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        pinky_mcp = hand_landmarks.landmark[17]

        hand_length = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        palm_width = math.hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y)

        # 손바닥 중심 계산
        palm_center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
        palm_center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3

        # 엄지 끝에서 손바닥 중심까지의 벡터 (정규화)
        thumb_vector_x = (thumb_tip.x - palm_center_x) / (palm_width + 1e-6)
        thumb_vector_y = (thumb_tip.y - palm_center_y) / (hand_length + 1e-6)

        # 왼손/오른손에 따른 조정
        if handedness == "Left":
            thumb_vector_x = -thumb_vector_x

        normalized_x = thumb_vector_x
        normalized_y = thumb_vector_y

        # Zone 판정 임계값 (tracking.py와 동일)
        THUMB_INNER_THRESHOLD = 0.54
        THUMB_OUTER_THRESHOLD = 1.4
        INNER_Y_HIGH_THRESHOLD = 0.55  # >= 0.55: In3
        INNER_Y_LOW_THRESHOLD = 0.27  # < 0.27: In1, 0.27~0.55: In2

        # Zone 판별
        thumb_zone = "center"

        if normalized_x <= THUMB_INNER_THRESHOLD:
            thumb_zone = "inner"
            # Inner State 세분화 (Y값 기준)
            # (서브존 정보는 여기서는 판정에 사용하지 않음)
            if normalized_y >= INNER_Y_HIGH_THRESHOLD:
                pass
            elif normalized_y >= INNER_Y_LOW_THRESHOLD:
                pass
            else:
                pass
        elif normalized_x >= THUMB_OUTER_THRESHOLD:
            thumb_zone = "outer"

        # Zone별 straight/bent 판정 (In1/In2/In3 세분화 적용)
        if thumb_zone == "outer":
            return 1, normalized_y  # straight
        elif thumb_zone == "inner":
            return -1, normalized_y  # bent
        else:
            return 0, normalized_y  # between


def classify_finger_state_single_angle(
    angle_side,
    angle_bottom=None,
    y_pos_side=None,
    y_pos_bottom=None,
    is_side_facing=False,
    finger_name=None,
):
    """
    싱글 각도 손가락 상태 분류: 1(straight) / 0(between) / -1(bent)

    핵심 로직:
    1. 측면 카메라 각도로 1차 판정 (MCP-PIP-TIP만 사용)
    2. 하단 카메라 있으면 두 카메라 융합
    3. Lower 각도는 사용하지 않음 (싱글 각도만)

    Args:
        angle_side: 측면 카메라 손가락 각도 (MCP-PIP-TIP)
        angle_bottom: 하단 카메라 손가락 각도 (optional)
        y_pos_side: 측면 카메라에서 손가락 끝 Y 위치 (optional)
        y_pos_bottom: 하단 카메라에서 손가락 끝 Y 위치 (optional)
        is_side_facing: 손날 상태 여부 (optional)
        finger_name: 손가락 이름 (optional)

    Returns:
        1: straight (펴짐), 0: between (중간), -1: bent (굽힘)
    """
    # 손가락별 임계값 (싱글 각도 전용)
    if finger_name in ["Middle", "Ring"]:
        straight_threshold = (
            169 if finger_name == "Middle" else 160
        )  # 중지: 169도, 약지: 160도 이상만 Straight
        bent_threshold = 90 if is_side_facing else 55  # 55도 이하만 Bent
    elif finger_name == "Pinky":
        # 소지: 160도 이상만 Straight
        straight_threshold = 160
        bent_threshold = 90 if is_side_facing else 50
    else:
        # 검지: 기본값
        straight_threshold = 165
        bent_threshold = 90 if is_side_facing else 50

    # 🎯 Step 1: 측면 카메라로 1차 판정 (싱글 각도만)
    if angle_side >= straight_threshold:
        side_state = 1  # straight
    elif angle_side <= bent_threshold:
        side_state = -1  # bent
    else:
        side_state = 0  # between

    # 🎯 Step 2: 하단 카메라 융합 (싱글 각도만)
    if angle_bottom is not None:
        # 하단 카메라도 같은 임계값 적용
        bottom_straight_threshold = 160 if finger_name == "Pinky" else 170
        if angle_bottom >= bottom_straight_threshold:
            bottom_state = 1  # straight
        elif angle_bottom <= 50:
            bottom_state = -1  # bent
        else:
            bottom_state = 0  # between

        # 두 카메라 의견이 일치하면 그대로 사용
        if side_state == bottom_state:
            final_state = side_state
        # BENT 판정 시 하단 카메라 우선
        elif bottom_state == -1:
            final_state = -1  # 하단 카메라가 BENT면 BENT로 결정
        elif side_state == -1 and bottom_state != -1:
            final_state = -1  # 측면 카메라가 BENT면 BENT로 결정
        # 의견이 다른 경우
        # straight <-> between: between 우선 (애매하면 중간으로)
        elif (side_state == 1 and bottom_state == 0) or (
            side_state == 0 and bottom_state == 1
        ):
            final_state = 0
        # between <-> bent: between 우선
        elif side_state == 0 and bottom_state == -1:
            final_state = 0  # Between으로 판정
        elif side_state == -1 and bottom_state == 0:
            final_state = 0  # Between으로 판정
        # straight <-> bent: between으로 보정
        elif side_state == 1 and bottom_state == -1:
            final_state = 0  # 애매하면 중간
        elif side_state == -1 and bottom_state == 1:
            final_state = 0  # 애매하면 중간
        else:
            final_state = side_state
    else:
        # Bottom 정보 없으면 측면만으로 판단
        final_state = side_state

    # 🎯 싱글 각도만 사용하므로 바로 반환 (Lower 각도 처리 없음)
    return final_state


def classify_finger_state_3way_side(
    angle_side,
    angle_bottom=None,
    y_pos_side=None,
    y_pos_bottom=None,
    is_side_facing=False,
    finger_name=None,
):
    """
    3단계 손가락 상태 분류 (측면 카메라용): 1(straight) / 0(between) / -1(bent)

    Args:
        angle_side: 측면 카메라 손가락 각도
        angle_bottom: 하단 카메라 손가락 각도 (optional)
        y_pos_side: 측면 카메라에서 손가락 끝 Y 위치 (optional)
        y_pos_bottom: 하단 카메라에서 손가락 끝 Y 위치 (optional)
        is_side_facing: 손날 상태 여부 (optional, default: False)
        finger_name: 손가락 이름 (optional) - 손가락별 임계값 적용

    Returns:
        1: straight (펴짐), 0: between (중간), -1: bent (굽힘)
    """
    # 손가락별로 다른 임계값 적용
    # 중지/약지는 카메라 각도에 민감하므로 더 관대한 임계값 사용
    if finger_name in ["Middle", "Ring"]:
        straight_threshold = (
            169 if finger_name == "Middle" else 160
        )  # 중지: 169도, 약지: 160도 이상만 Straight
        bent_threshold = 90 if is_side_facing else 55  # 55도 이하만 Bent
    else:
        # 검지/소지: 기본값
        straight_threshold = 165
        bent_threshold = 90 if is_side_facing else 50

    if angle_side >= straight_threshold:
        side_state = 1  # straight
    elif angle_side <= bent_threshold:
        side_state = -1  # bent
    else:
        side_state = 0  # between

    if angle_bottom is not None:
        # 하단 카메라도 Between 범위 확대
        if angle_bottom >= 170:
            bottom_state = 1  # straight
        elif angle_bottom <= 50:
            bottom_state = -1  # bent
        else:
            bottom_state = 0  # between

        # 두 카메라 의견이 일치하면 그대로 반환
        if side_state == bottom_state:
            return side_state

        # 의견이 다른 경우
        # straight <-> between: between 우선 (애매하면 중간으로)
        if (side_state == 1 and bottom_state == 0) or (
            side_state == 0 and bottom_state == 1
        ):
            return 0
        # between <-> bent: between 우선 (Between 범위를 넓혔으므로 Between 신뢰)
        elif side_state == 0 and bottom_state == -1:
            return 0  # Between으로 판정
        elif side_state == -1 and bottom_state == 0:
            return 0  # Between으로 판정
        # straight <-> bent: between으로 보정 (극단적 차이는 중간으로)
        elif side_state == 1 and bottom_state == -1:
            return 0  # 애매하면 중간
        elif side_state == -1 and bottom_state == 1:
            return 0  # 애매하면 중간

    # Bottom 정보 없으면 측면만으로 판단
    return side_state


def classify_thumb_position(camera_state):
    """하단 카메라의 저장된 팁 좌표와 정규화 값으로 엄지 위치를 규칙 기반(E/N/M)으로 판정합니다.

    반환값: (code, text) where code: 0=NEUTRAL, 1=ON_TOP, 2=BETWEEN
    간단 규칙:
      - nd_middle <= 0.08 -> ON_TOP
      - nd_index <= 0.06 or nd_ring <= 0.06 -> BETWEEN
      - otherwise -> NEUTRAL

    카메라 상태에 palm_width_pixels 또는 픽셀 거리가 없을 경우 픽셀 기준 값으로 대체합니다.
    안정화를 위해 같은 결과가 3프레임 연속일 때만 카운트가 증가합니다n"""
    # 기본
    NEUTRAL = 0
    ON_TOP = 1
    BETWEEN = 2

    # 필요한 값 읽기
    tt = getattr(camera_state, "thumb_tip_coords", None)
    it = getattr(camera_state, "index_tip_coords", None)
    mt = getattr(camera_state, "middle_tip_coords", None)
    rt = getattr(camera_state, "ring_tip_coords", None)
    palm_w = getattr(camera_state, "palm_width_pixels", None)

    if not (tt and it and mt and rt):
        return NEUTRAL, "NEUTRAL"

    tx, ty = tt
    ix, iy = it
    mx, my = mt
    rx, ry = rt

    d_index = math.hypot(tx - ix, ty - iy)
    d_middle = math.hypot(tx - mx, ty - my)
    d_ring = math.hypot(tx - rx, ty - ry)

    # 정규화 거리 계산
    nd_index = nd_middle = nd_ring = None
    if palm_w and palm_w > 1e-6:
        nd_index = d_index / palm_w
        nd_middle = d_middle / palm_w
        nd_ring = d_ring / palm_w

    # 규칙 적용 (정규화 값 우선)
    result = NEUTRAL
    result_text = "NEUTRAL"
    if nd_middle is not None:
        if nd_middle <= 0.08:
            result = ON_TOP
            result_text = "ON_TOP"
        elif (nd_index is not None and nd_index <= 0.06) or (
            nd_ring is not None and nd_ring <= 0.06
        ):
            result = BETWEEN
            result_text = "BETWEEN"
        else:
            result = NEUTRAL
            result_text = "NEUTRAL"
    else:
        # 픽셀 값으로 대체 조건
        if d_middle <= 25:
            result = ON_TOP
            result_text = "ON_TOP"
        elif d_index <= 20 or d_ring <= 20:
            result = BETWEEN
            result_text = "BETWEEN"
        else:
            result = NEUTRAL
            result_text = "NEUTRAL"

    # 안정성(3프레임) 카운터 관리
    last = getattr(camera_state, "last_thumb_position", None)
    count = getattr(camera_state, "thumb_position_count", 0)
    if last == result_text:
        count = min(count + 1, 10)
    else:
        count = 1
    camera_state.last_thumb_position = result_text
    camera_state.thumb_position_count = count

    # 실제 반환은 count >= 1 이면 바로 반환 (짧은 안정성 적용)
    return result, result_text


def is_thumb_between_fingers(camera_state):
    """엄지가 손가락 사이에 있는지 판정: z 값과 x,y 위치를 함께 사용."""
    tt = getattr(camera_state, "thumb_tip_coords", None)
    it = getattr(camera_state, "index_tip_coords", None)
    mt = getattr(camera_state, "middle_tip_coords", None)
    rt = getattr(camera_state, "ring_tip_coords", None)

    if not (tt and it and mt and rt):
        return False

    tx, ty = tt
    ix, iy = it
    mx, my = mt
    rx, ry = rt

    # check x between index and ring (loose check)
    min_x = min(ix, rx)
    max_x = max(ix, rx)
    in_x_band = min_x - 5 <= tx <= max_x + 5

    # use z-values: thumb_tip_z closer to finger tips z (i.e., between) -> small relative
    rel = getattr(camera_state, "thumb_rel_to_fingertips_norm", None)
    # rel ~ 0 => same plane; negative => thumb closer to camera (smaller z) depending on MP coord
    z_close = False
    if rel is not None:
        # threshold: abs(rel) < 0.03 considered same plane
        z_close = abs(rel) <= 0.03

    return in_x_band and z_close


def check_thumb_between_fingers_side(camera_state, hand_landmarks, w, h):
    """측면 카메라에서 엄지가 손가락 사이에 끼어있는지 판정.

    PIP(두 번째 마디) 기반 세그먼트 정의:
    - 엄지 TIP이 검지 PIP - 중지 PIP 사이: IM (검지-중지 사이) -> T
    - 엄지 TIP이 중지 PIP - 약지 PIP 사이: MR (중지-약지 사이) -> N
    - 엄지 TIP이 약지 PIP - 소지 PIP 사이: RP (약지-소지 사이) -> M

    Args:
        camera_state: 카메라 상태 객체
        hand_landmarks: MediaPipe 손 랜드마크
        w, h: 이미지 너비, 높이

    Returns:
        (is_between: bool, segment: str, details: dict)
    """
    try:
        # Thumb TIP
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # Finger PIPs (두 번째 마디)
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

        # Finger TIPs (끝 마디) - X축 범위 체크용
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # PIP 좌표
        ix_pip, iy_pip = int(index_pip.x * w), int(index_pip.y * h)
        mx_pip, my_pip = int(middle_pip.x * w), int(middle_pip.y * h)
        rx_pip, ry_pip = int(ring_pip.x * w), int(ring_pip.y * h)
        px_pip, py_pip = int(pinky_pip.x * w), int(pinky_pip.y * h)

        # TIP 좌표 (X축 범위용)
        ix_tip, iy_tip = int(index_tip.x * w), int(index_tip.y * h)
        mx_tip, my_tip = int(middle_tip.x * w), int(middle_tip.y * h)
        rx_tip, ry_tip = int(ring_tip.x * w), int(ring_tip.y * h)
        px_tip, py_tip = int(pinky_tip.x * w), int(pinky_tip.y * h)

    except Exception as e:
        return False, "ERROR", {"error": str(e)}

    # PIP 기반 세그먼트 Y 범위 정의
    im_y_min = min(iy_pip, my_pip)
    im_y_max = max(iy_pip, my_pip)
    mr_y_min = min(my_pip, ry_pip)
    mr_y_max = max(my_pip, ry_pip)
    rp_y_min = min(ry_pip, py_pip)
    rp_y_max = max(ry_pip, py_pip)

    # 세그먼트 Y 중심점 계산
    im_y_center = (im_y_min + im_y_max) / 2
    mr_y_center = (mr_y_min + mr_y_max) / 2
    rp_y_center = (rp_y_min + rp_y_max) / 2

    # 여유 범위 (PIP 간 거리의 일정 비율로 설정)
    im_height = abs(my_pip - iy_pip)
    mr_height = abs(ry_pip - my_pip)
    rp_height = abs(py_pip - ry_pip)

    # 최소 높이 설정 (0으로 나누기 방지)
    MIN_HEIGHT = 5.0  # 최소 5 픽셀
    if im_height < MIN_HEIGHT:
        im_height = MIN_HEIGHT
    if mr_height < MIN_HEIGHT:
        mr_height = MIN_HEIGHT
    if rp_height < MIN_HEIGHT:
        rp_height = MIN_HEIGHT

    # 여유 범위 (PIP 간 거리의 일정 비율로 설정)
    # IM(T 제스처)는 MR과 구분을 위해 마진을 더 넓게 설정
    # RP(M 제스처)는 끼우기 어려우므로 마진을 더 크게 설정
    im_margin = im_height * 0.4  # 40% 여유 (T 제스처 안정성 향상)
    mr_margin = mr_height * 0.3  # 30% 여유
    rp_margin = rp_height * 0.5  # 50% 여유 (M 제스처 안정성 향상)

    # X축 체크: 엄지가 손가락들 TIP 근처에 있어야 함
    fingers_x_min = min(ix_tip, mx_tip, rx_tip, px_tip)
    fingers_x_max = max(ix_tip, mx_tip, rx_tip, px_tip)
    x_margin = 40  # 픽셀 단위
    in_x_range = (fingers_x_min - x_margin) <= tx <= (fingers_x_max + x_margin)

    if not in_x_range:
        return (
            False,
            "OUT_OF_X_RANGE",
            {"tx": tx, "x_range": (fingers_x_min, fingers_x_max), "margin": x_margin},
        )

    # 모든 세그먼트 후보를 수집하고 중심으로부터 가장 가까운 것을 선택
    candidates = []

    # IM 세그먼트 체크 (검지 PIP - 중지 PIP 사이)
    if (im_y_min - im_margin) <= ty <= (im_y_max + im_margin):
        distance_from_center = abs(ty - im_y_center)
        denominator = im_height / 2 + im_margin
        confidence = (
            1.0 - (distance_from_center / denominator) if denominator > 0 else 0.0
        )
        confidence = max(0.0, min(1.0, confidence))

        candidates.append(
            {
                "segment": "IM",
                "distance": distance_from_center,
                "confidence": confidence,
                "details": {
                    "ty": ty,
                    "segment_y_range": (im_y_min, im_y_max),
                    "segment_y_center": im_y_center,
                    "distance_from_center": distance_from_center,
                    "confidence": confidence,
                    "pip_coords": {
                        "index": (ix_pip, iy_pip),
                        "middle": (mx_pip, my_pip),
                    },
                },
            }
        )

    # MR 세그먼트 체크 (중지 PIP - 약지 PIP 사이)
    if (mr_y_min - mr_margin) <= ty <= (mr_y_max + mr_margin):
        distance_from_center = abs(ty - mr_y_center)
        denominator = mr_height / 2 + mr_margin
        confidence = (
            1.0 - (distance_from_center / denominator) if denominator > 0 else 0.0
        )
        confidence = max(0.0, min(1.0, confidence))

        candidates.append(
            {
                "segment": "MR",
                "distance": distance_from_center,
                "confidence": confidence,
                "details": {
                    "ty": ty,
                    "segment_y_range": (mr_y_min, mr_y_max),
                    "segment_y_center": mr_y_center,
                    "distance_from_center": distance_from_center,
                    "confidence": confidence,
                    "pip_coords": {
                        "middle": (mx_pip, my_pip),
                        "ring": (rx_pip, ry_pip),
                    },
                },
            }
        )

    # RP 세그먼트 체크 (약지 PIP - 소지 PIP 사이)
    if (rp_y_min - rp_margin) <= ty <= (rp_y_max + rp_margin):
        distance_from_center = abs(ty - rp_y_center)
        denominator = rp_height / 2 + rp_margin
        confidence = (
            1.0 - (distance_from_center / denominator) if denominator > 0 else 0.0
        )
        confidence = max(0.0, min(1.0, confidence))

        candidates.append(
            {
                "segment": "RP",
                "distance": distance_from_center,
                "confidence": confidence,
                "details": {
                    "ty": ty,
                    "segment_y_range": (rp_y_min, rp_y_max),
                    "segment_y_center": rp_y_center,
                    "distance_from_center": distance_from_center,
                    "confidence": confidence,
                    "pip_coords": {"ring": (rx_pip, ry_pip), "pinky": (px_pip, py_pip)},
                },
            }
        )

    # 후보가 있으면 중심으로부터 가장 가까운 것 선택
    if candidates:
        # 중심으로부터의 거리가 가장 짧은 세그먼트 선택
        best = min(candidates, key=lambda x: x["distance"])
        return (True, best["segment"], best["details"])

    # 어느 범위에도 해당하지 않음
    return (
        False,
        "NONE",
        {
            "ty": ty,
            "im_range": (im_y_min - im_margin, im_y_max + im_margin),
            "mr_range": (mr_y_min - mr_margin, mr_y_max + mr_margin),
            "rp_range": (rp_y_min - rp_margin, rp_y_max + rp_margin),
            "im_center": im_y_center,
            "mr_center": mr_y_center,
            "rp_center": rp_y_center,
        },
    )


def is_thumb_between_fingers_3d(camera_state):
    """손가락 사이 판정: 2D로 엄지와 각 손가락 선분의 최단거리 투영을 구하고,
    해당 투영 위치의 z를 선형 보간하여 엄지와의 z 차이를 확인합니다.

    3개 세그먼트 체크:
    - IM: Index-Middle (검지-중지 사이) -> T 제스처
    - MR: Middle-Ring (중지-약지 사이) -> N 제스처
    - RP: Ring-Pinky (약지-소지 사이) -> M 제스처

    반환: (between_bool, details_dict)
    details_dict: { 'seg': 'IM'/'MR'/'RP', 't': t, 'nd': normalized_2d_dist, 'nz': normalized_z_diff }
    """
    tt = getattr(camera_state, "thumb_tip_coords", None)
    it = getattr(camera_state, "index_tip_coords", None)
    mt = getattr(camera_state, "middle_tip_coords", None)
    rt = getattr(camera_state, "ring_tip_coords", None)
    pt = getattr(camera_state, "pinky_tip_coords", None)

    tz = getattr(camera_state, "thumb_tip_z", None)
    iz = getattr(camera_state, "index_tip_z", None)
    mz = getattr(camera_state, "middle_tip_z", None)
    rz = getattr(camera_state, "ring_tip_z", None)
    pz = getattr(camera_state, "pinky_tip_z", None)

    palm_w = getattr(camera_state, "palm_width_pixels", None)
    hand_size_3d = getattr(camera_state, "hand_size_3d", None)

    # 필요 데이터 없으면 이전 간단 판정으로 폴백
    if not (tt and it and mt and rt and pt):
        return is_thumb_between_fingers(camera_state), {"reason": "missing_2d"}

    # 헬퍼: 2D point-to-segment projection
    def proj_t_and_dist(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom < 1e-6:
            return 0.0, math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t_clamped = max(0.0, min(1.0, t))
        projx = ax + t_clamped * dx
        projy = ay + t_clamped * dy
        dist = math.hypot(px - projx, py - projy)
        return t_clamped, dist

    thumb_p = tt

    # Check 3 finger-tip segments
    im_t, im_d = proj_t_and_dist(thumb_p, it, mt)  # Index-Middle
    mr_t, mr_d = proj_t_and_dist(thumb_p, mt, rt)  # Middle-Ring
    rp_t, rp_d = proj_t_and_dist(thumb_p, rt, pt)  # Ring-Pinky

    # Normalize distances
    if palm_w and palm_w > 1e-6:
        nd_im = im_d / palm_w
        nd_mr = mr_d / palm_w
        nd_rp = rp_d / palm_w
    else:
        nd_im = im_d
        nd_mr = mr_d
        nd_rp = rp_d

    # Interpolate z at projection if available
    def interp_z(t, z1, z2):
        if z1 is None or z2 is None:
            return None
        return z1 + t * (z2 - z1)

    im_interp_z = interp_z(im_t, iz, mz)
    mr_interp_z = interp_z(mr_t, mz, rz)
    rp_interp_z = interp_z(rp_t, rz, pz)

    # Calculate normalized z differences
    nz_im = None
    nz_mr = None
    nz_rp = None

    if (
        tz is not None
        and im_interp_z is not None
        and hand_size_3d
        and hand_size_3d > 1e-6
    ):
        nz_im = (tz - im_interp_z) / hand_size_3d
    if (
        tz is not None
        and mr_interp_z is not None
        and hand_size_3d
        and hand_size_3d > 1e-6
    ):
        nz_mr = (tz - mr_interp_z) / hand_size_3d
    if (
        tz is not None
        and rp_interp_z is not None
        and hand_size_3d
        and hand_size_3d > 1e-6
    ):
        nz_rp = (tz - rp_interp_z) / hand_size_3d

    # Thresholds (경험적): nd < 0.06 and abs(nz) < 0.04 and t in [0,1]
    between = False
    chosen = None

    # Check all three segments (우선순위: IM -> MR -> RP)
    if (
        nd_im is not None
        and nd_im <= 0.06
        and nz_im is not None
        and abs(nz_im) <= 0.04
        and 0.0 <= im_t <= 1.0
    ):
        between = True
        chosen = ("IM", im_t, nd_im, nz_im)
    elif (
        nd_mr is not None
        and nd_mr <= 0.06
        and nz_mr is not None
        and abs(nz_mr) <= 0.04
        and 0.0 <= mr_t <= 1.0
    ):
        between = True
        chosen = ("MR", mr_t, nd_mr, nz_mr)
    elif (
        nd_rp is not None
        and nd_rp <= 0.06
        and nz_rp is not None
        and abs(nz_rp) <= 0.04
        and 0.0 <= rp_t <= 1.0
    ):
        between = True
        chosen = ("RP", rp_t, nd_rp, nz_rp)

    if not between:
        # fallback: use simple rel check, but still show nd/nz values from closest segment
        simple = is_thumb_between_fingers(camera_state)
        between = simple

        # 가장 가까운 세그먼트 선택 (nd 값 기준)
        candidates = []
        if nd_im is not None:
            candidates.append(("fallback_IM", im_t, nd_im, nz_im, nd_im))
        if nd_mr is not None:
            candidates.append(("fallback_MR", mr_t, nd_mr, nz_mr, nd_mr))
        if nd_rp is not None:
            candidates.append(("fallback_RP", rp_t, nd_rp, nz_rp, nd_rp))

        if candidates:
            # nd 값이 가장 작은 세그먼트 선택
            closest = min(candidates, key=lambda x: x[4])
            chosen = closest[:4]  # (seg, t, nd, nz)
        else:
            chosen = ("fallback", 0.0, None, None)

    details = {
        "seg": chosen[0] if chosen else "none",
        "t": chosen[1] if chosen else 0.0,
        "nd": chosen[2] if chosen else None,
        "nz": chosen[3] if chosen else None,
    }
    return between, details


def check_index_middle_distance(camera_state, hand_size_3d=None):
    """
    검지와 중지 TIP 사이의 거리를 계산하여 붙어있는지/떨어져있는지 판별합니다.

    Args:
        camera_state: 카메라 상태 (index_tip_coords, middle_tip_coords 필요)
        hand_size_3d: 손 크기 (정규화용, 없으면 픽셀 거리 사용)

    Returns:
        tuple: (is_together: bool, distance: float, normalized_distance: float or None)
            - is_together: True면 붙어있음, False면 떨어져있음
            - distance: 픽셀 단위 거리
            - normalized_distance: 손 크기로 정규화된 거리 (hand_size_3d 있을 때만)
    """
    global INDEX_MIDDLE_DISTANCE_THRESHOLD

    it = getattr(camera_state, "index_tip_coords", None)
    mt = getattr(camera_state, "middle_tip_coords", None)

    if it is None or mt is None:
        return None, None, None

    # 유클리드 거리 계산
    ix, iy = it
    mx, my = mt
    distance = math.sqrt((ix - mx) ** 2 + (iy - my) ** 2)

    # 손 크기로 정규화 (있으면)
    normalized_distance = None
    if hand_size_3d is not None and hand_size_3d > 0:
        normalized_distance = distance / hand_size_3d

    # 임계값 설정 (정규화된 거리가 있으면 사용, 없으면 픽셀 거리)
    if normalized_distance is not None:
        # 정규화된 거리 기준: 전역 임계값 사용
        is_together = normalized_distance <= INDEX_MIDDLE_DISTANCE_THRESHOLD
    else:
        # 픽셀 거리 기준: 30 픽셀 이하면 붙어있음 (fallback)
        threshold = 30.0
        is_together = distance <= threshold

    return is_together, distance, normalized_distance


def check_extended_fingers_together(camera_state, finger_states, hand_size_3d=None):
    """
    펴진(straight=1) 손가락들이 모두 붙어있는지 확인합니다.

    Args:
        camera_state: 카메라 상태 (각 손가락 tip_coords 필요)
        finger_states: 손가락 상태 dict {finger_name: state}
        hand_size_3d: 손 크기 (정규화용, 선택적)

    Returns:
        dict: {
            "all_together": bool,  # 모든 펴진 손가락이 붙어있으면 True
            "extended_fingers": list,  # 펴진 손가락 리스트
            "pairwise_distances": dict,  # 각 쌍의 거리
            "separated_pairs": list  # 떨어진 쌍 리스트
        }
    """
    # 1. 펴진 손가락(state=1) 찾기 (엄지 제외)
    extended_fingers = [
        name
        for name in ["Index", "Middle", "Ring", "Pinky"]
        if finger_states.get(name) == 1
    ]

    # 2. 펴진 손가락이 2개 미만이면 체크 불필요 (항상 붙어있다고 간주)
    if len(extended_fingers) < 2:
        return {
            "all_together": True,
            "extended_fingers": extended_fingers,
            "pairwise_distances": {},
            "separated_pairs": [],
        }

    # 3. 모든 인접 손가락 쌍의 거리 계산
    threshold = 50.0  # 픽셀 거리 임계값 (모든 쌍 통일)
    pairwise_distances = {}
    separated_pairs = []

    finger_to_attr = {
        "Index": "index_tip_coords",
        "Middle": "middle_tip_coords",
        "Ring": "ring_tip_coords",
        "Pinky": "pinky_tip_coords",
    }

    for i in range(len(extended_fingers) - 1):
        finger1 = extended_fingers[i]
        finger2 = extended_fingers[i + 1]

        pair_key = f"{finger1}-{finger2}"

        # Ring-Pinky 쌍은 특별 처리: Ring DIP과 Pinky TIP 비교
        if pair_key == "Ring-Pinky":
            ring_dip_coords = getattr(camera_state, "ring_dip_coords", None)
            pinky_tip_coords = getattr(camera_state, "pinky_tip_coords", None)

            if ring_dip_coords and pinky_tip_coords:
                x1, y1 = ring_dip_coords
                x2, y2 = pinky_tip_coords
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                pairwise_distances[pair_key] = distance
                pair_threshold = 50.0

                if distance > pair_threshold:
                    separated_pairs.append(pair_key)
        else:
            # 다른 쌍은 TIP 대 TIP 비교
            coords1 = getattr(camera_state, finger_to_attr[finger1], None)
            coords2 = getattr(camera_state, finger_to_attr[finger2], None)

            if coords1 and coords2:
                x1, y1 = coords1
                x2, y2 = coords2
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                pairwise_distances[pair_key] = distance
                pair_threshold = threshold

                # 임계값보다 크면 떨어진 것으로 판정
                if distance > pair_threshold:
                    separated_pairs.append(pair_key)

    # 4. 모든 인접 쌍이 붙어있으면 True
    all_together = len(separated_pairs) == 0

    return {
        "all_together": all_together,
        "extended_fingers": extended_fingers,
        "pairwise_distances": pairwise_distances,
        "separated_pairs": separated_pairs,
    }


def check_thumb_touching_fingers(camera_state, hand_size_3d=None):
    """
    엄지 TIP이 검지/중지/약지/소지 TIP과 접촉했는지 감지합니다.

    Args:
        camera_state: 카메라 상태 (thumb_tip_coords, 다른 손가락 tip_coords 필요)
        hand_size_3d: 손 크기 (정규화용)

    Returns:
        dict: {
            "touching": bool (하나라도 접촉하면 True),
            "touched_finger": str or None (접촉한 손가락 이름),
            "distances": dict (각 손가락별 정규화된 거리),
            "min_distance": float (가장 가까운 거리)
        }
    """
    global THUMB_TOUCH_THRESHOLD

    tt = getattr(camera_state, "thumb_tip_coords", None)
    it = getattr(camera_state, "index_tip_coords", None)
    mt = getattr(camera_state, "middle_tip_coords", None)
    rt = getattr(camera_state, "ring_tip_coords", None)
    pt = getattr(camera_state, "pinky_tip_coords", None)

    if tt is None:
        return {
            "touching": False,
            "touched_finger": None,
            "distances": {},
            "min_distance": None,
        }

    tx, ty = tt
    fingers = {"Index": it, "Middle": mt, "Ring": rt, "Pinky": pt}

    distances = {}
    normalized_distances = {}

    # 각 손가락과의 거리 계산
    for finger_name, finger_coords in fingers.items():
        if finger_coords is not None:
            fx, fy = finger_coords
            distance = math.sqrt((tx - fx) ** 2 + (ty - fy) ** 2)
            distances[finger_name] = distance
            # 픽셀 거리를 직접 사용 (정규화하지 않음)
            normalized_distances[finger_name] = distance

    # 가장 가까운 손가락 찾기
    touching = False
    touched_finger = None
    min_distance = None

    if normalized_distances:
        # 거리가 있는 손가락들만 확인
        valid_fingers = {k: v for k, v in normalized_distances.items() if v is not None}

        if valid_fingers:
            # 가장 가까운 손가락
            closest_finger = min(valid_fingers.items(), key=lambda x: x[1])
            touched_finger = closest_finger[0]
            min_distance = closest_finger[1]

            # 임계값 이하면 접촉으로 판정 (픽셀 거리 기준)
            if min_distance <= THUMB_TOUCH_THRESHOLD:
                touching = True

    return {
        "touching": touching,
        "touched_finger": touched_finger if touching else None,
        "distances": normalized_distances,
        "min_distance": min_distance,
    }


def is_weapon_gesture(gesture_name):
    """무기 제스처인지 확인하는 헬퍼 함수 (즉시 전송용 - Fire/Reload만)"""
    if gesture_name is None:
        return False
    # SG와 S1은 일반 제스처처럼 안정화 시간 적용
    weapon_keywords = ["Fire", "Reload"]
    return any(keyword in gesture_name for keyword in weapon_keywords)


def classify_gesture_from_pattern_stabilized(
    integrated_states, bottom_camera_state, side_camera_state
):
    """
    손가락 상태 패턴으로 제스처를 분류합니다.
    0.5초 동안 동일한 자세가 유지되면 해당 제스처로 판단합니다.
    """
    global current_gesture_candidate, gesture_start_time, stable_gesture

    # 현재 제스처 후보 계산
    candidate_gesture = classify_gesture_from_pattern(
        integrated_states, bottom_camera_state, side_camera_state
    )

    current_time = time.time()

    # 새로운 제스처 후보가 감지되었을 때
    if candidate_gesture != current_gesture_candidate:
        current_gesture_candidate = candidate_gesture
        gesture_start_time = current_time

        # 무기 제스처(Fire/Reload)는 즉시 전송
        if is_weapon_gesture(candidate_gesture):
            send_gesture_to_unity(candidate_gesture)
            return candidate_gesture
        else:
            # 일반 제스처는 안정화 필요
            send_no_gesture_to_unity()
            return None

    # 동일한 제스처 후보가 계속 유지되고 있을 때
    if candidate_gesture is not None and gesture_start_time is not None:
        time_elapsed = current_time - gesture_start_time

        # 무기 제스처는 계속 전송 (안정화 시간 무관)
        if is_weapon_gesture(candidate_gesture):
            send_gesture_to_unity(candidate_gesture)
            return candidate_gesture

        # 일반 제스처는 0.3초 이상 유지되면 안정된 제스처로 판단
        if time_elapsed >= GESTURE_STABILIZATION_TIME:
            stable_gesture = candidate_gesture
            # Unity로 안정된 제스처 전송
            send_gesture_to_unity(stable_gesture)
            return stable_gesture

    # 아직 안정화 시간이 지나지 않았거나 제스처가 None인 경우
    # Unity에 제스처 없음 전송 (candidate_gesture가 None인 경우만)
    if candidate_gesture is None:
        send_no_gesture_to_unity()
    return None


def classify_gesture_from_pattern(
    integrated_states, bottom_camera_state, side_camera_state
):
    """
    손가락 상태 패턴으로 제스처를 즉시 분류합니다 (안정화 없이).

    제스처 패턴 (Thumb, Index, Middle, Ring, Pinky):
    - A: [0, -1, -1, -1, -1]
    - Open A: [1, -1, -1, -1, -1]
    - Bent B: [1, 0, 0, 0, 0]
    - Bent V: [-1, 0, 0, -1, -1]
    - W: [-1, 1, 1, 1, -1]
    - X: [-1, 0, -1, -1, -1]
    - F: [0, 0, 1, 1, 1] + 엄지-검지 떨어짐
    - Open F: [0, 0, 1, 1, 1] + 엄지-검지 접촉
    - Y: [1, -1, -1, -1, 1]
    - L-I (I Love You): [1, 1, -1, -1, 1]
    - 1-1: [-1, 1, -1, -1, 1]
    - 3: [1, 1, 1, -1, -1]
    - G: [0, 1, -1, -1, -1]
    - I: [-1, -1, -1, -1, 1]
    - L: [1, 1, -1, -1, -1]
    - Bent 3: [1, 0, 0, -1, -1]
    - 8: [-1, 1, -1, 1, 1]
    - Open N: [0, 1, 1, -1, -1]
    - Open 8: [-1, 1, 0, 1, 1]
    - Bent L: [0, 0, -1, -1, -1] + 엄지-검지 떨어짐
    - Baby O: [0, 0, -1, -1, -1] + 엄지-검지 접촉
    - B: [-1, 1, 1, 1, 1] + 검지-중지 붙음
    - 4: [-1, 1, 1, 1, 1] + 검지-중지 벌어짐
    - Open B: [1, 1, 1, 1, 1] + 검지-중지 붙음
    - 5: [1, 1, 1, 1, 1] + 검지-중지 벌어짐
    - U: [-1, 1, 1, -1, -1] + 검지-중지 붙음
    - V: [-1, 1, 1, -1, -1] + 검지-중지 벌어짐
    - C: [0, 0, 0, 0, 0] + 검지-중지 붙음
    - O: [0, 0, 0, 0, 0] + 엄지-검지 접촉
    - Bent5: [0, 0, 0, 0, 0] + 검지-중지 벌어짐

    Args:
        integrated_states: 통합 손가락 상태
        bottom_camera_state: 하단 카메라 상태 (E/S/M/N/T용)
        side_camera_state: 측면 카메라 상태 (E/S/M/N/T용)

    Returns:
        str: 제스처 이름 또는 None
    """
    if integrated_states is None:
        return None

    bottom_states = integrated_states.get("bottom", {})

    # 손가락 상태를 배열로 변환 [Thumb, Index, Middle, Ring, Pinky]
    pattern = [
        bottom_states.get("Thumb"),
        bottom_states.get("Index"),
        bottom_states.get("Middle"),
        bottom_states.get("Ring"),
        bottom_states.get("Pinky"),
    ]

    # None이 있으면 None 반환
    if None in pattern:
        return None

    # 손날 상태 확인 (공통)
    is_side_facing = False
    if side_camera_state is not None:
        is_side_facing = getattr(side_camera_state, "is_side_facing", False)

    # 특수 케이스 2-1: SG 샷건 발사 [0, 1, 1, 1, 1]
    if pattern == [0, 1, 1, 1, 1]:
        # 손날일 때는 차단
        if is_side_facing:
            return None
        return "SG"

    # 특수 케이스 2-2: S1 샷건 재장전 [-1, 1, 1, 1, 1]
    if pattern == [-1, 1, 1, 1, 1]:
        # 손날일 때는 차단
        if is_side_facing:
            return None

        # 모든 펴진 손가락(검지, 중지, 약지, 소지)이 함께 있는지 확인
        extended_fingers_info = None
        if bottom_camera_state is not None:
            extended_fingers_info = getattr(
                bottom_camera_state, "extended_fingers_info", None
            )

        if extended_fingers_info is not None:
            if extended_fingers_info.get("all_together", False):
                return "S1"
            else:
                return "S1"  # 손가락 벌어지면 인식하지 않음 (4 제스처)
        else:
            # 정보가 없으면 일단 S1로 처리
            return "S1"

    # 특수 케이스 2-2-1: M1 소총 재장전 [-1, 1, 1, -1, -1]
    if pattern == [-1, 1, 1, -1, -1]:
        # 손날일 때는 차단
        if is_side_facing:
            return None
        return "M1"

    # 특수 케이스 2-3: [1, 1, 1, 1, 1] 패턴은 B 제스처 (기본 샷건)
    if pattern == [1, 1, 1, 1, 1]:
        # 손날일 때는 차단
        if is_side_facing:
            return None

        # 네 손가락(검지, 중지, 약지, 소지)이 함께 있는지 확인 (엄지 제외)
        extended_fingers_info = None
        if bottom_camera_state is not None:
            extended_fingers_info = getattr(
                bottom_camera_state, "extended_fingers_info", None
            )

        if extended_fingers_info is not None:
            if extended_fingers_info.get("all_together", False):
                return "B"
            else:
                return "B"  # 손가락 벌어지면 인식하지 않음
        else:
            return None  # 정보가 없으면 인식하지 않음

    # 특수 케이스 3: [-1, 1, -1, -1, -1] 패턴은 엄지 normalized_y로 1 구분 (높은 위치)
    if pattern == [-1, 1, -1, -1, -1]:
        # 손날일 때는 차단
        if is_side_facing:
            return None

        # 엄지 normalized_y 확인
        thumb_norm_y = None
        if bottom_camera_state is not None and hasattr(
            bottom_camera_state, "thumb_debug"
        ):
            thumb_norm_y = bottom_camera_state.thumb_debug.get("normalized_y", None)

        if thumb_norm_y is not None:
            if thumb_norm_y >= 0.6:  # 엄지 높음일 때만 1 반환
                return "1"
            else:
                return None  # 엄지 낮으면 인식하지 않음
        else:
            return None  # normalized_y 값이 없으면 인식하지 않음

    # 제스처 패턴 매칭 (순서대로 확인)
    gesture_patterns = {
        "L": [1, 1, -1, -1, -1],
        "3": [1, 1, 1, -1, -1],
        "G": [0, 1, -1, -1, -1],
        "L-I": [1, 1, -1, -1, 1],
        "1-I": [-1, 1, -1, -1, 1],
        "8": [-1, 1, -1, 1, 1],
        "Open N": [0, 1, 1, -1, -1],
        "Bent 3": [1, 0, 0, -1, -1],
        "Baby O": [0, 0, -1, -1, -1],
    }

    # 패턴 매칭
    for gesture_name, gesture_pattern in gesture_patterns.items():
        if pattern == gesture_pattern:
            # 손날일 때는 G 제스처만 허용 (H는 위에서 이미 처리됨)
            is_side_facing = False
            if side_camera_state is not None:
                is_side_facing = getattr(side_camera_state, "is_side_facing", False)

            if is_side_facing:
                # 손날일 때는 G만 허용
                if gesture_name == "G":
                    return gesture_name
                else:
                    return None
            else:
                # 소총 3 제스처에 발사/재장전 로직 추가
                if gesture_name == "3":
                    thumb_state = bottom_states.get("Thumb")
                    if thumb_state == 0:
                        return "3_Fire"  # 소총 발사 (엄지 Between)
                    elif thumb_state == -1:
                        return "3_Reload"  # 소총 재장전 (엄지 Bent)
                    else:
                        return "3"  # 기본 소총 (엄지 Straight)
                else:
                    # 다른 제스처는 기본 처리
                    return gesture_name

    # 매칭되는 패턴이 없으면 None 반환
    return None


def classify_gesture_from_integrated_states(
    integrated_states, bottom_camera_state, side_camera_state
):
    """
    통합 손가락 상태로부터 제스처를 분류합니다 (하위 호환성을 위해 유지).
    새로운 classify_gesture_from_pattern 함수를 호출합니다.
    """
    return classify_gesture_from_pattern_stabilized(
        integrated_states, bottom_camera_state, side_camera_state
    )


def process_hand_landmarks(
    hand_landmarks, handedness, camera_state, image, other_camera_state=None
):
    """단일 손의 랜드마크를 처리하는 함수"""
    h, w, _ = image.shape

    # hand_landmarks와 이미지 크기 저장 (draw_results에서 사용)
    camera_state.hand_landmarks = hand_landmarks
    camera_state.image_width = w
    camera_state.image_height = h

    # 손 방향 확인
    is_arm_raised = check_hand_orientation(hand_landmarks)

    # 각 손가락 각도 계산 (MCP-PIP-TIP 방식으로 복구)
    angles = {
        "Thumb": finger_angle(hand_landmarks, 2, 3, 4),
        "Index": finger_angle(hand_landmarks, 5, 6, 8),
        "Middle": finger_angle(hand_landmarks, 9, 10, 12),
        "Ring": finger_angle(hand_landmarks, 13, 14, 16),
        "Pinky": finger_angle(hand_landmarks, 17, 18, 20),
    }

    # 현재 카메라의 각도 저장
    camera_state.finger_angles = angles.copy()

    # ===== 싱글 각도만 사용: 다중 관절 각도 계산 제거 =====
    # 싱글 각도만 사용하므로 Multi-joint 계산은 생략

    # 검지 각도 스무딩
    raw_index_angle = angles["Index"]
    smoothed_index_angle = camera_state.angle_smoother.smooth(raw_index_angle)

    # 손날(측면) 방향 감지 먼저 수행 (손가락 상태 분류에서 사용)
    is_side, palm_z, confidence = check_hand_side_orientation(
        hand_landmarks, camera_state.camera_type
    )
    camera_state.is_side_facing = is_side
    camera_state.palm_normal_z = palm_z
    camera_state.side_facing_confidence = confidence

    # 3단계 손가락 상태 분류 (1: straight, 0: between, -1: bent)
    finger_states_numeric = {}

    for finger_name in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        # 엄지는 특별 처리
        if finger_name == "Thumb":
            thumb_state, thumb_normalized_y = classify_thumb_state_side(
                hand_landmarks, camera_state.camera_type, handedness
            )
            finger_states_numeric["Thumb"] = thumb_state
            thumb_extension = None
            thumb_angle_raw = finger_angle(hand_landmarks, 2, 3, 4)
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            thumb_mcp = hand_landmarks.landmark[2]

            # 하단 카메라: normalized 좌표 기반 zone 및 In1/In2/In3 계산
            if camera_state.camera_type == "bottom":
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                pinky_mcp = hand_landmarks.landmark[17]

                hand_length = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
                palm_width = math.hypot(
                    index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y
                )

                palm_center_x = (wrist.x + index_mcp.x + pinky_mcp.x) / 3
                palm_center_y = (wrist.y + index_mcp.y + pinky_mcp.y) / 3

                thumb_vector_x = (thumb_tip.x - palm_center_x) / (palm_width + 1e-6)
                thumb_vector_y = (thumb_tip.y - palm_center_y) / (hand_length + 1e-6)

                if handedness == "Left":
                    thumb_vector_x = -thumb_vector_x

                normalized_x = thumb_vector_x
                normalized_y = thumb_vector_y

                THUMB_INNER_THRESHOLD = 0.54
                THUMB_OUTER_THRESHOLD = 1.4
                INNER_Y_HIGH_THRESHOLD = 0.55
                INNER_Y_LOW_THRESHOLD = 0.27

                thumb_zone = "center"
                thumb_inner_subzone = 0

                if normalized_x <= THUMB_INNER_THRESHOLD:
                    thumb_zone = "inner"
                    if normalized_y >= INNER_Y_HIGH_THRESHOLD:
                        thumb_inner_subzone = 3  # In3
                    elif normalized_y >= INNER_Y_LOW_THRESHOLD:
                        thumb_inner_subzone = 2  # In2
                    else:
                        thumb_inner_subzone = 1  # In1
                elif normalized_x >= THUMB_OUTER_THRESHOLD:
                    thumb_zone = "outer"

                thumb_extension = math.hypot(
                    thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y
                )
            else:
                # 측면 카메라: 기존 로직
                thumb_zone = "outer" if thumb_tip.x > index_mcp.x else "inner"
                thumb_inner_subzone = 0  # N/A
                thumb_extension = abs(thumb_tip.x - index_mcp.x)
                normalized_x = 0
                normalized_y = 0

            debug_angle = thumb_angle_raw if thumb_zone == "inner" else -thumb_angle_raw
            thumb_extension_for_result = thumb_extension

            # 엄지 디버그용 값 저장 (In1/In2/In3 정보 포함)
            camera_state.thumb_debug = {
                "thumb_extension": thumb_extension,
                "thumb_angle_raw": thumb_angle_raw,
                "thumb_zone": thumb_zone,
                "thumb_angle_debug": debug_angle,
                "thumb_inner_subzone": thumb_inner_subzone,
                "normalized_x": normalized_x
                if camera_state.camera_type == "bottom"
                else None,
                "normalized_y": thumb_normalized_y,
            }
            continue

        # 엄지가 아닌 손가락들: 다중 관절 각도 사용!
        # 다중 관절 각도가 있으면 사용, 없으면 기존 방식
        # 🎯 싱글 각도: 측면 각도 + 하단 각도만 (Lower 각도는 사용하지 않음)

        # 측면 카메라 각도 (MCP-PIP-TIP)
        angle_side = angles[finger_name]

        # 하단 카메라 각도 가져오기 (있으면)
        angle_bottom = None
        if (
            other_camera_state is not None
            and finger_name in other_camera_state.finger_angles
        ):
            angle_bottom = other_camera_state.finger_angles[finger_name]

        # 싱글 각도 분류 함수 사용 (MCP-PIP-TIP만)
        finger_states_numeric[finger_name] = classify_finger_state_single_angle(
            angle_side,  # 측면 각도 (MCP-PIP-TIP)
            angle_bottom,  # 하단 각도 (optional, 융합용!)
            finger_name=finger_name,
            is_side_facing=camera_state.is_side_facing,
        )

    # 🎯 특수 케이스 제거: 싱글 각도만 사용하므로 Lower 각도 체크 없음

    # 분류 결과 저장
    camera_state.finger_states_numeric = finger_states_numeric

    # 측면 카메라일 때: tip 좌표 저장 (손가락 사이 판정용)
    if camera_state.camera_type == "side":
        try:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            rx, ry = int(ring_tip.x * w), int(ring_tip.y * h)

            camera_state.thumb_tip_coords = (tx, ty)
            camera_state.index_tip_coords = (ix, iy)
            camera_state.middle_tip_coords = (mx, my)
            camera_state.ring_tip_coords = (rx, ry)
        except Exception:
            camera_state.thumb_tip_coords = None
            camera_state.index_tip_coords = None
            camera_state.middle_tip_coords = None
            camera_state.ring_tip_coords = None

    # 하단카메라일 때: tip 좌표 및 정규화 관련 값 저장 (거리 계산은 draw_results에서 수행)
    if camera_state.camera_type == "bottom":
        try:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
            rx, ry = int(ring_tip.x * w), int(ring_tip.y * h)
            rdx, rdy = int(ring_dip.x * w), int(ring_dip.y * h)
            px, py = int(pinky_tip.x * w), int(pinky_tip.y * h)

            camera_state.thumb_tip_coords = (tx, ty)
            camera_state.index_tip_coords = (ix, iy)
            camera_state.middle_tip_coords = (mx, my)
            camera_state.ring_tip_coords = (rx, ry)
            camera_state.ring_dip_coords = (rdx, rdy)
            camera_state.pinky_tip_coords = (px, py)

            # store 3D z-values for tip depth comparisons
            try:
                camera_state.thumb_tip_z = float(thumb_tip.z)
            except Exception:
                camera_state.thumb_tip_z = None
            try:
                camera_state.index_tip_z = float(index_tip.z)
            except Exception:
                camera_state.index_tip_z = None
            try:
                camera_state.middle_tip_z = float(middle_tip.z)
            except Exception:
                camera_state.middle_tip_z = None
            try:
                camera_state.ring_tip_z = float(ring_tip.z)
            except Exception:
                camera_state.ring_tip_z = None
            try:
                camera_state.pinky_tip_z = float(pinky_tip.z)
            except Exception:
                camera_state.pinky_tip_z = None

            # store middle finger PIP (두번째 마디) for S gesture detection
            try:
                middle_pip = hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                ]
                mpx, mpy = int(middle_pip.x * w), int(middle_pip.y * h)
                camera_state.middle_pip_coords = (mpx, mpy)
                camera_state.middle_pip_z = float(middle_pip.z)
            except Exception:
                camera_state.middle_pip_coords = None
                camera_state.middle_pip_z = None

            # palm width in pixels (index MCP to pinky MCP) and hand size in 3D for normalization
            try:
                index_mcp = hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_MCP
                ]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            except Exception:
                # fall back to tips if MCP not available
                index_mcp = index_tip
                pinky_mcp = ring_tip

            palm_w_px = math.hypot(
                (index_mcp.x - pinky_mcp.x) * w, (index_mcp.y - pinky_mcp.y) * h
            )
            camera_state.palm_width_pixels = palm_w_px

            # hand size (3D) for depth normalization: index_mcp to wrist
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_mcp_3d = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_MCP
            ]
            hand_size_3d = math.sqrt(
                (index_mcp_3d.x - wrist.x) ** 2
                + (index_mcp_3d.y - wrist.y) ** 2
                + (index_mcp_3d.z - wrist.z) ** 2
            )
            camera_state.hand_size_3d = hand_size_3d

            # thumb depth (z) normalized by hand_size_3d
            try:
                thumb_z = thumb_tip.z
                thumb_depth_norm = (thumb_z - wrist.z) / (hand_size_3d + 1e-6)
                camera_state.thumb_depth_norm = thumb_depth_norm
                camera_state.thumb_tip_z = thumb_z
            except Exception:
                camera_state.thumb_depth_norm = None
                camera_state.thumb_tip_z = None

            # compute thumb relative to mean fingertip z (normalized)
            try:
                tip_zs = [
                    v
                    for v in [
                        camera_state.index_tip_z,
                        camera_state.middle_tip_z,
                        camera_state.ring_tip_z,
                    ]
                    if v is not None
                ]
                if (
                    tip_zs
                    and camera_state.thumb_tip_z is not None
                    and hand_size_3d > 1e-6
                ):
                    mean_fingertips_z = sum(tip_zs) / len(tip_zs)
                    camera_state.thumb_rel_to_fingertips_norm = (
                        camera_state.thumb_tip_z - mean_fingertips_z
                    ) / (hand_size_3d + 1e-6)
                else:
                    camera_state.thumb_rel_to_fingertips_norm = None
            except Exception:
                camera_state.thumb_rel_to_fingertips_norm = None

            # 검지-중지 거리 체크 (V/U 구분용: straight일 때, C/Bent5 구분용: between일 때)
            index_state = finger_states_numeric.get("Index")
            middle_state = finger_states_numeric.get("Middle")

            # 검지와 중지가 모두 straight(1)이거나 모두 between(0)일 때 거리 체크
            if (index_state == 1 and middle_state == 1) or (
                index_state == 0 and middle_state == 0
            ):
                is_together, distance, norm_dist = check_index_middle_distance(
                    camera_state, hand_size_3d
                )
                camera_state.index_middle_together = is_together
                camera_state.index_middle_distance = distance
                camera_state.index_middle_distance_norm = norm_dist
                camera_state.index_middle_norm_distance = norm_dist  # 표시용 alias
            else:
                camera_state.index_middle_together = None
                camera_state.index_middle_distance = None
                camera_state.index_middle_distance_norm = None
                camera_state.index_middle_norm_distance = None

            # 모든 펴진 손가락이 함께 있는지 체크
            extended_fingers_info = check_extended_fingers_together(
                camera_state, finger_states_numeric, hand_size_3d
            )
            camera_state.extended_fingers_info = extended_fingers_info

            # 엄지-다른손가락 접촉 체크 (특정 패턴에서만)
            # 패턴: [0,0,0,0,0], [0,0,1,1,1], [0,0,-1,-1,-1]
            pattern = [
                finger_states_numeric.get("Thumb"),
                finger_states_numeric.get("Index"),
                finger_states_numeric.get("Middle"),
                finger_states_numeric.get("Ring"),
                finger_states_numeric.get("Pinky"),
            ]

            # 엄지-다른손가락 접촉 체크 (항상 확인)
            thumb_touch_info = check_thumb_touching_fingers(camera_state, hand_size_3d)
            camera_state.thumb_touch_info = thumb_touch_info

            # 🎯 O vs Flattened O 구분을 위한 측정값 계산 (패턴 [-1,0,0,0,0]일 때)
            if pattern == [-1, 0, 0, 0, 0]:
                try:
                    # 1위: Tip Clustering (4개 손가락 TIP의 집중도)
                    # 검지, 중지, 약지, 소지 TIP 좌표 수집
                    tips_3d = []
                    if (
                        camera_state.index_tip_coords
                        and camera_state.index_tip_z is not None
                    ):
                        tips_3d.append(
                            (index_tip.x, index_tip.y, camera_state.index_tip_z)
                        )
                    if (
                        camera_state.middle_tip_coords
                        and camera_state.middle_tip_z is not None
                    ):
                        tips_3d.append(
                            (middle_tip.x, middle_tip.y, camera_state.middle_tip_z)
                        )
                    if (
                        camera_state.ring_tip_coords
                        and camera_state.ring_tip_z is not None
                    ):
                        tips_3d.append(
                            (ring_tip.x, ring_tip.y, camera_state.ring_tip_z)
                        )
                    if (
                        camera_state.pinky_tip_coords
                        and camera_state.pinky_tip_z is not None
                    ):
                        tips_3d.append(
                            (pinky_tip.x, pinky_tip.y, camera_state.pinky_tip_z)
                        )

                    if len(tips_3d) == 4:
                        # 4개 TIP의 평균 위치 계산
                        avg_x = sum(t[0] for t in tips_3d) / 4
                        avg_y = sum(t[1] for t in tips_3d) / 4
                        avg_z = sum(t[2] for t in tips_3d) / 4

                        # 각 TIP에서 평균까지의 3D 거리 계산
                        distances = [
                            math.sqrt(
                                (t[0] - avg_x) ** 2
                                + (t[1] - avg_y) ** 2
                                + (t[2] - avg_z) ** 2
                            )
                            for t in tips_3d
                        ]

                        # 평균 거리 계산 (hand_size_3d로 정규화)
                        avg_distance = sum(distances) / 4
                        tip_clustering_norm = avg_distance / (hand_size_3d + 1e-6)
                        camera_state.tip_clustering_value = tip_clustering_norm
                    else:
                        camera_state.tip_clustering_value = None

                    # 2위: 검지 TIP-DIP 거리
                    index_dip = hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_DIP
                    ]

                    # 3D 거리 계산
                    tip_dip_dist_3d = math.sqrt(
                        (index_tip.x - index_dip.x) ** 2
                        + (index_tip.y - index_dip.y) ** 2
                        + (index_tip.z - index_dip.z) ** 2
                    )

                    # hand_size_3d로 정규화
                    index_tip_dip_norm = tip_dip_dist_3d / (hand_size_3d + 1e-6)
                    camera_state.index_tip_dip_distance = index_tip_dip_norm

                except Exception as e:
                    print(f"[O/Flattened O 측정 에러] {e}")
                    camera_state.tip_clustering_value = None
                    camera_state.index_tip_dip_distance = None
            else:
                camera_state.tip_clustering_value = None
                camera_state.index_tip_dip_distance = None

        except Exception:
            camera_state.thumb_tip_coords = None
            camera_state.index_tip_coords = None
            camera_state.middle_tip_coords = None
            camera_state.ring_tip_coords = None

    # 거리 계산
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    x1_base, y1_base = int(index_mcp.x * w), int(index_mcp.y * h)
    x2_base, y2_base = int(wrist.x * w), int(wrist.y * h)
    base_dist_pixel = math.sqrt((x1_base - x2_base) ** 2 + (y1_base - y2_base) ** 2)

    norm_dist = -1.0
    raw_norm_dist = -1.0

    if base_dist_pixel > 1e-6:
        if camera_state.mode == "mode2":
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_pip = hand_landmarks.landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            ]
            x1_pip, y1_pip = int(index_pip.x * w), int(index_pip.y * h)
            x2_pip, y2_pip = int(middle_pip.x * w), int(middle_pip.y * h)
            pip_distance = math.sqrt((x1_pip - x2_pip) ** 2 + (y1_pip - y2_pip) ** 2)
            raw_norm_dist = pip_distance / base_dist_pixel
            norm_dist = camera_state.distance_smoother.smooth(raw_norm_dist)

        elif camera_state.mode == "mode1":
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            x1_pip, y1_pip = int(index_pip.x * w), int(index_pip.y * h)
            x2_pip, y2_pip = int(thumb_ip.x * w), int(thumb_ip.y * h)
            pip_distance = math.sqrt((x1_pip - x2_pip) ** 2 + (y1_pip - y2_pip) ** 2)
            raw_norm_dist = pip_distance / base_dist_pixel
            norm_dist = camera_state.distance_smoother.smooth(raw_norm_dist)

    # 손가락 펴짐/굽힘 판단
    fingers = {finger: (angle > ANGLE_THRESHOLD) for finger, angle in angles.items()}
    fingers["Index"] = smoothed_index_angle > ANGLE_THRESHOLD

    # 엄지 처리
    thumb_angle = 0
    if handedness:
        fingers["Thumb"] = is_thumb_extended(hand_landmarks, handedness)
        raw_thumb_angle = calculate_thumb_spread_angle(hand_landmarks, handedness)
        thumb_angle = camera_state.thumb_angle_smoother.smooth(raw_thumb_angle)

    # 모드 판별
    if not is_arm_raised:
        # mode5 조건
        if (
            not fingers["Thumb"]
            and fingers["Index"]
            and fingers["Middle"]
            and fingers["Ring"]
            and fingers["Pinky"]
        ):
            camera_state.mode5_counter += 1
            if camera_state.mode5_counter >= MODE5_CONFIRM_FRAMES:
                current_mode = "mode5"
            else:
                current_mode = None
        else:
            camera_state.mode5_counter = 0

            # mode0 조건
            if (
                fingers["Thumb"]
                and fingers["Index"]
                and fingers["Middle"]
                and fingers["Ring"]
                and fingers["Pinky"]
            ):
                current_mode = "mode0"

            # 기존 mode1, mode2 판별 로직
            elif camera_state.mode == "mode1":
                if (
                    fingers["Index"]
                    and fingers["Middle"]
                    and not fingers["Ring"]
                    and not fingers["Pinky"]
                ):
                    current_mode = "mode2"
                else:
                    current_mode = "mode1"
            elif camera_state.mode == "mode2":
                if (
                    fingers["Index"]
                    and not fingers["Middle"]
                    and not fingers["Ring"]
                    and not fingers["Pinky"]
                ):
                    current_mode = "mode1"
                else:
                    current_mode = "mode2"
            else:
                if (
                    fingers["Index"]
                    and fingers["Middle"]
                    and not fingers["Ring"]
                    and not fingers["Pinky"]
                ):
                    current_mode = "mode2"
                elif (
                    fingers["Index"]
                    and not fingers["Middle"]
                    and not fingers["Ring"]
                    and not fingers["Pinky"]
                ):
                    current_mode = "mode1"
                else:
                    current_mode = None

        # 모드 확정 시스템
        if current_mode == camera_state.last_detected_mode:
            camera_state.mode_confirmation_count += 1
        else:
            camera_state.mode_confirmation_count = 1
            camera_state.last_detected_mode = current_mode

        if (
            camera_state.mode_confirmation_count >= MODE_CONFIRMATION_THRESHOLD
            and current_mode != camera_state.last_confirmed_mode
        ):
            camera_state.last_confirmed_mode = current_mode
            if current_mode:
                camera_state.mode = current_mode
                if camera_state.prev_mode != camera_state.mode:
                    camera_state.distance_smoother.reset()
                    camera_state.thumb_angle_smoother.reset()
            else:
                camera_state.mode = None
        elif camera_state.mode_confirmation_count >= MODE_CONFIRMATION_THRESHOLD:
            camera_state.mode = camera_state.last_confirmed_mode

    camera_state.prev_mode = camera_state.mode

    result_dict = {
        "fingers": fingers,
        "smoothed_index_angle": smoothed_index_angle,
        "thumb_angle": thumb_angle,
        "norm_dist": norm_dist,
        "mode": camera_state.mode,
    }
    if camera_state.camera_type == "side" and "thumb_extension_for_result" in locals():
        result_dict["thumb_extension"] = thumb_extension_for_result
    return result_dict


def draw_results(
    image,
    results,
    camera_id,
    x_offset=0,
    camera_state=None,
    integrated_states=None,
    all_camera_states=None,
):
    """기존 단일 각도 및 손가락 상태 표시"""
    y0 = 30

    # 카메라 ID 및 타입 표시
    camera_type_text = (
        camera_state.camera_type.upper() if camera_state is not None else ""
    )
    cv2.putText(
        image,
        f"Camera {camera_id} ({camera_type_text})",
        (x_offset + 10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    y0 += 40

    # ======= 엄지 터치 상태 표시 =======
    if camera_state is not None and hasattr(camera_state, "thumb_touch_info"):
        thumb_touch_info = camera_state.thumb_touch_info

        # 제목
        cv2.putText(
            image,
            "THUMB TOUCH STATUS",
            (x_offset + 15, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),  # 노란색
            2,
        )
        y0 += 30

        if thumb_touch_info is not None:
            touching = thumb_touch_info.get("touching", False)
            touched_finger = thumb_touch_info.get("touched_finger", None)
            min_distance = thumb_touch_info.get("min_distance", None)
            distances = thumb_touch_info.get("distances", {})

            # 터치 상태 표시
            if touching and touched_finger:
                touch_color = (0, 255, 0)  # 녹색 (터치됨)
                touch_text = f"TOUCHING: {touched_finger}"
            else:
                touch_color = (0, 0, 255)  # 빨간색 (터치 안됨)
                touch_text = "NOT TOUCHING"

            cv2.putText(
                image,
                touch_text,
                (x_offset + 15, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                touch_color,
                2,
            )
            y0 += 25

            # 최소 거리 표시
            if min_distance is not None:
                distance_color = (0, 255, 0) if touching else (255, 255, 255)
                cv2.putText(
                    image,
                    f"Min Distance: {min_distance:.1f}px (Thresh: {THUMB_TOUCH_THRESHOLD}px)",
                    (x_offset + 15, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    distance_color,
                    1,
                )
                y0 += 20

            # 각 손가락별 거리 표시
            for finger_name in ["Index", "Middle", "Ring", "Pinky"]:
                if finger_name in distances:
                    distance = distances[finger_name]
                    is_closest = touched_finger == finger_name and touching

                    # 색상 결정
                    if is_closest:
                        finger_color = (0, 255, 0)  # 녹색 (터치된 손가락)
                    elif distance <= THUMB_TOUCH_THRESHOLD:
                        finger_color = (0, 255, 255)  # 노란색 (임계값 내)
                    else:
                        finger_color = (200, 200, 200)  # 회색

                    cv2.putText(
                        image,
                        f"  {finger_name}: {distance:.1f}px",
                        (x_offset + 15, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        finger_color,
                        1,
                    )
                    y0 += 18
        else:
            # thumb_touch_info가 None인 경우
            cv2.putText(
                image,
                "No touch data available",
                (x_offset + 15, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )
            y0 += 25

        y0 += 15

    # ======= 통합 제스처 표시 (상단 카메라만) =======
    if camera_id == 0 and integrated_states is not None:
        y0 += 10
        h, w = image.shape[:2]
        y_base = h - 100
        x_base = w - 250

        cv2.rectangle(
            image, (x_base - 10, y_base - 10), (w - 10, h - 10), (40, 40, 40), -1
        )
        cv2.putText(
            image,
            "[Final Integrated States]",
            (x_base, y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        y = y_base + 25

        # Bent5 제스처 감지 확인 (하단 카메라 상태 확인)
        is_bent5 = False
        if all_camera_states and len(all_camera_states) > 1:
            bottom_camera_state = all_camera_states[1]
            four_fingers_lower_bent = getattr(
                bottom_camera_state, "four_fingers_lower_bent", False
            )
            if four_fingers_lower_bent:
                is_bent5 = True

        for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
            bottom_val = integrated_states.get("bottom", {}).get(finger, None)
            side_val = integrated_states.get("side", {}).get(finger, None)

            # Bent5일 때 4손가락 강제로 -1
            if is_bent5 and finger in ["Index", "Middle", "Ring", "Pinky"]:
                final_val = -1
            # 일반 융합 로직
            elif bottom_val == -1 or side_val == -1:
                final_val = -1
            elif bottom_val == 1 and side_val == 1:
                final_val = 1
            else:
                final_val = 0

            # 디버그: 4손가락 상태 출력
            if finger in ["Index", "Middle", "Ring", "Pinky"]:
                bent5_mark = " [Bent5 강제]" if is_bent5 else ""
                print(
                    f"[통합] {finger}: bottom={bottom_val}, side={side_val} → final={final_val}{bent5_mark}"
                )

            txt = f"{finger}: {final_val:+d}"
            cv2.putText(
                image,
                txt,
                (x_base + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            y += 18

    # 하단 카메라일 경우: 제스처 분류 및 표시
    bottom_camera_state = None
    side_camera_state = None

    if camera_id == 1 and all_camera_states:  # 하단 카메라
        bottom_camera_state = (
            all_camera_states[1] if len(all_camera_states) > 1 else None
        )
        side_camera_state = all_camera_states[0] if len(all_camera_states) > 0 else None

    if (
        camera_id == 1
        and camera_state is not None
        and camera_state.camera_type == "bottom"
        and integrated_states is not None
    ):
        gesture = classify_gesture_from_integrated_states(
            integrated_states, bottom_camera_state, side_camera_state
        )

        # 제스처 표시 (큰 글씨, 눈에 띄는 색상)
        if gesture:
            gesture_color_map = {
                # E/S/M/N/T/C (모든 손가락 bent 제스처)
                "C": (0, 255, 255),  # 노랑 (손날 방향) - 최우선!
                "T": (255, 255, 255),  # 흰색 (상단: 검지-중지 사이)
                "N": (255, 255, 0),  # 시안 (상단: 중지-약지 사이)
                "M": (0, 165, 255),  # 주황색 (상단: 약지-소지 사이)
                "E": (0, 255, 0),  # 초록색 (하단 Y: 0.3~0.5)
                "S": (255, 0, 255),  # 마젠타 (하단 Y: 0.64~0.8)
                "Bent5": (255, 100, 255),  # 밝은 마젠타 (4손가락 Lower bent)
                # 패턴 기반 제스처
                "A": (100, 200, 255),  # 연한 주황색
                "Open A": (0, 200, 255),  # 진한 주황색
                "Bent V": (200, 150, 100),  # 갈색
                "W": (255, 200, 100),  # 골드
                "X": (150, 100, 200),  # 보라색
                "F": (100, 255, 100),  # 연한 초록
                "Open F": (0, 200, 100),  # 진한 초록
                "Y": (255, 100, 200),  # 핑크
                "L-I": (200, 100, 255),  # 연보라
                "1-1": (100, 255, 255),  # 연한 시안
                "3": (255, 150, 0),  # 오렌지
                "G": (150, 200, 255),  # 하늘색
                "I": (200, 200, 100),  # 황록색
                "L": (255, 180, 180),  # 연분홍
                "Bent 3": (180, 100, 150),  # 자주색
                "8": (100, 180, 255),  # 밝은 파랑
                "Open N": (150, 255, 150),  # 연두색
                "Open 8": (255, 150, 150),  # 연빨강
                "Bent L": (200, 255, 100),  # 연두-노랑
                "Baby O": (255, 200, 200),  # 연한 핑크
                "B": (80, 127, 255),  # 주황-코랄
                "4": (255, 191, 0),  # 딥 스카이블루
                # 무기 제스처 색상
                "3_Fire": (255, 0, 0),  # 빨강 (소총 발사)
                "3_Reload": (255, 255, 0),  # 노랑 (소총 재장전)
                "SG": (255, 0, 200),  # 밝은 마젠타 (샷건 발사)
                "S1": (255, 150, 0),  # 밝은 주황 (샷건 재장전)
                "M1": (200, 255, 0),  # 연두-노랑 (소총 재장전2)
                "Open B": (147, 20, 255),  # 딥 핑크
                "5": (0, 255, 255),  # 옐로우
                "U": (180, 105, 255),  # 핫 핑크
                "V": (203, 192, 255),  # 로즈 브라운
                "K": (50, 200, 50),  # K: 녹색 계열
                "R": (0, 120, 255),  # R: 주황-파랑 계열
                "1": (255, 255, 100),  # 1: 밝은 노랑
                "D": (100, 100, 255),  # D: 밝은 파랑
                "O": (255, 128, 0),  # O: 주황색
                "Flattened O": (0, 128, 255),  # Flattened O: 밝은 주황색
            }
            gesture_color = gesture_color_map.get(gesture, (200, 200, 200))
            cv2.putText(
                image,
                f"GESTURE: {gesture}",
                (x_offset + 10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,  # 큰 글씨
                gesture_color,
                3,
            )
            y0 += 40

    # 손가락 붙음/펴짐 상태 표시 (엄지 제외)
    if camera_state is not None and hasattr(camera_state, "extended_fingers_info"):
        extended_info = camera_state.extended_fingers_info
        if extended_info:
            # 펴진 손가락 리스트 (엄지 제외: Index, Middle, Ring, Pinky만)
            extended_fingers = extended_info.get("extended_fingers", [])
            all_together = extended_info.get("all_together", False)

            if extended_fingers:
                fingers_text = ", ".join(extended_fingers)
                status_text = "TOGETHER" if all_together else "SEPARATED"
                status_color = (
                    (0, 255, 0) if all_together else (0, 0, 255)
                )  # 초록색: 붙음, 빨강색: 펴짐

                cv2.putText(
                    image,
                    f"Fingers (no Thumb): {fingers_text}",
                    (x_offset + 10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                y0 += 30

                cv2.putText(
                    image,
                    f"Status: {status_text}",
                    (x_offset + 10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )
                y0 += 35

                # 각 쌍의 거리 표시 (디버깅용)
                pairwise_distances = extended_info.get("pairwise_distances", {})
                if pairwise_distances:
                    for pair, distance in pairwise_distances.items():
                        # Ring-Pinky는 특별 표시
                        if pair == "Ring-Pinky":
                            pair_text = f"{pair} (Ring DIP-Pinky TIP): {distance:.1f}px"
                        else:
                            pair_text = f"{pair} (TIP-TIP): {distance:.1f}px"

                        cv2.putText(
                            image,
                            pair_text,
                            (x_offset + 10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (200, 200, 200),
                            1,
                        )
                        y0 += 20

    # 엄지 접촉 정보 표시
    if camera_state is not None and hasattr(camera_state, "thumb_touch_info"):
        thumb_touch_info = camera_state.thumb_touch_info
        if thumb_touch_info:
            is_touching = thumb_touch_info.get("touching", False)
            touched_finger = thumb_touch_info.get("touched_finger", None)
            min_distance = thumb_touch_info.get("min_distance", None)

            if is_touching and touched_finger:
                cv2.putText(
                    image,
                    f"Thumb Touch: {touched_finger}",
                    (x_offset + 10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),  # 노란색
                    2,
                )
                y0 += 30

                if min_distance is not None:
                    cv2.putText(
                        image,
                        f"Distance: {min_distance:.1f}px",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                    )
                    y0 += 25
            else:
                cv2.putText(
                    image,
                    "Thumb Touch: None",
                    (x_offset + 10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (150, 150, 150),  # 회색
                    1,
                )
                y0 += 30

    # 측면 카메라: 엄지가 손가락 사이에 끼어있는지 항상 표시
    if camera_state is not None and camera_state.camera_type == "side":
        # 저장된 hand_landmarks와 이미지 크기 가져오기
        hand_landmarks = getattr(camera_state, "hand_landmarks", None)
        img_w = getattr(camera_state, "image_width", None)
        img_h = getattr(camera_state, "image_height", None)

        if hand_landmarks and img_w and img_h:
            # 손가락 사이 판정 (PIP 기반)
            is_between, segment, details = check_thumb_between_fingers_side(
                camera_state, hand_landmarks, img_w, img_h
            )

            # 결과 표시
            if is_between:
                if segment == "IM":
                    result_text = "Thumb: Index-Middle (T)"
                    result_color = (255, 255, 255)  # 흰색
                elif segment == "MR":
                    result_text = "Thumb: Middle-Ring (N)"
                    result_color = (255, 255, 0)  # 시안
                elif segment == "RP":
                    result_text = "Thumb: Ring-Pinky (M)"
                    result_color = (0, 165, 255)  # 주황
                else:
                    result_text = f"Thumb: {segment}"
                    result_color = (0, 255, 0)  # 초록색
            else:
                result_text = "Thumb: NOT BETWEEN"
                result_color = (100, 100, 100)  # 회색

            cv2.putText(
                image,
                result_text,
                (x_offset + 10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                result_color,
                2,
            )
            y0 += 30

            # 상세 값 표시 (빨간색)
            cv2.putText(
                image,
                "==== Between Fingers Details ====",
                (x_offset + 10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            y0 += 25

            # Thumb Y 위치
            ty = details.get("ty")
            if ty is not None:
                cv2.putText(
                    image,
                    f"Thumb Y: {ty}",
                    (x_offset + 10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                y0 += 22

            # X 범위 체크 (OUT_OF_X_RANGE인 경우)
            if segment == "OUT_OF_X_RANGE":
                tx = details.get("tx")
                x_range = details.get("x_range")
                margin = details.get("margin", 40)
                if tx is not None and x_range is not None:
                    cv2.putText(
                        image,
                        f"X: {tx} (range: {x_range[0]}-{x_range[1]} +/-{margin})",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 22

            # 세그먼트별 범위 정보 (NONE인 경우)
            if segment == "NONE":
                im_range = details.get("im_range")
                mr_range = details.get("mr_range")
                rp_range = details.get("rp_range")

                if im_range:
                    cv2.putText(
                        image,
                        f"IM(T) range: {im_range[0]:.1f} - {im_range[1]:.1f}",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 22

                if mr_range:
                    cv2.putText(
                        image,
                        f"MR(N) range: {mr_range[0]:.1f} - {mr_range[1]:.1f}",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 22

                if rp_range:
                    cv2.putText(
                        image,
                        f"RP(M) range: {rp_range[0]:.1f} - {rp_range[1]:.1f}",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 22

            # 매칭된 세그먼트 정보 (IM, MR, RP인 경우)
            if is_between:
                seg_range = details.get("segment_y_range")
                seg_center = details.get("segment_y_center")
                dist_from_center = details.get("distance_from_center")
                confidence = details.get("confidence")

                if seg_range:
                    cv2.putText(
                        image,
                        f"Segment Y range: {seg_range[0]} - {seg_range[1]}",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 22

                if seg_center is not None:
                    cv2.putText(
                        image,
                        f"Segment center: {seg_center:.1f}",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 22

                if dist_from_center is not None:
                    cv2.putText(
                        image,
                        f"Distance from center: {dist_from_center:.1f}",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    y0 += 22

                if confidence is not None:
                    cv2.putText(
                        image,
                        f"Confidence: {confidence:.2f}",
                        (x_offset + 10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if confidence > 0.7 else (0, 0, 255),
                        2,
                    )
                    y0 += 22

            # 임계값 정보 표시
            cv2.putText(
                image,
                "Thresholds: X_margin=40px, Y_margin=30%",
                (x_offset + 10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
            y0 += 25

        # --- 디버그: 손날일 때 측면 카메라 추적값 표시 (숨김 처리) ---
        # 필요시 아래 주석을 해제하여 디버그 가능
        # try:
        #     # 손날 상태 확인
        #     is_side_facing = getattr(camera_state, "is_side_facing", False)
        #
        #     if is_side_facing:
        #         debug_x = x_offset + 10
        #         # 헤더 (빨간색)
        #         cv2.putText(
        #             image,
        #             "==== SIDE FACING: Finger Tracking Values ====",
        #             (debug_x, y0),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6,
        #             (0, 0, 255),
        #             2,
        #         )
        #         y0 += 25
        #
        #         # 다른 카메라 상태(가능하면 하단)
        #         other_state = None
        #         if all_camera_states and len(all_camera_states) > 1:
        #             other_state = all_camera_states[1]
        #
        #         for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        #             side_ang = None
        #             other_ang = None
        #             state_val = None
        #             if hasattr(camera_state, "finger_angles") and camera_state.finger_angles:
        #                 side_ang = camera_state.finger_angles.get(finger)
        #             if other_state is not None and hasattr(other_state, "finger_angles") and other_state.finger_angles:
        #                 other_ang = other_state.finger_angles.get(finger)
        #             if hasattr(camera_state, "finger_states_numeric") and camera_state.finger_states_numeric:
        #                 state_val = camera_state.finger_states_numeric.get(finger)
        #
        #             # 포맷 텍스트
        #             side_txt = f"{side_ang:.1f}" if side_ang is not None else "N/A"
        #             other_txt = f"{other_ang:.1f}" if other_ang is not None else "N/A"
        #             state_txt = f"{state_val:+d}" if isinstance(state_val, int) else str(state_val)
        #
        #             # 각 손가락별 정보 (빨간색, 더 두껍게)
        #             cv2.putText(
        #                 image,
        #                 f"{finger}: side={side_txt} other={other_txt} state={state_txt}",
        #                 (debug_x, y0),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.6,
        #                 (0, 0, 255),
        #                 2,
        #             )
        #             y0 += 22
        #
        #         # 엄지 세부값(가능하면) - 빨간색
        #         thumb_dbg = getattr(camera_state, "thumb_debug", None)
        #         if thumb_dbg:
        #             tx = thumb_dbg.get("thumb_extension")
        #             tz = thumb_dbg.get("thumb_angle_debug")
        #             tzone = thumb_dbg.get("thumb_zone")
        #             nx = thumb_dbg.get("normalized_x")
        #             ny = thumb_dbg.get("normalized_y")
        #
        #             nx_str = f"{nx:.3f}" if nx is not None else "N/A"
        #             ny_str = f"{ny:.3f}" if ny is not None else "N/A"
        #             tx_str = f"{tx:.3f}" if tx is not None else "N/A"
        #             tz_str = f"{tz:.1f}" if tz is not None else "N/A"
        #
        #             cv2.putText(
        #                 image,
        #                 f"Thumb: ext={tx_str} ang={tz_str} zone={tzone}",
        #                 (debug_x, y0),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.6,
        #                 (0, 0, 255),
        #                 2,
        #             )
        #             y0 += 22
        #             cv2.putText(
        #                 image,
        #                 f"Thumb: nx={nx_str} ny={ny_str}",
        #                 (debug_x, y0),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.6,
        #                 (0, 0, 255),
        #                 2,
        #             )
        #             y0 += 25
        # except Exception:
        #     pass

        # 거리
        norm_dist = results["norm_dist"]
        if norm_dist > 0:
            cv2.putText(
                image,
                f"Distance: {norm_dist:.3f}",
                (x_offset + 10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
        else:
            cv2.putText(
                image,
                "Distance: N/A",
                (x_offset + 10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
    y0 += 25


def main():
    global INDEX_MIDDLE_DISTANCE_THRESHOLD, THUMB_TOUCH_THRESHOLD

    # Unity 웹소켓 연결 초기화
    print("Unity 웹소켓 연결 시도...")
    init_unity_websocket()

    # 카메라 상태 초기화 (0: side/측면, 1: bottom/하단)
    camera_states = {
        0: CameraState(0, camera_type="side"),
        1: CameraState(1, camera_type="bottom"),
    }

    # 카메라 열기 (멀티스레딩 버전 사용)
    print("Starting threaded cameras...")
    cap0 = ThreadedCamera(0).start()
    cap1 = ThreadedCamera(1).start()

    # 카메라 초기화 대기
    time.sleep(0.5)

    print("Threaded cameras ready")

    # 웹소켓 연결
    ws = None
    try:
        ws = websocket.create_connection("ws://192.168.0.210:5678", timeout=2)
        print("WebSocket connected.")
    except Exception as e:
        print(f"WebSocket connection failed: {e}")
        print("Continuing without WebSocket connection...")
        ws = None

    # 각 카메라별로 독립적인 MediaPipe Hands 인스턴스 생성
    # 최적화: model_complexity=0 (가장 빠른 모델)
    hands_side = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,  # 0=가장 빠름, 1=기본값
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands_bottom = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,  # 0=가장 빠름, 1=기본값
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # FPS 계산을 위한 변수
    frame_count = 0
    fps_start_time = time.time()
    fps = 0

    # 시간 측정을 위한 누적 변수
    total_read_time = 0
    total_resize_time = 0
    total_mediapipe_time = 0
    total_draw_time = 0
    total_loop_time = 0

    try:
        while True:
            loop_start = time.time()

            # 두 카메라에서 프레임 읽기 (이미 별도 스레드에서 읽고 있음)
            t0 = time.time()
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            read_time = time.time() - t0

            if not ret0 or not ret1 or frame0 is None or frame1 is None:
                print("카메라에서 프레임을 읽을 수 없습니다.")
                continue  # break 대신 continue로 변경

            # 프레임 크기 조정 (더 작게 해서 처리 속도 향상)
            t0 = time.time()
            frame0 = cv2.resize(frame0, (480, 360))  # 640x480 → 480x360
            frame1 = cv2.resize(frame1, (480, 360))
            resize_time = time.time() - t0

            # 각 카메라별로 손 추적 처리
            results_data = {}
            processed_frames = [None, None]

            # 1단계: 모든 카메라에서 랜드마크 추적 시작
            t0 = time.time()

            # 1단계: 모든 카메라에서 랜드마크 추출 및 각도 계산
            for camera_id, (frame, camera_state) in enumerate(
                [(frame0, camera_states[0]), (frame1, camera_states[1])]
            ):
                # 카메라별로 적절한 hands 인스턴스 선택
                hands = hands_side if camera_id == 0 else hands_bottom

                # 이미지 처리
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                # 결과 다시 BGR로
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 랜드마크 처리
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        handedness = None
                        if results.multi_handedness:
                            handedness = (
                                results.multi_handedness[idx].classification[0].label
                            )

                        # 다른 카메라 상태 가져오기
                        other_camera_id = 1 - camera_id
                        other_camera_state = camera_states[other_camera_id]

                        # 손 랜드마크 처리 (다른 카메라 정보 전달)
                        hand_results = process_hand_landmarks(
                            hand_landmarks,
                            handedness,
                            camera_state,
                            image,
                            other_camera_state,
                        )
                        results_data[camera_id] = hand_results

                        # 결과 그리기 (camera_state 전달)
                        # 하단카메라(1번)일 때 통합 결과 전달
                        integrated_states = None
                        if camera_id == 1:
                            integrated_states = {
                                "side": camera_states[0].finger_states_numeric.copy()
                                if camera_states[0].finger_states_numeric
                                else None,
                                "bottom": camera_states[1].finger_states_numeric.copy()
                                if camera_states[1].finger_states_numeric
                                else None,
                            }
                        draw_results(
                            image,
                            hand_results,
                            camera_id,
                            camera_state=camera_state,
                            integrated_states=integrated_states,
                            all_camera_states=camera_states,
                        )

                        # 손 랜드마크 시각화
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                        # 캘리브레이션 수집
                        if (
                            camera_state.calibration.state
                            in ["mode1_collect", "mode2_collect"]
                            and hand_results["mode"]
                            and hand_results["norm_dist"] > 0
                        ):
                            camera_state.calibration.collect_sample(
                                hand_results["mode"], hand_results["norm_dist"]
                            )

                processed_frames[camera_id] = image

            mediapipe_time = time.time() - t0

            # 두 프레임을 나란히 합치기
            t0 = time.time()
            processed_frame0 = (
                processed_frames[0] if processed_frames[0] is not None else frame0
            )
            processed_frame1 = (
                processed_frames[1] if processed_frames[1] is not None else frame1
            )

            # 측면카메라(0번) 우측하단에 최종 통합값만 표시
            if processed_frames[0] is not None:
                h, w = processed_frame0.shape[:2]
                y_base = h - 160
                x_base = w - 180
                cv2.rectangle(
                    processed_frame0,
                    (x_base - 10, y_base - 10),
                    (w - 10, h - 10),
                    (40, 40, 40),
                    -1,
                )
                cv2.putText(
                    processed_frame0,
                    "[Finger States]",
                    (x_base, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                y = y_base + 30
                # 통합값: Thumb은 하단카메라, 나머지는 측면카메라 기준
                thumb_bottom = camera_states[1].finger_states_numeric.get("Thumb", None)
                fingers_side = camera_states[0].finger_states_numeric
                for idx, finger in enumerate(
                    ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                ):
                    if finger == "Thumb":
                        val = thumb_bottom
                    else:
                        val = fingers_side.get(finger, None)
                    if val is not None:
                        txt = f"{finger}: {val:+d}"
                        cv2.putText(
                            processed_frame0,
                            txt,
                            (x_base + 10, y + idx * 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                        )

            combined_frame = np.hstack((processed_frame0, processed_frame1))
            draw_time = time.time() - t0

            # 화면에 표시
            t0 = time.time()
            cv2.imshow("Dual Camera Hand Tracking", combined_frame)
            display_time = time.time() - t0

            # 전체 루프 시간
            loop_time = time.time() - loop_start

            # 시간 누적
            total_read_time += read_time
            total_resize_time += resize_time
            total_mediapipe_time += mediapipe_time
            total_draw_time += draw_time
            total_loop_time += loop_time

            # FPS 계산 및 출력
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 0.1:  # 1초마다 FPS 출력
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")

                frame_count = 0
                fps_start_time = time.time()
                total_read_time = 0
                total_resize_time = 0
                total_mediapipe_time = 0
                total_draw_time = 0
                total_loop_time = 0

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' 또는 ESC
                break
            elif key == ord("1"):  # Mode1 캘리브레이션
                for camera_state in camera_states.values():
                    camera_state.calibration.start_mode1_calibration()
                print("Mode1 calibration started for both cameras")
            elif key == ord("2"):  # Mode2 캘리브레이션
                for camera_state in camera_states.values():
                    camera_state.calibration.start_mode2_calibration()
                print("Mode2 calibration started for both cameras")
            elif key == ord("0"):  # 웹소켓 재연결
                if ws is None:
                    try:
                        ws = websocket.create_connection(
                            "ws://192.168.0.210:5678", timeout=2
                        )
                        print("WebSocket reconnected.")
                    except Exception as e:
                        print(f"WebSocket reconnection failed: {e}")
                        ws = None
                else:
                    print("WebSocket is already connected.")
            elif key == ord("+") or key == ord("="):  # 임계값 증가
                INDEX_MIDDLE_DISTANCE_THRESHOLD += 0.01
                print(
                    f"Index-Middle threshold increased to {INDEX_MIDDLE_DISTANCE_THRESHOLD:.4f}"
                )
            elif key == ord("-") or key == ord("_"):  # 임계값 감소
                INDEX_MIDDLE_DISTANCE_THRESHOLD = max(
                    0.01, INDEX_MIDDLE_DISTANCE_THRESHOLD - 0.01
                )
                print(
                    f"Index-Middle threshold decreased to {INDEX_MIDDLE_DISTANCE_THRESHOLD:.4f}"
                )
            elif key == ord("t"):  # 엄지 접촉 임계값 감소
                THUMB_TOUCH_THRESHOLD = max(0.01, THUMB_TOUCH_THRESHOLD - 0.01)
                print(f"Thumb touch threshold decreased to {THUMB_TOUCH_THRESHOLD:.4f}")
            elif key == ord("T"):  # 엄지 접촉 임계값 증가
                THUMB_TOUCH_THRESHOLD += 0.01
                print(f"Thumb touch threshold increased to {THUMB_TOUCH_THRESHOLD:.4f}")

            # 웹소켓 전송 (첫 번째 카메라 데이터 사용)
            if 0 in results_data and ws is not None:
                hand_data = results_data[0]
                current_time = time.time()

                if (hand_data["mode"] and hand_data["norm_dist"] > 0) or hand_data[
                    "mode"
                ] in ["mode0", "mode5"]:
                    camera_state = camera_states[0]

                    mode_changed = (
                        camera_state.last_confirmed_mode != camera_state.last_sent_mode
                        and camera_state.last_confirmed_mode is not None
                    )

                    if hand_data["mode"] in ["mode0", "mode5"] and mode_changed:
                        payload = {"m": 0 if hand_data["mode"] == "mode0" else 5}
                        try:
                            ws.send(json.dumps(payload))
                            camera_state.last_sent_mode = hand_data["mode"]
                            print(f"Mode change sent: {hand_data['mode']}")
                        except Exception as e:
                            print(f"WebSocket send error: {e}")
                            ws = None

                    elif (
                        hand_data["mode"] in ["mode1", "mode2"]
                        and current_time - camera_state.last_send_time >= 0.05
                    ):
                        # 검지 상태 분류
                        smoothed_index_angle = hand_data["smoothed_index_angle"]
                        if smoothed_index_angle <= 81:
                            index_status_code = 1
                        elif 82 <= smoothed_index_angle <= 114:
                            index_status_code = 2
                        else:
                            index_status_code = 3

                        if mode_changed or camera_state.last_sent_mode is None:
                            m_val = 1 if hand_data["mode"] == "mode1" else 2
                            payload = {"m": m_val, "is": index_status_code}
                            try:
                                ws.send(json.dumps(payload))
                                camera_state.last_sent_mode = hand_data["mode"]
                                camera_state.last_sent_is = index_status_code
                                print(
                                    f"Mode change sent: {hand_data['mode']} with is: {index_status_code}"
                                )
                            except Exception as e:
                                print(f"WebSocket send error: {e}")
                                ws = None
                        else:
                            if index_status_code != camera_state.last_sent_is:
                                payload = {"is": index_status_code}
                                try:
                                    ws.send(json.dumps(payload))
                                    camera_state.last_sent_is = index_status_code
                                    print(f"Data update sent: is={index_status_code}")
                                except Exception as e:
                                    print(f"WebSocket send error: {e}")
                                    ws = None

                        camera_state.last_send_time = current_time

    finally:
        # 정리
        hands_side.close()
        hands_bottom.close()
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()

        if ws:
            ws.close()

        # Unity 웹소켓 연결 종료
        if unity_websocket:
            unity_websocket.close()


def init_unity_websocket():
    """Unity 웹소켓 연결을 초기화합니다."""
    global unity_websocket
    try:
        unity_websocket = websocket.WebSocket()
        unity_websocket.connect(UNITY_WEBSOCKET_URL)
        print(f"Unity 웹소켓 연결 성공: {UNITY_WEBSOCKET_URL}")
        return True
    except Exception as e:
        print(f"Unity 웹소켓 연결 실패: {e}")
        unity_websocket = None
        return False


def send_gesture_to_unity(gesture_name):
    """제스처를 Unity로 숫자로 전송합니다."""
    global unity_websocket, last_sent_gesture

    if not unity_websocket:
        return False

    # 무기 제스처(Fire/Reload)는 항상 전송 (중복 방지 안함)
    # 일반 제스처(SG, S1 포함)는 이전과 동일한 제스처 재전송 방지
    if not is_weapon_gesture(gesture_name):
        if gesture_name == last_sent_gesture:
            return True

    try:
        gesture_number = GESTURE_TO_NUMBER.get(gesture_name, 0)

        if gesture_number == 0:
            print(f"경고: 제스처 '{gesture_name}'에 대한 번호를 찾을 수 없음!")

        unity_websocket.send(str(gesture_number))

        # 제스처 전송 로그
        if gesture_name == "SG":
            print(f"Unity 전송: SG (샷건 발사) -> {gesture_number}")
        elif gesture_name == "S1":
            print(f"Unity 전송: S1 (샷건 재장전) -> {gesture_number}")
        elif "Fire" in gesture_name:
            print(f"Unity 전송: {gesture_name} (발사) -> {gesture_number}")
        elif "Reload" in gesture_name:
            print(f"Unity 전송: {gesture_name} (재장전) -> {gesture_number}")
        else:
            print(f"Unity 전송: {gesture_name} -> {gesture_number}")

        last_sent_gesture = gesture_name
        return True
    except Exception as e:
        print(f"Unity 전송 실패: {e}")
        # 연결이 끊어진 경우 재연결 시도
        init_unity_websocket()
        return False


def send_no_gesture_to_unity():
    """제스처가 없음을 Unity로 전송합니다 (0)."""
    global unity_websocket, last_sent_gesture

    if not unity_websocket:
        return False

    # 현재 전송된 제스처가 Fire/Reload 무기 관련이면 0을 보내지 않음
    if is_weapon_gesture(last_sent_gesture):
        return True

    # 이미 0을 보낸 상태면 재전송하지 않음
    if last_sent_gesture is None:
        return True

    try:
        unity_websocket.send("0")
        print("Unity 전송: No Gesture -> 0")
        last_sent_gesture = None
        return True
    except Exception as e:
        print(f"Unity 전송 실패: {e}")
        # 연결이 끊어진 경우 재연결 시도
        init_unity_websocket()
        return False


if __name__ == "__main__":
    main()
