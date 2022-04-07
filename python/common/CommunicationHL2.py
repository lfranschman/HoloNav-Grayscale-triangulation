import threading
import datetime
import struct
import queue
import socketserver
import time
# import pytz
import traceback
import zlib
import numpy as np

from Logging import log_print

# INITIAL_IP_ADDRESS = "127.0.0.1"
INITIAL_IP_ADDRESS = "0.0.0.0"
INITIAL_PORT = 22222

COMMUNICATION_SIZE = 1 + 4 + 8 + 4 + 4 # u8 + u32 + f64 + u32 + u32

SEND_TCP_CONNECTION_SLEEP_TIME = 0.05 # in s
RECEIVE_TCP_CONNECTION_SLEEP_TIME = 0.001 # in s

PING_TIME = 8 # in s

MESSAGE_ID_START = 0
# MESSAGE_ID_START = 247122 # debug

G_COMMUNICATION_HL2 = None

G_REMAINING_RECEIVED_BUFFER = b"" # used for recvexactly function, will work only if there is only one client

def recvexactly(request, size):
    # buffers = []
    # current_size = 0
    global G_REMAINING_RECEIVED_BUFFER
    buffers = [G_REMAINING_RECEIVED_BUFFER]
    current_size = len(G_REMAINING_RECEIVED_BUFFER)

    while current_size < size:
        buffer = request.recv(size)
        if len(buffer) == 0:
            return buffer
        current_size += len(buffer)
        buffers.append(buffer)

    # if current_size > size:
        # log_print(f"Error recvexactly buffer size bigger than expected {current_size} > {size}")
    # return b"".join(buffers)

    final_buffer = b"".join(buffers)
    G_REMAINING_RECEIVED_BUFFER = final_buffer[size:]
    return final_buffer[:size]

def create_communication_hl2():
    global G_COMMUNICATION_HL2
    G_COMMUNICATION_HL2 = CommunicationHL2()
    return G_COMMUNICATION_HL2

class Communication:
    ACTION_PING = 0
    ACTION_SEND_IMAGE = 1
    ACTION_SEND_FRONT_LEFT_IMAGE = 2
    ACTION_SEND_FRONT_RIGHT_IMAGE = 3
    ACTION_SEND_AHAT_DEPTH_IMAGE = 4
    ACTION_SEND_LT_DEPTH_IMAGE = 5
    ACTION_SEND_QR_CODE_POSITION = 6
    ACTION_SEND_LUT_CAMERA_PROJECTION = 7
    ACTION_SEND_SEED_POSITION = 8
    ACTION_COUNT = 9
    ACTION_INVALID = 10

    next_message_id_mutex = threading.Lock()
    next_message_id = MESSAGE_ID_START

    def __init__(self, action = None, timestamp = None):
        self.action = action
        self.id = None
        self.timestamp = timestamp
        self.message_size = None
        self.mandatory_to_receive = False

    @staticmethod
    def receive(request):
        buffer = recvexactly(request, COMMUNICATION_SIZE)
        if len(buffer) == 0: # closed connection
            log_print("error connection receive")
            return None

        action, message_id, timestamp, message_size, compressed_message_size = struct.unpack('<BLdLL', buffer)  # uint8, uint32, float64, uint32, uint32 # '>' big-endian, '<' little-endian
        # datetime.datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)
        # tz = pytz.timezone('Europe/Amsterdam')
        # timestamp = datetime.datetime.fromtimestamp(timestamp, tz)
        timestamp = datetime.datetime.fromtimestamp(timestamp)
        # log_print(f"receive communication {message_id} {action} {timestamp} {message_size} {compressed_message_size}")

        communication = None
        if action == Communication.ACTION_PING:
            communication = Communication(action, timestamp)
        elif action == Communication.ACTION_SEND_IMAGE:
            communication = CommunicationImage()
        elif action == Communication.ACTION_SEND_FRONT_LEFT_IMAGE:
            communication = CommunicationFrontLeftImage()
        elif action == Communication.ACTION_SEND_FRONT_RIGHT_IMAGE:
            communication = CommunicationFrontRightImage()
        elif action == Communication.ACTION_SEND_AHAT_DEPTH_IMAGE:
            communication = CommunicationAHATDepthImage()
        elif action == Communication.ACTION_SEND_LT_DEPTH_IMAGE:
            communication = CommunicationLTDepthImage()
        elif action == Communication.ACTION_SEND_QR_CODE_POSITION:
            communication = CommunicationQRCode()
        elif action == Communication.ACTION_SEND_LUT_CAMERA_PROJECTION:
            communication = CommunicationLUTCameraProjection()
        communication.timestamp = timestamp
        communication.id = message_id
        communication.message_size = message_size

        if message_size > 0:
            buffer = recvexactly(request, compressed_message_size)
            if len(buffer) == 0: # closed connection
                log_print("error connection receive 2")
                return None
            if compressed_message_size != message_size:
                buffer = zlib.decompress(buffer)
            inc = communication.receive_specific(buffer)
            if inc < 0:
                return None

        return communication

    @staticmethod
    def get_next_message_id():
        Communication.next_message_id_mutex.acquire()
        message_id = Communication.next_message_id
        Communication.next_message_id = Communication.next_message_id + 1
        Communication.next_message_id_mutex.release()
        return message_id

    def send(self, request):
        message_id = Communication.get_next_message_id()
        message_size = self.get_message_size()
        # log_print(f"send message {self.action} {message_id} {message_size} {self.timestamp}")
        request.sendall(struct.pack('>BLdLL', self.action, message_id, self.timestamp.timestamp(), message_size, message_size)) # uint8, uint32, float64, uint32, uint32

        self.send_specific(request)

    def get_message_size(self):
        return 0

    def send_specific(self, request):
        pass

    # Return the number of bytes read if no error occured, else -1
    def receive_specific(self, buffer):
        return 0

class CommunicationCommonImage(Communication):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_INVALID

    def initialize(self):
        self.width = None
        self.height = None
        self.translation_rig_to_world = None
        self.rotation_rig_to_world = None
        self.rig_to_camera = None
        self.focal_length_x = None
        self.focal_length_y = None
        self.center_coordinate_x = None
        self.center_coordinate_y = None
        self.distortion_coefficients = None
        self.downsampling_factor = None

    def receive_specific(self, buffer):
        try:
            inc = 0
            size = 0
            size += 2 # qu16 width
            size += 2 # qu16 height
            size += 3*4 # 3*qf32 translation_rig_to_world
            size += 4*4 # 4*qf32 rotation_rig_to_world
            size += 4*4*4 # mat44f32 rig_to_camera
            size += 8 # f64 focal_length_x
            size += 8 # f64 focal_length_y
            size += 8 # f64 center_coordinate_x
            size += 8 # f64 center_coordinate_y
            size += 5*8 # 5*qf64 distortion_coefficients
            size += 1 # qu8 downsampling_factor
            parameters = struct.unpack('<HH3f4f16fdddd5dB', buffer[:size]) # uint16, uint16, float32, float64 uint8 # '>' big-endian, '<' little-endian
            param_id = 0
            self.width = parameters[param_id]
            param_id += 1
            self.height = parameters[param_id]
            param_id += 1
            self.translation_rig_to_world = parameters[param_id:param_id + 3]
            param_id += 3
            self.rotation_rig_to_world = parameters[param_id:param_id + 4]
            param_id += 4
            self.rig_to_camera = np.array(parameters[param_id:param_id + 16]).reshape((4,4))
            param_id += 16
            self.focal_length_x = parameters[param_id]
            param_id += 1
            self.focal_length_y = parameters[param_id]
            param_id += 1
            self.center_coordinate_x = parameters[param_id]
            param_id += 1
            self.center_coordinate_y = parameters[param_id]
            param_id += 1
            self.distortion_coefficients = parameters[param_id:param_id + 5]
            param_id += 5
            self.downsampling_factor = parameters[param_id]
            param_id += 1
            inc += size
            # print(f"{self.focal_length_x} {self.focal_length_y} {self.center_coordinate_x} {self.center_coordinate_y}")

        except struct.error as e:
            log_print("EXCEPTION CommunicationCommonImage before")
            log_print(str(e))
            log_print("EXCEPTION CommunicationCommonImage after")
            # traceback.print_exc()
            traceback.format_exc()
            # log_print(sys.exc_info()[2])
            # logging.exception("Error happened")
            self.initialize()
            return -1

        return inc

class CommunicationImage8(CommunicationCommonImage):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_INVALID

    def initialize(self):
        super().initialize()
        self.nb_channel = None
        self.image = None

    def receive_specific(self, buffer):
        inc = super().receive_specific(buffer)
        if inc < 0:
            return -1

        try:
            size = 0
            size += 1 # qu8 nb_channel
            parameters = struct.unpack('<B', buffer[inc:inc + size]) # uint8 # '>' big-endian, '<' little-endian
            param_id = 0
            self.nb_channel = parameters[param_id]
            param_id += 1
            inc += size

            size = self.width*self.height*self.nb_channel
            if len(buffer[inc:inc + size]) != size:
                log_print(f"Error CommunicationImage8 receive_specific buffer size {len(buffer[inc:inc + size])} != expected size {size}")
            self.image = struct.unpack(f"<{size}B", buffer[inc:inc + size]) # uint8 '>' big-endian, '<' little-endian
            inc += size
            # log_print(f"len(image) {len(self.image)} type(image[0]) {type(self.image[0])}")
            # log_print(f"{self.image[0]}{self.image[1]}{self.image[2]}{self.image[3]}{self.image[4]}{self.image[5]}{self.image[6]}{self.image[7]}{self.image[8]}{self.image[9]}")
            if self.nb_channel == 1:
                self.image = np.array(self.image, dtype=np.uint8).reshape((self.height,self.width))
            else:
                self.image = np.array(self.image, dtype=np.uint8).reshape((self.height,self.width,self.nb_channel))

        except struct.error as e:
            log_print("EXCEPTION CommunicationImage8 before")
            log_print(str(e))
            log_print("EXCEPTION CommunicationImage8 after")
            # traceback.print_exc()
            traceback.format_exc()
            # log_print(sys.exc_info()[2])
            # logging.exception("Error happened")
            self.initialize()
            return -1

        return inc

class CommunicationDepthImage16(CommunicationCommonImage):
    BUFFER_DEPTH = 1
    BUFFER_ACTIVE_BRIGHTNESS = 2

    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_INVALID

    def initialize(self):
        super().initialize()
        self.depth_buffer = None
        self.active_brightness_buffer = None

    def receive_specific(self, buffer):
        inc = super().receive_specific(buffer)
        if inc < 0:
            return -1

        try:
            size = 1 # qu8 buffer_flags
            parameters = struct.unpack('<B', buffer[inc:inc + size]) # uint8 # '>' big-endian, '<' little-endian
            buffer_flags = parameters[0]
            inc += size

            if buffer_flags & CommunicationDepthImage16.BUFFER_DEPTH:
                size = self.width*self.height*2 # qu16
                if len(buffer[inc:inc + size]) != size:
                    log_print(f"Error CommunicationDepthImage16 receive_specific buffer size {len(buffer[inc:inc + size])} != expected size {size}")
                self.depth_buffer = struct.unpack(f"<{size//2}H", buffer[inc:inc + size]) # uint16 '>' big-endian, '<' little-endian
                # log_print(f"len(image) {len(self.depth_buffer)} type(image[0]) {type(self.depth_buffer[0])}")
                inc += size
                self.depth_buffer = np.array(self.depth_buffer, dtype=np.uint16).reshape((self.height,self.width))

            if buffer_flags & CommunicationDepthImage16.BUFFER_ACTIVE_BRIGHTNESS:
                size = self.width*self.height*2 # qu16
                if len(buffer[inc:inc + size]) != size:
                    log_print(f"Error CommunicationDepthImage16 receive_specific buffer size {len(buffer[inc:inc + size])} != expected size {size}")
                self.active_brightness_buffer = struct.unpack(f"<{size//2}H", buffer[inc:inc + size]) # uint16 '>' big-endian, '<' little-endian
                inc += size
                self.active_brightness_buffer = np.array(self.active_brightness_buffer, dtype=np.uint16).reshape((self.height,self.width))

        except struct.error as e:
            log_print("EXCEPTION CommunicationDepthImage16 before")
            log_print(str(e))
            log_print("EXCEPTION CommunicationDepthImage16 after")
            # traceback.print_exc()
            traceback.format_exc()
            # log_print(sys.exc_info()[2])
            # logging.exception("Error happened")
            self.initialize()
            return -1

        return inc

class CommunicationImage(CommunicationImage8):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_IMAGE
        self.initialize()

class CommunicationFrontLeftImage(CommunicationImage8):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_FRONT_LEFT_IMAGE
        self.initialize()

class CommunicationFrontRightImage(CommunicationImage8):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_FRONT_RIGHT_IMAGE
        self.initialize()

class CommunicationAHATDepthImage(CommunicationDepthImage16):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_AHAT_DEPTH_IMAGE
        self.initialize()

class CommunicationLTDepthImage(CommunicationDepthImage16):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_LT_DEPTH_IMAGE
        self.initialize()

class CommunicationQRCode(Communication):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_QR_CODE_POSITION
        self.initialize()

    def initialize(self):
        self.qr_code_ids = None
        # self.qr_code_translations = None
        # self.qr_code_rotations = None
        self.qr_code_transforms = None

    def receive_specific(self, buffer):
        try:
            inc = 0
            size = 0
            size += 1 # qu8 nb qrcode
            parameters = struct.unpack('<B', buffer[inc:inc + size]) # uint8 # '>' big-endian, '<' little-endian
            param_id = 0
            nb_qrcode = parameters[param_id]
            param_id += 1
            inc += size

            if nb_qrcode > 0:
                self.qr_code_ids = []
                # self.qr_code_translations = []
                # self.qr_code_rotations = []
                self.qr_code_transforms = []
                for _ in range(nb_qrcode):
                    # size = 1*1 + 2*3*8 # id*sizeof(qu8) + (translation + rotation)*vec3*sizeof(qf64)
                    # transformation = struct.unpack('<B3d3d', buffer[inc:inc + size]) # uint8 float64 # '>' big-endian, '<' little-endian
                    size = 1*1 + 16*4 # id*sizeof(qu8) + (matrix)*sizeof(qf32)
                    transformation = struct.unpack('<B16f', buffer[inc:inc + size]) # uint8 float32 # '>' big-endian, '<' little-endian
                    self.qr_code_ids.append(transformation[0])
                    # translation = transformation[1:4]
                    # rotation = transformation[4:]
                    # self.qr_code_translations.append(translation)
                    # self.qr_code_rotations.append(rotation)
                    # transform = np.array(transformation[1:17]).reshape((4,4))
                    transform = transformation[1:17]
                    self.qr_code_transforms.append(transform)
                    inc += size

        except struct.error as e:
            log_print("EXCEPTION CommunicationQRCode before")
            log_print(str(e))
            log_print("EXCEPTION CommunicationQRCode after")
            # traceback.print_exc()
            traceback.format_exc()
            # log_print(sys.exc_info()[2])
            # logging.exception("Error happened")
            self.initialize()
            return -1

        return inc

class CommunicationLUTCameraProjection(Communication):
    RM_SENSOR_VL_FRONT_LEFT = 0
    RM_SENSOR_VL_FRONT_RIGHT = 1
    RM_SENSOR_AHAT_DEPTH = 2
    RM_SENSOR_LT_DEPTH = 3
    RM_SENSOR_COUNT = 4

    FLAGS = [1,2,4,8]

    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_LUT_CAMERA_PROJECTION
        self.mandatory_to_receive = True
        self.initialize()

    def initialize(self):
        self.lut_camera_projection_x = [None,None,None,None]
        self.lut_camera_projection_y = [None,None,None,None]

    def receive_specific(self, buffer):
        try:
            inc = 0
            size = 0
            size += 1 # qu8 rm_sensor_flag
            parameters = struct.unpack('<B', buffer[inc:inc + size]) # uint8 # '>' big-endian, '<' little-endian
            param_id = 0
            rm_sensor_flag = parameters[param_id]
            # log_print(f"rm_sensor_flag {rm_sensor_flag}")
            param_id += 1
            inc += size

            if rm_sensor_flag > 0:
                for i in range(CommunicationLUTCameraProjection.RM_SENSOR_COUNT):
                    if rm_sensor_flag & CommunicationLUTCameraProjection.FLAGS[i]:
                        width, height = CommunicationLUTCameraProjection.get_camera_size(i)
                        # log_print(f"width {width} height {height}")
                        lut_size = (width*2 + 1)*(height*2 + 1)
                        size = lut_size*4 # lut_size*sizeof(qf32)

                        self.lut_camera_projection_x[i] = struct.unpack(f"<{lut_size}f", buffer[inc:inc + size]) # float32 # '>' big-endian, '<' little-endian
                        self.lut_camera_projection_x[i] = np.array(self.lut_camera_projection_x[i], dtype=np.float32).reshape((height*2 + 1,width*2 + 1))
                        inc += size

                        self.lut_camera_projection_y[i] = struct.unpack(f"<{lut_size}f", buffer[inc:inc + size]) # float32 # '>' big-endian, '<' little-endian
                        self.lut_camera_projection_y[i] = np.array(self.lut_camera_projection_y[i], dtype=np.float32).reshape((height*2 + 1,width*2 + 1))
                        inc += size

        except struct.error as e:
            log_print("EXCEPTION CommunicationLUTCameraProjection before")
            log_print(str(e))
            log_print("EXCEPTION CommunicationLUTCameraProjection after")
            # traceback.print_exc()
            traceback.format_exc()
            # log_print(sys.exc_info()[2])
            # logging.exception("Error happened")
            self.initialize()
            return -1

        return inc

    @staticmethod
    def get_camera_size(rm_sensor):
        if rm_sensor in [CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT, CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]:
            return 640, 480
        if rm_sensor == CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH:
            return 512, 512
        if rm_sensor == CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH:
            return 320, 288
        assert False

class CommunicationSeed(Communication):
    def __init__(self):
        super().__init__()
        self.action = Communication.ACTION_SEND_SEED_POSITION
        self.seed_pos_w = [0,0,0]

    def get_message_size(self):
        message_size = 8*3 # x float64 y float64 z float64
        return message_size

    def send_specific(self, request):
        request.sendall(struct.pack('>ddd', self.seed_pos_w[0], self.seed_pos_w[1], self.seed_pos_w[2])) # float64, float64, float64

class CommunicationHL2:
    def __init__(self):
        self.host = INITIAL_IP_ADDRESS
        self.port = INITIAL_PORT
        self.server_thread = None
        self.sock = None

        self.ready_to_send = False
        self.queue_data_to_send = queue.Queue()
        self.ready_to_receive = False
        self.queue_data_to_receive = queue.Queue()

        self.stop_communication_now = False

    def run_routine(self):
        log_print("run_routine begin")
        with socketserver.ThreadingTCPServer((self.host, self.port), LoggerHandler) as sock:
            self.sock = sock
            log_print("run_routine with")
            sock.serve_forever()
            log_print("run_routine with after serve_forever")

        log_print("run_routine end")

    def start(self):
        # self.stop_server = False
        self.server_thread = threading.Thread(target=self.run_routine)
        self.server_thread.daemon = True
        self.server_thread.start()

    def stop(self):
        log_print("CommunicationHL2::stop")
        # TODO do a clean stop, for now don't know what happen when shutdown is called
        self.stop_communication_now = True
        if self.sock is not None:
            self.sock.shutdown()
            self.sock.server_close()

class LoggerHandler(socketserver.BaseRequestHandler):
    global G_COMMUNICATION_HL2

    def __init__(self, request, client_address, server):
        log_print("LoggerHandler.__init__")
        super().__init__(request, client_address, server)

    def send_routine(self):
        log_print("LoggerHandler.send_routine")

        last_ping_time = datetime.datetime(1971,1,1,0,0,0,0)
        # last_ping_time = datetime.datetime(1971,1,1,0,0,0,0,pytz.UTC)
        while True:
            time.sleep(SEND_TCP_CONNECTION_SLEEP_TIME)

            timestamp = datetime.datetime.now()
            # timestamp = timestamp.replace(tzinfo=pytz.utc)
            if timestamp - last_ping_time > datetime.timedelta(seconds=PING_TIME):
                last_ping_time = timestamp
                communication = Communication(Communication.ACTION_PING, timestamp)
                communication.send(self.request)

            G_COMMUNICATION_HL2.ready_to_send = True
            if not G_COMMUNICATION_HL2.queue_data_to_send.empty():
                communication = G_COMMUNICATION_HL2.queue_data_to_send.get()
                communication.send(self.request)

            if G_COMMUNICATION_HL2.stop_communication_now:
                break

    def handle(self):
        log_print("LoggerHandler.handle")

        self.send_thread = threading.Thread(target=self.send_routine)
        self.send_thread.daemon = True
        self.send_thread.start()

        while True:
            # time.sleep(RECEIVE_TCP_CONNECTION_SLEEP_TIME)

            communication = Communication.receive(self.request)
            if communication is not None:
                if communication.action == Communication.ACTION_PING:
                    dt = communication.timestamp.strftime('%H:%M:%S.%f') # [:-3] # strip off microseconds
                    log_print(f"receive ping {communication.id} {dt} {communication.timestamp}")
                elif G_COMMUNICATION_HL2.ready_to_receive or communication.mandatory_to_receive:
                    # log_print("add new message to the queue")
                    G_COMMUNICATION_HL2.queue_data_to_receive.put(communication)
            # else:
                # log_print("error LoggerHandler.handle receive")
                # break

            if G_COMMUNICATION_HL2.stop_communication_now:
                break

        log_print("LoggerHandler.handle end")
