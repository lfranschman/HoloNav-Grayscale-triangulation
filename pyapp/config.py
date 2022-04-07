import numpy as np

ORDER_DIVOT_INDEX_QR_CODE_2022_01_28 = None
DELETE_GT_DIVOT_INDEX_QR_CODE_2022_01_28 = np.array([2 ,3 ,5 ,6  \
                                                 ,8 ,9 ,10,11,12,13,14 \
                                                 ,15,16,17,18,19,20,21 \
                                                 ,23,24,26,27 \
                                                 ,29,30,31,32,33,34,35 \
                                                 ,36,37,38,39,40,41,42 \
                                                 ,44,45,47,48]) - 1

class qConfig:
    def __init__(self):
        self.path = "C:/test/"

        self.folder = self.path

        self.record = None

        self.temporal_shift_optical = 0 # in s
        self.optical_calibration_starting_time = -1
        self.optical_calibration_ending_time = -1

        self.temporal_shift_hololens = 0 # in s
        self.qr_code_calibration_starting_time = -1
        self.qr_code_calibration_ending_time = -1

        self.order_divot_index = None
        self.to_delete_gt_divot_index = None

    def get_filename(self, record=None):
        if record is not None:
            return self.folder + record + ".pickle.gz"
        return self.folder + self.record + ".pickle.gz"

class qConfig_2022_02_10_qr_code_test(qConfig):
    def __init__(self):
        super().__init__()

        self.folder = self.path + "/2022_02_10_qr_code_test/"

        # self.record = "pivot_calibration"
        # self.record = "calibration_front_qr_code"
        # self.record = "calibration_left_qr_code"
        # self.record = "calibration_right_qr_code"
        # self.record = "calibration_top_qr_code"

        self.record = "qr_code_test"
        # self.record = "qr_code_test2"

        self.temporal_shift_hololens = 6 # in s
        self.qr_code_calibration_starting_time = 8

        if self.record in ["calibration_front_qr_code", "calibration_left_qr_code", "calibration_right_qr_code", "calibration_top_qr_code"]:
            self.order_divot_index = ORDER_DIVOT_INDEX_QR_CODE_2022_01_28
            self.to_delete_gt_divot_index = DELETE_GT_DIVOT_INDEX_QR_CODE_2022_01_28

# config = qConfig()
config = qConfig_2022_02_10_qr_code_test()
