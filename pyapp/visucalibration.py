import datetime
import numpy as np

import dearpygui.dearpygui as dpg
# from dearpygui.demo import show_demo

from python.common.Logging import set_logger, log_print
from python.common.UtilDearPyGui import DpgLogger, help_widget

from DataAcquisition import ACQUISITIONS
# from DataAcquisition import UTC_SHIFT, ACQUISITIONS
UTC_SHIFT = 0
from visuapp import App, CalibrationParameters, ALL_VISIBILITY

ACTIONS = {"calibration_optical_pointer":App.ACTION_CALIBRATION_OPTICAL_POINTER
    ,"calibration_optical_front_qr_code":App.ACTION_CALIBRATION_OPTICAL_FRONT_QR_CODE
    ,"calibration_optical_left_qr_code":App.ACTION_CALIBRATION_OPTICAL_LEFT_QR_CODE
    ,"calibration_optical_right_qr_code":App.ACTION_CALIBRATION_OPTICAL_RIGHT_QR_CODE
    ,"calibration_optical_top_qr_code":App.ACTION_CALIBRATION_OPTICAL_TOP_QR_CODE
    ,"calibration_optical_seed":App.ACTION_CALIBRATION_OPTICAL_SEED}

def action_callback(sender, app_data, user_data):
    log_print(sender)
    user_data.mutex_update.acquire()
    user_data.actions.append(ACTIONS[sender])
    user_data.mutex_update.release()

def update_slider_value(app, acquisition):
    for other_acquisition in ACQUISITIONS:
        if other_acquisition != acquisition:
            if not app.acquisitions[other_acquisition].empty:
                dpg.set_value(other_acquisition + "_slider", app.slider_value[other_acquisition])

def timestamp_slider_callback(sender, app_data, user_data):
    acquisition = sender[:-7] # we remove "_slider" of the widget name (e.g sender = "optical_slider")
    if user_data.slider_value[acquisition] != app_data:
        user_data.set_slider_value(acquisition, app_data)
        update_slider_value(user_data, acquisition)
        update_gui(user_data, change_timestamp = True, change_shift_optical = False, change_shift_hololens = False)

def time_offset_optical_callback(sender, app_data, user_data):
    if user_data.temporal_shift_optical != app_data:
        user_data.set_temporal_shift_optical(app_data)
        update_slider_value(user_data, "optical")
        update_gui(user_data, change_timestamp = True, change_shift_optical = True, change_shift_hololens = False)

def time_offset_hololens_callback(sender, app_data, user_data):
    if user_data.temporal_shift_hololens != app_data:
        acquisition = user_data.set_temporal_shift_hololens(app_data)
        update_slider_value(user_data, acquisition)
        update_gui(user_data, change_timestamp = True, change_shift_optical = False, change_shift_hololens = True)

def set_optical_starting_callback(sender, app_data, user_data):
    user_data.optical_calibration_starting_time = user_data.slider_value["optical"]
    if user_data.optical_calibration_ending_time < user_data.optical_calibration_starting_time:
        user_data.optical_calibration_ending_time = user_data.optical_calibration_starting_time

    update_gui(user_data, change_timestamp = True, change_shift_optical = False, change_shift_hololens = False)

def set_optical_ending_callback(sender, app_data, user_data):
    user_data.optical_calibration_ending_time = user_data.slider_value["optical"]
    if user_data.optical_calibration_ending_time < user_data.optical_calibration_starting_time:
        user_data.optical_calibration_starting_time = user_data.optical_calibration_ending_time

    update_gui(user_data, change_timestamp = True, change_shift_optical = False, change_shift_hololens = False)

def set_qr_code_starting_callback(sender, app_data, user_data):
    user_data.qr_code_calibration_starting_time = user_data.slider_value["qr_code_position"]
    if user_data.qr_code_calibration_ending_time < user_data.qr_code_calibration_starting_time:
        user_data.qr_code_calibration_ending_time = user_data.qr_code_calibration_starting_time

    update_gui(user_data, change_timestamp = True, change_shift_optical = False, change_shift_hololens = False)

def set_qr_code_ending_callback(sender, app_data, user_data):
    user_data.qr_code_calibration_ending_time = user_data.slider_value["qr_code_position"]
    if user_data.qr_code_calibration_ending_time < user_data.qr_code_calibration_starting_time:
        user_data.qr_code_calibration_starting_time = user_data.qr_code_calibration_ending_time

    update_gui(user_data, change_timestamp = True, change_shift_optical = False, change_shift_hololens = False)

def end_calibration(acquisition, cs_probe, cs_world, pos_t, pos_s, residuals, residuals_timestamp):
    dpg.set_value("tip_" + cs_probe + "_input", pos_t)
    dpg.set_value("seed_" + cs_world + "_input", pos_s)

    # print(residuals_timestamp)
    # print(residuals)
    residuals_timestamp = [(x - datetime.timedelta(hours=UTC_SHIFT)).timestamp() for x in residuals_timestamp ] # seconds since 1970
    dpg.set_value(acquisition + "_scatter_residuals", [residuals_timestamp, residuals])

def launch_calibration(app, cs_probe, cs_world, end_calibration_callback, action):
    init_seed=(0,0,0)
    if dpg.get_value('init_seed'):
        init_seed = dpg.get_value("seed_" + cs_world + "_input")[:3]
        # print(f"init_seed {init_seed}")
    init_tip=(0,0,0)
    if dpg.get_value('init_tip') or (not dpg.get_value('tip_unknown')):
        init_tip = dpg.get_value("tip_" + cs_probe + "_input")[:3]
        # print(f"init_tip {init_tip}")

    calibration_parameters = CalibrationParameters(init_seed, init_tip, dpg.get_value('tip_unknown'), end_calibration_callback)
    app.mutex_update.acquire()
    app.actions.append([action, calibration_parameters])
    app.mutex_update.release()

def end_optical_calibration_callback(pos_t_m, pos_s_o, residuals, residuals_timestamp):
    end_calibration("optical", "m", "o", pos_t_m, pos_s_o, residuals, residuals_timestamp)

def launch_optical_callback(sender, app_data, user_data):
    log_print("launch optical calibration")
    launch_calibration(user_data, "m", "o", end_optical_calibration_callback, user_data.ACTION_OPTICAL_CALIBRATION)

def end_qr_code_calibration_callback(pos_t_q, pos_s_w, residuals, residuals_timestamp):
    end_calibration("qr", "q", "w", pos_t_q, pos_s_w, residuals, residuals_timestamp)

def launch_qr_code_callback(sender, app_data, user_data):
    log_print("launch qr code calibration")
    launch_calibration(user_data, "q", "w", end_qr_code_calibration_callback, user_data.ACTION_QR_CODE_CALIBRATION)

def tip_m_callback(sender, app_data, user_data):
    user_data.mutex_update.acquire()
    user_data.pos_t_m = app_data[:3]
    user_data.has_to_update = True
    user_data.mutex_update.release()

def seed_o_callback(sender, app_data, user_data):
    user_data.mutex_update.acquire()
    user_data.pos_s_o = app_data[:3]
    user_data.has_to_update = True
    user_data.mutex_update.release()

def tip_q_callback(sender, app_data, user_data):
    user_data.mutex_update.acquire()
    # print(f"tip_q_callback app_data {type(app_data)} {app_data}")
    user_data.pos_t_q = app_data[:3]
    user_data.has_to_update = True
    user_data.mutex_update.release()

def seed_w_callback(sender, app_data, user_data):
    user_data.mutex_update.acquire()
    user_data.pos_s_w = app_data[:3]
    user_data.has_to_update = True
    user_data.mutex_update.release()

def visibility_callback(sender, app_data, user_data):
    # print(f"visibility_callback {sender} {app_data}")
    acquisition = sender[11:] # we remove "visibility " of the widget name
    user_data.mutex_update.acquire()
    user_data.visibility[acquisition] = app_data
    user_data.has_to_update = True
    user_data.mutex_update.release()

def scatter_callback(sender, app_data, user_data):
    # print(f"scatter_callback {sender} {app_data}")
    plot_name = sender[:-9] # we remove " checkbox" of the widget name (e.g sender = "Magnetic seed checkbox")
    if app_data:
        timestamps = dpg.get_value(plot_name + " line")[0]
        distances = dpg.get_value(plot_name + " line")[1]
        dpg.add_scatter_series(timestamps, distances, label=plot_name, tag=plot_name + " scatter", parent="plot_axis")
    else:
        # dpg.disable_item(plot_name + " scatter")
        dpg.delete_item(plot_name + " scatter") #, children_only=True, slot=1)

def get_time_seed_distances(df_magnetic_seed):
    timestamps = [x.timestamp() for x in df_magnetic_seed.index.to_pydatetime()] # seconds since 1970
    distances = list(df_magnetic_seed['distance'])
    # distances = list(df_magnetic_seed['angle'])
    # distances = list(df_magnetic_seed['azimut'])
    return timestamps, distances

def get_time_plot(df, column, only_timestamps=False, direction=-1):
    if df.index.size == 0:
        return [], []

    distances = None
    if not only_timestamps:
        distances = direction*np.array(df[column])
        # min_point = np.percentile(distances, 5)
        # max_point = np.percentile(distances, 95)
        min_point = np.min(distances)
        max_point = np.max(distances)
        distances[distances < min_point] = min_point
        distances[distances > max_point] = max_point
        distances = (distances - min_point)/(max_point - min_point) # normalize [0,1]
        distances = 45*distances + 10 # range [10,55], same as pintuition tracker
        distances = list(distances)
    timestamps = [x.timestamp() for x in df.index.to_pydatetime()] # seconds since 1970
    return timestamps, distances

def update_gui(app, change_timestamp, change_shift_optical, change_shift_hololens):
    if change_timestamp:
        for acquisition in ACQUISITIONS:
            if not app.acquisitions[acquisition].empty:
                timestamp = app.df[acquisition].index[app.slider_value[acquisition]]
                break
        slider_timestamp = (timestamp - datetime.timedelta(hours=UTC_SHIFT)).timestamp()

        dpg.set_value("text_timestamp", f"Timestamp {timestamp}")

        # print(f"dpg.get_value('vline_slider_timestamp') {dpg.get_value('vline_slider_timestamp')}")
        dpg.set_value("vline_slider_timestamp", [[slider_timestamp]])
        # print(f"dpg.get_value('vline_slider_timestamp') 2 {dpg.get_value('vline_slider_timestamp')}")

        if not app.acquisitions["magnetic_seed"].empty:
            if not app.acquisitions["optical"].empty:
                optical_calibration_starting_timestamp = (app.df["optical"].index[app.optical_calibration_starting_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()
                optical_calibration_ending_timestamp = (app.df["optical"].index[app.optical_calibration_ending_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()
                dpg.set_value("vline_optical_calibration_starting_timestamp", [[optical_calibration_starting_timestamp]])
                dpg.set_value("vline_optical_calibration_ending_timestamp", [[optical_calibration_ending_timestamp]])

            if not app.acquisitions["qr_code_position"].empty:
                qr_code_calibration_starting_timestamp = (app.df["qr_code_position"].index[app.qr_code_calibration_starting_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()
                qr_code_calibration_ending_timestamp = (app.df["qr_code_position"].index[app.qr_code_calibration_ending_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()
                dpg.set_value("vline_qr_code_calibration_starting_timestamp", [[qr_code_calibration_starting_timestamp]])
                dpg.set_value("vline_qr_code_calibration_ending_timestamp", [[qr_code_calibration_ending_timestamp]])

    names = []
    dfs = []
    columns = []
    directions = []

    if change_shift_optical:
        names.extend(["Optical probe", "Optical pointer", "Optical seed"])
        dfs.extend([app.df_probe, app.df_pointer, app.df_optical_seed])
        columns.extend(['tx 1', 'tx 2', 'tx 3'])
        directions.extend([-1,-1,-1])

    if change_shift_hololens:
        names.extend(["QR code"])
        dfs.extend([app.df["qr_code_position"]])
        # columns.extend(['qr code 1 translation y'])
        columns.extend(['q1_m24'])
        directions.extend([-1])

    for i, df in enumerate(dfs):
        name_line = f"{names[i]} line"
        # print(f"dpg.get_value({name_line}) {dpg.get_value(name_line)}")
        timestamps, distances = get_time_plot(df, columns[i], only_timestamps=True, direction=directions[i])
        distances = dpg.get_value(name_line)[1]
        dpg.set_value(name_line, [timestamps, distances])
        name_scatter = f"{names[i]} scatter"
        # print(f"does_alias_exist {dpg.does_alias_exist(name_scatter)} does_item_exist {dpg.does_item_exist(name_scatter)}")
        if dpg.does_alias_exist(name_scatter):
            dpg.set_value(name_scatter, [timestamps, distances])

    app.mutex_update.acquire()
    app.has_to_update = True
    app.mutex_update.release()

def start_test_gui(app):
    dpg.create_context()
    dpg.create_viewport(title='Visualization', width=840, height=900+30, x_pos=0, y_pos=0)

    set_logger(DpgLogger([2,690],820,199))

    log_print("start logger")

    # show_demo()

    main_window_id = dpg.window(label="OSAR",width=820,height=686,pos=[2,2])
    with main_window_id:
        dpg.add_text("timestamp", tag="text_timestamp")
        for acquisition in ACQUISITIONS:
            if not app.acquisitions[acquisition].empty:
                # width=765
                dpg.add_drag_int(label=acquisition , width=670, min_value=0, max_value=(app.df[acquisition].index.size - 1), callback=timestamp_slider_callback, user_data=app, tag=acquisition + "_slider")
        help_widget("Click and drag to edit value.\n"
              "Hold SHIFT/ALT for faster/slower edit.\n"
              "Double-click or CTRL+click to input value.")

        with dpg.group(horizontal=True):
            if not app.acquisitions["optical"].empty:
                dpg.add_drag_float(width=70, label="time offset in s (optical tracker)", default_value=app.temporal_shift_optical, min_value=-120, max_value=120, callback=time_offset_optical_callback, user_data=app)
            if app.available_hololens_acquisitions():
                dpg.add_drag_float(width=70, label="time offset in s (hololens)", default_value=app.temporal_shift_hololens, min_value=-120, max_value=120, callback=time_offset_hololens_callback, user_data=app)

        if not app.acquisitions["magnetic_seed"].empty:
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="init seed", tag="init_seed")
                dpg.add_checkbox(label="init tip", tag="init_tip")
                dpg.add_checkbox(label="tip unknown", tag="tip_unknown", default_value=True)

            if not app.acquisitions["optical"].empty:
                with dpg.group(horizontal=True):
                    dpg.add_text("optical")
                    dpg.add_button(label="starting", callback=set_optical_starting_callback, user_data=app)
                    dpg.add_button(label="ending", callback=set_optical_ending_callback, user_data=app)
                    dpg.add_button(label="launch", callback=launch_optical_callback, user_data=app, tag="launch_optical")
                    dpg.add_input_floatx(label="tip m", width=220, size=3, callback=tip_m_callback, user_data=app, default_value=app.pos_t_m, tag="tip_m_input")
                    dpg.add_input_floatx(label="seed o", width=220, size=3,  callback=seed_o_callback, user_data=app, default_value=app.pos_s_o, tag="seed_o_input")

            if not app.acquisitions["qr_code_position"].empty:
                with dpg.group(horizontal=True):
                    dpg.add_text("qr code")
                    dpg.add_button(label="starting", callback=set_qr_code_starting_callback, user_data=app)
                    dpg.add_button(label="ending", callback=set_qr_code_ending_callback, user_data=app)
                    dpg.add_button(label="launch", callback=launch_qr_code_callback, user_data=app, tag="launch_qr_code")
                    dpg.add_input_floatx(label="tip q", width=220, size=3,  callback=tip_q_callback, user_data=app, default_value=app.pos_t_q, tag="tip_q_input")
                    dpg.add_input_floatx(label="seed w", width=220, size=3,  callback=seed_w_callback, user_data=app, default_value=app.pos_s_w, tag="seed_w_input")

        with dpg.group(horizontal=True):
            for acquisition in ALL_VISIBILITY[:4]:
                if not app.acquisitions[acquisition].empty:
                    dpg.add_checkbox(label=acquisition, callback=visibility_callback, user_data=app, tag="visibility " + acquisition, default_value=app.visibility[acquisition])
        with dpg.group(horizontal=True):
            for acquisition in ALL_VISIBILITY[4:]:
                if (acquisition == "ahat_ab" and not app.acquisitions["ahat_depth_cam"].empty) \
                or (acquisition == "lt_ab" and not app.acquisitions["lt_depth_cam"].empty) \
                or (acquisition in app.acquisitions and not app.acquisitions[acquisition].empty):
                    dpg.add_checkbox(label=acquisition, callback=visibility_callback, user_data=app, tag="visibility " + acquisition, default_value=app.visibility[acquisition])

        with dpg.group(horizontal=True):
            dpg.add_text("scatter plot")
            if not app.acquisitions["magnetic_seed"].empty:
                dpg.add_checkbox(label="seed distance", callback=scatter_callback, user_data=app, tag="Magnetic seed checkbox")
            if not app.acquisitions["probe"].empty:
                dpg.add_checkbox(label="-x optical probe", callback=scatter_callback, user_data=app, tag="Optical probe checkbox")
            if not app.acquisitions["pointer"].empty:
                dpg.add_checkbox(label="-x optical pointer", callback=scatter_callback, user_data=app, tag="Optical pointer checkbox")
            if not app.acquisitions["optical_seed"].empty:
                dpg.add_checkbox(label="-x optical seed", callback=scatter_callback, user_data=app, tag="Optical seed checkbox")
            if not app.acquisitions["qr_code_position"].empty:
                dpg.add_checkbox(label="y qr code probe", callback=scatter_callback, user_data=app, tag="QR code checkbox")

        if not app.acquisitions["pointer"].empty:
            with dpg.group(horizontal=True):
                dpg.add_text("calib optical")
                dpg.add_button(label="pointer", callback=action_callback, user_data=app, tag="calibration_optical_pointer")
                if not app.acquisitions["probe"].empty:
                    dpg.add_text("qr code")
                    dpg.add_button(label="front", callback=action_callback, user_data=app, tag="calibration_optical_front_qr_code")
                    dpg.add_button(label="left", callback=action_callback, user_data=app, tag="calibration_optical_left_qr_code")
                    dpg.add_button(label="right", callback=action_callback, user_data=app, tag="calibration_optical_right_qr_code")
                    dpg.add_button(label="top", callback=action_callback, user_data=app, tag="calibration_optical_top_qr_code")
                if not app.acquisitions["optical_seed"].empty:
                    dpg.add_button(label="seed", callback=action_callback, user_data=app, tag="calibration_optical_seed")

        if not app.acquisitions["magnetic_seed"].empty:
            timestamps_seed, distances_seed = get_time_seed_distances(app.acquisitions["magnetic_seed"])

        if not app.acquisitions["optical"].empty:
            timestamps_optical_probe, distances_optical_probe = get_time_plot(app.df_probe, 'tx 1')
            timestamps_optical_pointer, distances_optical_pointer = get_time_plot(app.df_pointer, 'tx 2')
            timestamps_optical_seed, distances_optical_seed = get_time_plot(app.df_optical_seed, 'tx 3')

            if not app.acquisitions["magnetic_seed"].empty:
                optical_calibration_starting_timestamp = (app.df["optical"].index[app.optical_calibration_starting_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()
                optical_calibration_ending_timestamp = (app.df["optical"].index[app.optical_calibration_ending_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()

        if not app.acquisitions["qr_code_position"].empty:
            # timestamps_qr_code, distances_qr_code = get_time_plot(app.df["qr_code_position"], 'qr code 1 translation y', only_timestamps=False, direction=1)
            timestamps_qr_code, distances_qr_code = get_time_plot(app.df["qr_code_position"], 'q1_m24', only_timestamps=False, direction=-1)
            distances_qr_code = [10 + 50 - x for x in distances_qr_code] # y axis for the camera is going down, range [10,55]


            if not app.acquisitions["magnetic_seed"].empty:
                qr_code_calibration_starting_timestamp = (app.df["qr_code_position"].index[app.qr_code_calibration_starting_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()
                qr_code_calibration_ending_timestamp = (app.df["qr_code_position"].index[app.qr_code_calibration_ending_time] - datetime.timedelta(hours=UTC_SHIFT)).timestamp()

        for acquisition in ACQUISITIONS:
            if not app.acquisitions[acquisition].empty:
                timestamp = app.df[acquisition].index[app.slider_value[acquisition]]
                break
        slider_timestamp = (timestamp - datetime.timedelta(hours=UTC_SHIFT)).timestamp()
        # print(f"slider_timestamp {slider_timestamp} timestamps_seed[0] {timestamps_seed[0]}")

        # with dpg.plot(label="Time Plot", height=350, width=-1):
        with dpg.plot(height=420, width=-1):
            # xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="Date", time=True)
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, time=True)
            with dpg.plot_axis(dpg.mvYAxis, label="distance in mm", tag="plot_axis"):
                dpg.add_vline_series((slider_timestamp,), label="current timestamp", tag="vline_slider_timestamp")

                if not app.acquisitions["magnetic_seed"].empty:
                    dpg.add_line_series(timestamps_seed, distances_seed, label="Magnetic seed", tag="Magnetic seed line")

                if not app.acquisitions["optical"].empty:
                    dpg.add_line_series(timestamps_optical_probe, distances_optical_probe, label="Optical probe", tag="Optical probe line")
                    dpg.add_line_series(timestamps_optical_pointer, distances_optical_pointer, label="Optical pointer", tag="Optical pointer line")
                    dpg.add_line_series(timestamps_optical_seed, distances_optical_seed, label="Optical seed", tag="Optical seed line")
                    if not app.acquisitions["magnetic_seed"].empty:
                        dpg.add_vline_series((optical_calibration_starting_timestamp,), label="Optical calibration starting", tag="vline_optical_calibration_starting_timestamp")
                        dpg.add_vline_series((optical_calibration_ending_timestamp,), label="Optical calibration ending", tag="vline_optical_calibration_ending_timestamp")
                        # dpg.add_scatter_series([timestamps_seed[0]], [0], label="Optical residuals", tag="optical_scatter_residuals")
                        dpg.add_scatter_series([timestamps_optical_probe[0]], [0], label="Optical residuals", tag="optical_scatter_residuals")

                if not app.acquisitions["qr_code_position"].empty:
                    dpg.add_line_series(timestamps_qr_code, distances_qr_code, label="QR code", tag="QR code line")
                    if not app.acquisitions["magnetic_seed"].empty:
                        dpg.add_vline_series((qr_code_calibration_starting_timestamp,), label="QR code calibration starting", tag="vline_qr_code_calibration_starting_timestamp")
                        dpg.add_vline_series((qr_code_calibration_ending_timestamp,), label="QR code calibration ending", tag="vline_qr_code_calibration_ending_timestamp")
                        # dpg.add_scatter_series([timestamps_seed[0]], [0], label="QR code residuals", tag="qr_scatter_residuals")
                        dpg.add_scatter_series([timestamps_qr_code[0]], [0], label="QR code residuals", tag="qr_scatter_residuals")

                dpg.fit_axis_data(dpg.top_container_stack())
            dpg.fit_axis_data(xaxis)

        # with dpg.plot(height=180, width=-1):
        #     xaxis = dpg.add_plot_axis(dpg.mvXAxis, time=True)
        #     with dpg.plot_axis(dpg.mvYAxis, label="in mm"):
        #         dpg.add_scatter_series([timestamps_seed[0]], [0], label="Residuals", tag="optical_scatter_residuals")
        #         dpg.fit_axis_data(dpg.top_container_stack())
        #     dpg.fit_axis_data(xaxis)

        dpg.add_text("Left button + drag         : Rotate")
        dpg.add_text("Ctrl + left button + drag  : Translate")
        dpg.add_text("Wheel button + drag        : Translate")
        dpg.add_text("Shift + left button + drag : Roll")
        dpg.add_text("Wheel                      : Zoom in/out")
        dpg.add_text("[/]          : Increase/decrease field of view")
        dpg.add_text("R            : Reset view point")
        dpg.add_text("Ctrl/Cmd + C : Copy current view status into the clipboard")
        dpg.add_text("Ctrl/Cmd + V : Paste view status from clipboard")
        dpg.add_text("Q, Esc       : Exit window")
        dpg.add_text("H            : Print help message")
        dpg.add_text("P, PrtScn    : Take a screen capture")
        dpg.add_text("D            : Take a depth capture")
        dpg.add_text("O            : Take a capture of current rendering settings")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if  __name__ == '__main__':
    print("start")

    if True:
    # if False:
        app = App()
        app.run()
        start_test_gui(app)
