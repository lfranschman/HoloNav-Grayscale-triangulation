# HoloNav

## Python libraries to install
- numpy
- scipy
- scikit-image
- pandas
- cv2/aruco
- open3d
- dearpygui

If you use Windows, you can get python wheel here [https://www.lfd.uci.edu/~gohlke/pythonlibs](https://www.lfd.uci.edu/~gohlke/pythonlibs). It can be interesting for numpy library for example as it is compiled using Intel MKL and is much faster than the default one.

Command-line and version used:
```
python -m pip install --upgrade pip
python -m pip install cython
python -m pip install numpy-1.21.2+mkl-cp38-cp38-win_amd64.whl
python -m pip install scipy-1.7.1-cp38-cp38-win_amd64.whl
python -m pip install scikit_image-0.18.3-cp38-cp38-win_amd64.whl
python -m pip install pandas
python -m pip install opencv_python-4.5.3-cp38-cp38-win_amd64.whl
python -m pip install opencv-contrib-python
python -m pip install open3d
python -m pip install dearpygui
```

## Quick start
- Edit the file `userSpecific/globalVariables.default.bat` to change the paths and parameters fitting your configuration.
- Edit the file `pyapp/config.py` to change `self.path` where is your data folder. To change dataset change `self.record`.

- check `pyapp/processqrcode.py` for an example to find qr codes (pv camera or vl cameras) and compute distance error between optical and hololens
- you can execute `pyapp/processqrcodeRedirect.bat` and find log in the `pyapp/generated` folder
- you can uncomment data.save_data(config.get_filename("qr_code_test2")) to save data in qr_code_test2.pickle.gz with the new computed qr code (for now qr_code_test.pickle.gz has the qr code of pv camera). You can then visualize qr_code_test2.pickle.gz (change `self.record` in `pyapp/config.py`).

- To visualize dataset execute `pyapp/visucalibrationRedirect.bat` and find log in the `pyapp/generated` folder
- When you are in the 3d window, you can:
	- mouse wheel to zoom in/out
	- left click to rotate
	- shift + left click to roll
	- ctrl + left click (or middle click) to translate
	- press `L` to deactivate the lighting (useful because the light are not set properly right now)
	- press `W` to have wireframe
- On the plot you can:
	- right click for some option like display the legend
	- double left click to center to default view (all plots visible)
	- right click selection enable to choose a window where you want to center/zoom

## Thirdparties
- [scikit-surgerycalibration](https://github.com/SciKit-Surgery/scikit-surgerycalibration)