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
- Edit the file `pyapp/config.py` to change `self.path` (where is your data folder). To change dataset edit the end of the file.

- Check `pyapp/sphere_tracking.py` for an example to get the center of the sphere position in images using the optical tracking aligned via the qr code to the hololens world coordinate.
	- You can execute `pyapp/sphere_trackingRedirect.bat` and find log in the `pyapp/generated` folder.
	- Check the `docs` folder to get more information on the different coordinate systems.

- Check `pyapp/processqrcode.py` for an example to find qr codes (pv camera or vl cameras) and compute distance error between optical and hololens.
	- You can execute `pyapp/processqrcodeRedirect.bat` and find log in the `pyapp/generated` folder.
	- You can uncomment data.save_data(config.get_filename("optical_sphere")) to save data in optical_sphere.pickle.gz with the new computed qr code (for now optical_sphere.pickle.gz has the qr code of vl cameras).

- To visualize dataset execute `pyapp/visucalibrationRedirect.bat` and find log in the `pyapp/generated` folder
	- When you are in the 3d window, you can:
		- Mouse wheel to zoom in/out
		- Left click to rotate
		- Shift + left click to roll
		- Ctrl + left click (or middle click) to translate
		- Press `L` to deactivate the lighting (useful because the light are not set properly right now)
		- Press `W` to have wireframe
	- On the plot you can:
		- Right click for some option like display the legend
		- Double left click to center to default view (all plots visible)
		- Right click selection enable to choose a window where you want to center/zoom

## Thirdparties
- [scikit-surgerycalibration](https://github.com/SciKit-Surgery/scikit-surgerycalibration)