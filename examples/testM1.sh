rm -r TEST*
rm -r multi_mouse*
rm -r 3D*
rm -r OUT

cd ..
pip uninstall deeplabcut
pythonw setup.py sdist bdist_wheel
pip install dist/deeplabcut-2.2.0.2-py3-none-any.whl

# download: https://drive.google.com/file/d/17pSwfoNuyf3YR8vCaVggHeI-pMQ3xL7l/view?usp=sharing
# assuming it's in Downloads...
pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl --no-dependencies --force-reinstall

cd examples

pythonw testscript.py
pythonw testscript_3d.py #does not work in container
pythonw testscript_multianimal.py
