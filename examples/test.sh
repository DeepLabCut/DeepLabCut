cd ..
pip uninstall deeplabcut
python3 setup.py sdist bdist_wheel
pip install dist/deeplabcut-2.2b4-py3-none-any.whl

cd examples
python3 testscript.py
