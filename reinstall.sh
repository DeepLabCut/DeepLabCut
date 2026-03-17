pip uninstall deeplabcut
rm -rf dist/ build/ *.egg-info
python3 setup.py sdist bdist_wheel
pip install dist/deeplabcut-3.0.0rc14-py3-none-any.whl
