rm -r TEST*
rm -r multi_mouse*
rm -r 3D*
rm -r OUT

cd ..
pip uninstall deeplabcut
python3 setup.py sdist bdist_wheel
pip install dist/deeplabcut-2.1.10.2-py3-none-any.whl

cd examples

python3 testscript.py
python3 testscript_3d.py #does not work in container
#python3 testscript_mobilenets.py
python3 testscript_multianimal.py

#python3 testscript_openfielddata_netcomparison.py
