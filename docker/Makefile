prepare_build:
	python3 -m pip install --upgrade twine build

build: prepare_build
	python3 -m build

upload_test: prepare_build
	python3 -m twine upload --repository testpypi dist/*

upload: prepare_build
	python3 -m twine upload dist/*
