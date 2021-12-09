clean:
	rm -rf dist/
	rm -f PYPI_README.md

prepare_build:
	python3 -m pip install --upgrade twine build
	cp ../docs/docker.md PYPI_README.md

build: clean prepare_build  
	python3 -m build

upload_test: prepare_build
	python3 -m twine upload --repository testpypi dist/*

upload: prepare_build
	python3 -m twine upload dist/*
