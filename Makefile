venv:
	python3 -m venv env

freeze:
	pip-compile --no-index --generate-hashes --rebuild --output-file requirements.txt requirements.in

kernel_add:
	ipython kernel install --user --name=cv-classification

kernel_remove:
	jupyter kernelspec uninstall cv-classification -f