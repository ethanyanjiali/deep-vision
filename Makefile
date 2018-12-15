freeze:
	pip-compile --no-index --generate-hashes --rebuild --output-file requirements.txt requirements.in

kernel_add:
	ipython kernel install --user --name=image-classification

kernel_remove:
	jupyter kernelspec uninstall image-classification -f