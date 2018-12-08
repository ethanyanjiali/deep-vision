freeze:
	pip-compile --no-index --generate-hashes --rebuild --output-file requirements.txt requirements.in
