python_venv = venv\Scripts\python.exe
pip_venv = venv\Scripts\pip.exe

install:
	python -m venv venv &&\
	$(pip_venv) install -r requirements.txt

run:
	@$(python_venv) main.py