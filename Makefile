# ================================== Variable ================================== 
VENV = env
PYTHON = @$(VENV)/bin/python
PIP = @$(VENV)/bin/pip


# ==================================   ================================== 
default: venv

venv: 
	@echo "Creating enviroment..."
	@python3 -m venv $(VENV)
	$(PYTHON) --version

install:
	@echo "Installing package..."
	@$(PIP) install -r requerements.txt

prepare:
	@echo "Preapring..."

train: prepare
	@echo "Training..."

run: 
	@echo "Running..."

clean: clean-pyc clean-venv
	@echo "Clean done."

clean-pyc:
	@echo "Cleaning *.pyc ..."
	@find . -type f -name '*.pyc' -delete

clean-venv:
	@echo "Cleaning enviroment..."
	@rm -rf $(VENV)

help:
	@echo "OPTIONS"
	@echo "\tvenv"
	@echo "\t\tCreate enviroment for project\n"
	@echo "\tinstall"
	@echo "\t\tInstall packages for project\n"
	@echo "\tprepare"
	@echo "\t\tPrepare dataset to run train model\n"
	@echo "\ttrain"
	@echo "\t\tTrain model\n"
	@echo "\trun"
	@echo "\t\tRun application\n"
	@echo "\tclean"
	@echo "\t\tRemove all cache and enviroment\n"