init:
	pip install -r requirements.txt

unittest:
	nosetests tests

regtest:
	nosetests regressions/regressions.py

test:
	nosetests --with-xunit -i tests -i regression

.PHONY: init unittest regtest test
