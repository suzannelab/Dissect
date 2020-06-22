.PHONY: flake8 test coverage

flake8:
	flake8 --max-line-length=100 --count --statistics --exit-zero Dissects/

test:
	cd tests && pytest . && cd ..

coverage:
	cd tests &&  pytest --cov=Dissects --cov-config=../.coveragerc . && mv .coverage .. && cd ..

nbtest:
	cd doc/notebooks && pytest --nbval-lax && cd ../..

pdbtest:
	cd tests && pytest --pdb . && cd ..