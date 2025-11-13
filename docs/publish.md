

1) Make sure you git is up to date
```bash
git status
```

1) Run tests locally.
```bash
PYTHONPATH="./src/" pytest --cov ./src/pylelemmatize/ ./test/pytest/
./test/test_shell_scripts.sh
```
2) Build documentation locally
```bash
cd docs && make html
```

2) Set the version number and release number in src/pylelemmatize/version.py and docs/conf.py

3) Push to github
```bash
git push
```

4) Check docs and code is passing. All the shields are fine in github

5) Upload to pypi
```bash
rm ./dist/*
python -m build
twine upload --verbose  --repository pypi dist/pylelemmatize-*
```