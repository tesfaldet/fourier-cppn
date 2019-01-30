# Improving the Realism of CPPN-based Image Parameterizations via Differentiable Texture Mapping
TensorFlow implementation of "Improving the Realism of CPPN-based Image Parameterizations via Differentiable Texture Mapping" (2019)

## Setup (CPU-version on Mac)
```
Install pyenv
$ brew update
$ brew install pyenv

Install Python 3.6.* (latest version supported by TensorFlow)
$ pyenv install 3.6.5

Set Python 3.6.5 as default
$ pyenv global 3.6.5

Install pipenv
$ brew install pipenv

Install all dependencies for this project (including dev):
$ cd texture-cppn
$ pipenv install --dev

Activate this project's virtualenv:
$ pipenv shell
```

### If using the [fish shell](https://github.com/fish-shell/fish-shell)
To get pyenv to execute when starting a fish shell, add this to your fish.config file:
`source (pyenv init - | psub)`

Also, There's this [plugin](https://github.com/kennethreitz/fish-pipenv) which autoloads virtualenvs.