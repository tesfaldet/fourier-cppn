# Fourier-CPPNs for Image Synthesis
TensorFlow implementation of "Fourier-CPPNs for Image Synthesis" (2019). Presented at the Second Workshop on Computer Vision for Fashion, Art, and Design at ICCV '19.

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
$ cd fourier-cppn
$ pipenv install --dev

Activate this project's virtualenv:
$ pipenv shell
```

## Usage (Requires [Shuriken](https://github.com/ElementAI/shuriken-client))
```
$ saga submit --file docker/Dockerfile --verbose --config experiments/shk-tests.json
```

### If using the [fish shell](https://github.com/fish-shell/fish-shell)
To get pyenv to execute when starting a fish shell, add this to your fish.config file:
`source (pyenv init - | psub)`

Also, There's this [plugin](https://github.com/kennethreitz/fish-pipenv) which autoloads virtualenvs.