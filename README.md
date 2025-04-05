# Object detection algorithms in embedded systems

My master's thesis about object detection. In progress...

## Setup

### Cloning
This repository relies on submodules (OpenCV and GoogleTest) therefore you need to clone with --recursive flag

```bash
$ git clone https://github.com/aleksander-piprek/Object-detection-algorithms-in-embedded-systems.git --recursive
```

### Installing  
To install dependencies needed to run this project please check this shell script and then run

```bash
./setup.sh
```

## Dependencies
 - OpenCV
 - GoogleTest

## Build
### Release
```bash
mkdir build && cd build
cmake ..
make
```

### Test
```bash
mkdir build && cd build
cmake ..
make test
```

## Notes

- Do not use VS Code snap version! It uses its own packages which collides with the system's software (GLIBC, libpthread)