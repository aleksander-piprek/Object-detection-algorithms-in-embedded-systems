# Object detection algorithms in embedded systems

## Setup

### Cloning
This repository relies on submodules (OpenCV and GoogleTest) therefore you need to clone with --recursive flag
~~~
$ git clone https://github.com/aleksander-piprek/Object-detection-algorithms-in-embedded-systems.git --recursive
~~~

### Installing  
To install dependencies needed to run this project please check this shell script and then run

~~~
$ ./setup.sh
~~~

## Notes

* Do not use VS Code snap version! It uses its own packages which collides with the system's software (GLIBC, libpthread)