# lpepy - LED Pose Estimation in Python

`lpepy` is a small program to figure out how a string of addressable LEDs is
arranged in 3D, and calculate coordinates for them so that they can be used for
3D light effects.

The goal was to build something which does _not_ require precise placement of
the camera, and is robust to capture errors -- capturing more images from more
positions should fix any problems in the calculated coordinates.

Inspired by the Christmas tree by Stand-up Maths:
https://www.youtube.com/watch?v=TvlpIojusBE

## Installation

Installation with `pip` should work:

```sh
pip install git+https://github.com/rcloran/lpepy.git
```

## Usage

There are two programs included with lpepy, `lpe-calibrate` and `lpe-capture`.
`lpe-capture` is the main functionality, but `lpe-calibrate` needs to generate
and save some data first.

The short version:

```
$ open data/calibration-chart.pdf
$ lpe-calibrate --camera 1
$ lpe-capture --camera 1 --wled-address 192.168.1.10 coords.json
```

### Camera convenience

An integrated laptop or monitor webcam will work, but is not very convenient (at
least, it hasn't been for me). There's lots of software which allows you to use
a smartphone camera as if it was a webcam, and that general approach has worked
for me.

- [Continuity camera](https://support.apple.com/en-us/HT213244) is built into
  newer versions of macOS and iOS, and allows using an iPhone as a camera for
  your Mac. I've found it works pretty well for me, but annoyingly the indexes
  of cameras is not stable, so I need to switch between 0, 1, etc in between
  invocations.

Other untested solutions:

- [Camo](https://reincubate.com/camo/) looks like it should work for iPhone and
  Android on Windows and Mac

### lpe-calibrate

`lpe-calibrate` saves information about your camera, and is only needed once for
any camera you will use. It calculates parameters about your camera such as lens
distortion, and the calibration data from it is required for `lpe-capture` to
work.

```
$ lpe-calibrate --help
usage: lpe-calibrate [-h] [--camera CAMERA] [--columns COLUMNS] [--rows ROWS]
                     [--block-size BLOCK_SIZE] [--calibration-file FILE]

options:
  -h, --help            show this help message and exit
  --camera CAMERA       The index of the camera to use (according to OpenCV) (default: 0)
  --columns COLUMNS     Number of vertices in the grid in a rightwards direction (along the X
                        axis) (default: 9)
  --rows ROWS           Number of vertices in the grid in a downward direction (along the Y axis)
                        (default: 6)
  --block-size BLOCK_SIZE
                        The size of each block (distance between vertices along an axis), in mm
                        (default: 20)
  --calibration-file FILE
                        Name of file to which calibration data will be written (default:
                        ~/Library/Application Support/lpepy/calibration.yml)
```

To use `lpe-capture`, first print out or display the calibration chart in
`data/calibration-chart.pdf`, then simply run `lpe-capture`, and point your
camera at the chart. Use the `--camera` argument with integer camera index to
select a camera if the default (0) does not work.

The calibration program keeps capturing data at 1-second intervals, and
indicates that the board is detected through a series of multi-coloured lines
across the grid, and a plot of the 3D axes on the top-left corner.

Keep moving the camera around so that the chart is viewed from a number of
different positions, including the edges of the captured image. I've found that
when the plotted axes look consistently "correct" the calibration is good, and
you can exit by pressing escape.

Learn more about camera calibration in OpenCV:

- https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

The calibration chart included in `lpepy` was generated with [gen_pattern.py
included with OpenCV][gen_pattern] using the following command:

```
gen_pattern.py -w 270 -h 200 -u mm -c 10 -r 7 -T radon_checkerboard
```

`lpe-calibrate` options `--rows` and `--columns` can be used if a chart with
different dimensions is used.

[gen_pattern]:
  https://github.com/opencv/opencv/blob/4.x/doc/pattern_tools/gen_pattern.py

### lpe-capture

`lpe-capture` is the main program. It lights up LEDs on an LED strip, takes an
image from the webcam, and then analyzes that image to figure out where the LEDs
are.

```
$ lpe-capture --help
usage: lpe-capture [-h] [--camera CAMERA] [--calibration-file FILE] [--leds LEDS]
                   [--wled-address WLED_ADDRESS] [--wled-port WLED_PORT]
                   FILE

positional arguments:
  FILE                  JSON file to which resulting coordinates will be written

options:
  -h, --help            show this help message and exit
  --camera CAMERA       The index of the camera to use (according to OpenCV) (default: 0)
  --calibration-file FILE
                        Name of file containing camera calibration data (created with lpe-
                        calibrate) (default: ~/Library/Application
                        Support/lpepy/calibration.yml)
  --leds LEDS           Number of LEDs in the LED string (default: 100)
  --wled-address WLED_ADDRESS
                        Address of host running WLED which will display LEDs (default:
                        255.255.255.255)
  --wled-port WLED_PORT
                        Port number that WLED is listening on (default: 21324)
```

Currently, the only supported way to control an LED strip is with the [DRGB WLED
UDP protocol][wled-udp]. I have only tested using the [ESPHome WLED
Effect][esphome-wled], but the actual [WLED firmware][wled] should work well,
too. I would welcome contributions to control LED strips locally, or through any
other mechanisms.

When you run `lpe-capture` some basic instructions will appear in the bottom
left of the window. Once you have placed your camera somewhere stable with all
(or most) of your LEDs in view, press `c` to begin capturing. Once that capture
is complete (the "Press c ..." instruction appears again), move the camera to
another position, and capture again.

The initial camera position will determine the direction of the X (rightwards),
Y (downwards), and Z (away) axes. There's no (included) way to re-orient your
points at the moment.

Other than the initial capture, you do not need to be careful about where the
camera is placed each time, `lpepy` will attempt to determine the camera's
orientation with respect to the LEDs, and calculate 3D coordinates
automatically.

After every capture (other than the first!), the output file specified on the
command line will be written to with the the current estimate of the LED's world
co-ordinates. Press escape to exit the program.

Tips:

- `lpe-capture` uses a low brightness by default. If you're using ESPHome, the
  overall brightness of the light is taken into account too -- make sure that
  brightness is at the maximum.
- `lpe-capture` uses background extraction to try to avoid erroneous detection
  of lights, but I've found a dark room without any bright spots in the images
  is still most reliable.
- Lots of reflection (eg, a bunch of LEDs in a glass vase) might cause poor
  location detection. In this case, I've found just capturing from many
  different locations (7+ usually works for me) will get me close enough to the
  right answer.

[wled-udp]: https://github.com/Aircoookie/WLED/wiki/UDP-Realtime-Control
[esphome-wled]: https://esphome.io/components/light/index.html#wled-effect
[wled]: https://kno.wled.ge

## Wishlist

- More LED control mechanisms
- An included way to re-orient the calculated points
- A more complete UI (probably using Open3D for plotting the point clouds as
  they're captured?)
- OpenCV's camera indexing is not stable on macOS. A better camera selection
  would be welcome. Possibly only worth doing with/after the UI.
- Acceptable results without doing camera calibration
- A way to detect the optimal brightness of LEDs before capture
- Only calculate each camera position once
- A way to highlight LEDs with no or suspected-bad information
- More reliable LED-location detection
