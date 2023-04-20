#! /bin/bash

convert -delay 10 -loop 0 *.png -resize 50% -verbose anim.gif
