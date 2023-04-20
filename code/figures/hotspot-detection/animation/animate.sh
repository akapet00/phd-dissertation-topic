#! /bin/bash

convert -delay 20 -loop 0 *.png -resize 20% -verbose anim.gif
