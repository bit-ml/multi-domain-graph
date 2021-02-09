#/bin/sh
aria2c --continue=true -x 16 -j 10 -i $1.txt
