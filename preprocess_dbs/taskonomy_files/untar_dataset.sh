#!/bin/sh

declare -a trainlist=("hanson" "merom" "klickitat" "onaga" "leonardo" "marstons" "newfields" "pinesdale" "lakeville" "cosmos" "benevolence" "pomaria" "tolstoy" "shelbyville" "allensville" "wainscott" "beechwood" "coffeen" "stockman" "hiteman" "woodbine" "lindenwood" "forkland" "mifflinburg" "ranchester")

declare -a validlist=("wiconisco" "corozal" "collierville" "markleeville" "darden")

folder="tiny-val"

for room in "${validlist[@]}"
do
    for domain in rgb depth_zbuffer edge_texture normal
    do
        echo ${folder}/${room}_${domain}.tar
        tar -xf ${folder}/${room}_${domain}.tar -C $folder
    done
done


