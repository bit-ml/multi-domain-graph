#/bin/sh

declare -a validlist=("wiconisco" "corozal" "collierville" "markleeville" "darden")
folder="tiny-val"


for room in "${validlist[@]}"
do
    #for domain in rgb depth_zbuffer edge_texture normal segment_semantic
    for domain in segment_semantic
    do
        echo ${folder}/${room}_${domain}.tar
        nohup tar -xf ${folder}/${room}_${domain}.tar --transform s/point_/${room}_point_/ -C $folder &
    done
done
