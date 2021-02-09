#/bin/sh

declare -a testlist=('ihlen' 'mcdade' 'muleshoe' 'noxapater' 'uvalda')
folder="tiny-test"



for room in "${testlist[@]}"
do
    #for domain in rgb depth_zbuffer edge_texture normal segment_semantic
    for domain in segment_semantic
    do
        echo ${folder}/${room}_${domain}.tar
        nohup tar -xf ${folder}/${room}_${domain}.tar --transform s/point_/${room}_point_/ -C $folder &
    done
done

