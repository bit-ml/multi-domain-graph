#/bin/sh

declare -a trainlist=("hanson" "merom" "klickitat" "onaga" "leonardo" "marstons" "newfields" "pinesdale" "lakeville" "cosmos" "benevolence" "pomaria" "tolstoy" "shelbyville" "allensville" "wainscott" "beechwood" "coffeen" "stockman" "hiteman" "woodbine" "lindenwood" "forkland" "mifflinburg" "ranchester")
folder="tiny-train"


for room in "${trainlist[@]}"
do
    #for domain in rgb depth_zbuffer edge_texture normal segment_semantic
    for domain in segment_semantic
    do
        echo ${folder}/${room}_${domain}.tar
        nohup tar -xf ${folder}/${room}_${domain}.tar --transform s/point_/${room}_point_/ -C $folder &
    done
done

