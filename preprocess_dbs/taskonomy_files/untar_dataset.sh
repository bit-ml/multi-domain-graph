#/bin/sh

declare -a trainlist=("hanson" "merom" "klickitat" "onaga" "leonardo" "marstons" "newfields" "pinesdale" "lakeville" "cosmos" "benevolence" "pomaria" "tolstoy" "shelbyville" "allensville" "wainscott" "beechwood" "coffeen" "stockman" "hiteman" "woodbine" "lindenwood" "forkland" "mifflinburg" "ranchester")
folder="tiny-train"

#declare -a validlist=("wiconisco" "corozal" "collierville" "markleeville" "darden")
#folder="tiny-val"

#declare -a testlist=('ihlen' 'mcdade' 'muleshoe' 'noxapater' 'uvalda')
#folder="tiny-test"



for room in "${validlist[@]}"
do
    for domain in rgb depth_zbuffer edge_texture normal
    do
        echo ${folder}/${room}_${domain}.tar
        nohup tar -xf ${folder}/${room}_${domain}.tar --transform s/point_/${room}_point_/ -C $folder &
    done
done

