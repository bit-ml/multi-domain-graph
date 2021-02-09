#/bin/bash

declare -a trainlist=("stockman" "hiteman" "woodbine" "lindenwood" "forkland" "mifflinburg" "ranchester")


# declare -a domains=("rgb" "depth_zbuffer" "edge_texture" "normal" "segment_semantic")
declare -a domains=("segment_semantic")

for room in "${trainlist[@]}"
do
    for domain in "${domains[@]}"
    do
        tar -xf /data/multi-domain-graph-3/datasets/Taskonomy/tiny-train//${room}_${domain}.tar  --files-from /data/multi-domain-graph-3/datasets/Taskonomy/tiny-train//$1/${room}_${domain}.txt --transform s/point_/${room}_point_/ -C /data/multi-domain-graph-3/datasets/Taskonomy/tiny-train/ &
    done
    wait
    echo "Done " ${room}
done

echo "Done all 25/25"


fname=../tiny-train-0.15-part3
mkdir -p $fname

for domain in "${domains[@]}"
do    
    mv ${domain} $fname
done

