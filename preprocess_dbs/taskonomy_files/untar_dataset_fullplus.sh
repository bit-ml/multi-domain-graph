#/bin/sh


declare -a testlist=('espanola' 'matoaca' 'vacherie' 'ihlen' 'caruthers' 'darrtown' 'manassas' 'bonfield' 'cousins' 'hammon' 'muleshoe' 'leilani' 'kingfisher' 'paige' 'benicia' 'imbery' 'landing' 'uvalda' 'wappingers' 'mcnary' 'donaldson' 'purple' 'plessis' 'ewansville' 'poipu' 'kihei' 'lovilia' 'goodview' 'gluek' 'akiak' 'calavo' 'bellemeade' 'morris' 'macland' 'liddieville' 'brentsville' 'rockport' 'experiment' 'cauthron' 'german' 'connellsville' 'rosenberg' 'rabbit' 'bertram' 'dalcour' 'norvelt' 'broadwell' 'dauberville' 'destin' 'peconic' 'maben' 'shingler' 'wilkesboro' 'edson' 'mcclure' 'wando' 'barahona' 'ohoopee' 'noxapater' 'wakeman' 'blenheim' 'mcdade' 'helton' 'grigston' 'gastonia' 'natural' 'aldine' 'badger' 'cochranton' 'sontag' 'belpre' 'tysons' 'gluck' 'losantville' 'lathrup' 'samuels' 'ladue' 'coeburn')
folder="fullplus-test"





for room in "${testlist[@]}"
do
    # for domain in rgb depth_zbuffer edge_texture normal
    for domain in normal
    do
        echo ${folder}/${room}_${domain}.tar
        nohup tar -xf ${folder}/${room}_${domain}.tar --transform s/point_/${room}_point_/ -C $folder &
    done
    # sleep 60
done

