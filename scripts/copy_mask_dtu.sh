# base="output/dtu/"
base=$1
scan_id=$2
mask_path="data/DTU/submission_data/idrmasks"


if [ -d $base ]; then
    # rm -r $base/$scan_id/mask
    mkdir $base/mask
    id=0
    if [ -d ${mask_path}/$scan_id/mask ]; then
        for file in ${mask_path}/scan8/*
        do  
            # echo $file
            file_name=$(printf "%05d" $id).png;
            cp ${file//scan8/$scan_id'/mask'} $base/mask/$file_name
            ((id = id + 1))
        done

        else

        for file in ${mask_path}/$scan_id/*
        do
            # echo $file
            file_name=$(printf "%05d" $id).png;
            cp $file $base/mask/$file_name
            ((id = id + 1))
        done
    fi
fi
