#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
set +x
if [ $# == 0 ]; then
for pilot in "Pilot1" "Pilot2" "Pilot3" "examples"
do
    for dir in $pilot/*/
    do
        echo $dir
        cd $dir
# more test.sh
        ./test.sh > test.log
        rm test.log
        cd ../../
    done
done
elif [ $# == 1 ]; then
    for dir in $1/*/
    do
        echo $dir
        cd $dir
#        more test.sh
        ./test.sh > test.log
        rm test.log
        cd ../../
    done
elif [ $# == 2 ]; then
    for dir in $1/$2/
    do
        echo $dir
        cd $dir
#        more test.sh
        ./test.sh > test.log
        rm test.log
        cd ../../
    done
else
    echo "Too many arguments"
    echo "With no arguments the script will loop over Pilot1, Pilot2, Pilot3 and examples"
    echo "Or you can enter a directory, or a directory and benchmark to test a smaller set"
fi
