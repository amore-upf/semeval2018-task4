#/bin/bash

GET_SEMEVAL_DATA=true
GET_PEPROCESSED_SEMEVAL_DATA=true
GET_GOOGLE_NEWS_EMBEDDINGS=false

friends_subfolder=data/friends/

echo ">> Downloading Data into "$friends_subfolder
if [ \! -d $friends_subfolder ]; then 
    mkdir -p $friends_subfolder; 
fi

if [ $GET_SEMEVAL_DATA == true ]; then    
    echo ">> Downloading SemEval-Task-4 data (test data)."
    github_path="https://raw.githubusercontent.com/emorynlp/semeval-2018-task4/master/dat/"
    file_prefix="friends."
    for split in "train" "test"; do
        for setting in "episode" "scene"; do 
            fname=$file_prefix$split'.'$setting'_delim.conll'; 
            wget $github_path$fname -O $friends_subfolder/$fname;
            if [ $split == "test" ]; then
                fname=$fname'.nokey';
                wget $github_path$fname -O $friends_subfolder/$fname;
            fi
        done
    done
    wget $github_path'ref.out' -O $friends_subfolder/'ref.out';
    wget $github_path'friends_entity_map.txt' -O $friends_subfolder/'friends_entity_map.txt';
    
    echo ">> Downloading SemEval-Task-4 trial data."
    https://github.com/emorynlp/semeval-2018-task4/tree/master/dat

    OUT_F_FRIENDS=data/friends_train.trial.zip
    wget https://competitions.codalab.org/my/datasets/download/d8e0b7e1-1c4f-4171-93e9-74339e6c759e -O $OUT_F_FRIENDS
    #unzip $OUT_F_FRIENDS -d $friends_subfolder/.
    unzip -j $OUT_F_FRIENDS 'friends.trial.episode_delim.conll' -d $friends_subfolder/.
    unzip -j $OUT_F_FRIENDS 'friends.trial.scene_delim.conll' -d $friends_subfolder/.
    rm $OUT_F_FRIENDS
fi

if [ $GET_PEPROCESSED_SEMEVAL_DATA == true ]; then
    echo ">> Downloading data obtained by preprocessing SemEval-Task-4 data."
    echo "TODO"
fi


if [ $GET_GOOGLE_NEWS_EMBEDDINGS == true ]; then
    echo ">> Downloading GoogleNews skipgram embeddings"
    # Acknowledgements to https://gist.github.com/yanaiela/ for the script:
    # https://gist.github.com/yanaiela/cfef50380de8a5bfc8c272bb0c91d6e1.js
    OUTPUT=$( wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p' )
    CODE=${OUTPUT##*Code: }
    echo $CODE

    F='GoogleNews-vectors-negative300.bin.gz'
    OUT_F=data/$F

    wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm='$CODE'&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O $OUT_F
fi
