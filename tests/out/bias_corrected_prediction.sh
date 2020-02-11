#! /bin/bash

result=/home/sumanta/Project/structural_dynamics/nma_revision/augmentation/5MFX_atp_pred.txt
wts=/home/sumanta/Project/structural_dynamics/nma_revision/augmentation/5MFX/model_sampling_weights.csv
tag=5MFX

cat "$result" | awk 'NR%3==1{ 
                    printf("%s ",$1) 
              } NR%3 == 2 {
                   printf("%s %s %s ", $3, $4, $5)
              } NR%3 == 0 {
                   print $NF
              }'  | 
              sed -e "s#.*${tag}/\(.*\)/nma/\(.*\) #\1 \2 #" | 
              sed -e 's/[(),]//g' | 
              sort -k1 | 
              awk -vwts=$wts 'BEGIN{
                  while(getline line < wts){
                      split(line,arr,",");
                      if(length(arr) == 3){
                         weight[arr[1],arr[2]] = arr[3];
                      }
                  }
              }NF==6{ 
                     id=$1; 
                     model=$2;
                     if( weight[id, model] > 0)
                     {
                       n[id]++;
                       neg[id] += $3; 
                       neu[id] += $4; 
                       pos[id] += $5; 
                     }
                     neg_data += $3;
                     neu_data += $4;
                     pos_data += $5;
                     counter ++;
              } END{ 
                   neg_data = neg_data / counter;
                   neu_data = neu_data / counter;
                   pos_data = pos_data / counter;
                   for(id in n){
                       neg_avg = neg[id]/n[id];
                       neu_avg = neu[id]/n[id];
                       pos_avg = pos[id]/n[id];
                      
                       printf("%s %5.3f %5.3f %5.3f\n",id, neg_avg, neu_avg, pos_avg); 
                   } 
             }'
