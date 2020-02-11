#! /bin/bash

awk -F, -vseed=$RANDOM 'BEGIN{
          srand(seed);
      }
      {
       mutant=substr($1,1,5); 
       tag=$NF; 
       mutant_counter[mutant]++; 
       mutant_class[mutant] = tag; 
       class_counter[tag]++; 
       mutant_tag[mutant, mutant_counter[mutant]] = $1;
       for(i=2; i<NF;++i)
         mutant_data[mutant,mutant_counter[mutant], i]=$i; 
       n_features = NF - 1; 
     }END{
       min_count=-1;
       min_tag=-1;
       for(c in class_counter){
         if((min_count < 0) || (min_count > class_counter[c])){
           min_count = class_counter[c];
           min_tag = c;
         }
       }

       for(m in mutant_class)
         class_mutant[mutant_class[m]]++;

       for(c in class_counter){
          target_count[c] = int(min_count/class_mutant[c]);
       }
 
       for(m in mutant_counter){
         acceptance = target_count[mutant_class[m]]/mutant_counter[m];
         for(i=0; i < target_count[mutant_class[m]]; ++i){
           c = 1;
           while((accepted[c] != 0) || (rand() > acceptance))
           {
              c = (c + 1) % (mutant_counter[m] + 1);
              if(c == 0) c++;
           }
           accepted[c] = 1;
         }
         for(a in accepted){
           if(accepted[a] == 1)
           {
             printf("%s,", mutant_tag[m,accepted[a]]);
             for(i=2; i <= n_features; ++i)
                printf("%f,",mutant_data[m,a,i]);
             printf("%d\n",mutant_class[m]);
           }
         }
         delete accepted;
       }
  
     }' atp_site_signature_with_tag.csv
