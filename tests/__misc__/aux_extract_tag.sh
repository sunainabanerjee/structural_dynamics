#! /bin/bash

cat ../out/all_sig_rna.csv |  while read line; 
do 
  tag=$(echo $line | cut -d, -f1); 
  if [ $(grep $(echo $tag |cut -d_ -f1) /tmp/rna.csv | wc -l) -eq 1 ]; 
  then  
      class=$(grep $(echo $tag | cut -d_ -f1) /tmp/rna.csv | cut -d, -f2); 
      echo $line","$class; 
  fi  
done > out/all_sig_rna_with_class.csv

