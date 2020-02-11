#! /bin/bash

exe_dir=$(dirname $(readlink -f $0))
#base_signature_file="${exe_dir}/atp_signature_25snapshot.csv"
base_signature_file="${exe_dir}/atp_feature_vector_jan2020.csv"
#mutant_tag_file="${exe_dir}/mutant_tags.csv"
mutant_tag_file="${exe_dir}/1a1v_atp_activity_jan2020.csv"
#mutant_tag_file="${exe_dir}/revised_rna_mutant_tag.csv"
tag_fld=2

for file in "${base_signature_file}" "${mutant_tag_file}";
do
   if [ ! -f $file ]
   then
     echo "Error: missing file [$file]!!" 1>&2;
     exit 1;
   fi
done

duplicates=$(cut -d, -f1 "${base_signature_file}" | sort | uniq -c | awk '$1 > 1' | wc -l)

if [ $duplicates -gt 0 ]
then
  echo "Error: input signature file contains repeatation !!" 1>&2;
  exit 1;
fi


cut -d, -f1 "${base_signature_file}" | while read mutant_tag; 
do 
    mutant=$(echo ${mutant_tag} | cut -d_ -f1)
    if [ $(grep $(echo $mutant | cut -c2-) "${mutant_tag_file}" | wc -l) -eq 1 ]; 
    then 
       tag=$(grep $(echo $mutant | cut -c2-) "${mutant_tag_file}" | cut -d, -f${tag_fld}); 
       grep -E "^${mutant_tag}" "${base_signature_file}" | while read line; 
       do 
         echo "${line},${tag}"; 
       done; 
    fi; 
done
