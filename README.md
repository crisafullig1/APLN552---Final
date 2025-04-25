# APLN552---Final

This code is a replication of the main code in https://github.com/aidanematzadeh/word_learning.git - All code, support documents, and data were taken from this source. The goal of this was to run new data on this code to make comparisons between the original and AI generated Child Directed Speech Data

-
-

There are two main files here - Replication and CDS Expansion. Both use the original author's main() script, data, and supporting documents almost identically, with a few tweaks for compatibility. They should all be able to be run as is, as long as the support documents and data are downloaded.



Replication:

    - To replicate with the original author's data only, download all files in the Replication folder including all three datasets (gold standard, dev set and test set)
    - Run the Copy_of_apln552_final_project_(new_data).ipynb script as is - it will call to all the supporting documents and takes the three datasets as input. It will output the following 
      documents:
              - acq_score_timestamp.pkl
              - aligns_lm_-1.0_a20.0_ep0.01
              - input_wn_fu_cs_scaled_categ.dev.words1000
              - lex_lm_-1.0_a20.0_ep0.01
              - time_props_1000.csv
