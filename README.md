# Frugal Prompting for Dialog Models 
Official repository for experiments in the 'Frugal Prompting for Dialog Models' paper. 

# Steps

```
cd utils/deb/
bash setup.sh
cd ../

# Downloads the BLEURT-base checkpoint. 
cd bleurt/bleurt
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
cd ../../../

# Concat ./context_data/topical_chat/generated_summary/test_summary_and_knowledge_dialogdata_part1.txt and part2.txt
# to ./context_data/topical_chat/generated_summary/test_summary_and_knowledge_dialogdata.txt:
cat ./context_data/topical_chat/generated_summary/test_summary_and_knowledge_dialogdata_part1.txt ./context_data/topical_chat/generated_summary/test_summary_and_knowledge_dialogdata_part2.txt > ./context_data/topical_chat/generated_summary/test_summary_and_knowledge_dialogdata.txt

# nltk
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('wordnet')"
```

# Experiments

## Inference Step
To run the experiments described in the paper, first execute all_exp_batch.sh. This will automatically generate and start the experiments.

## Evaluation Step

For evaluating the generation outputs, the same commands from **Inference Step** has to be run with the extra `-e` flag.


