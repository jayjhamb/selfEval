# selfEval
evaluating the davinci-002 model on how accurate it is on P(T) as mentioned by the P(True) calibration experiment from Language Models (Mostly) Know What They Know (Kadavath et al., 2022)

This repo consists of the following files
- few_shot_examples.py
- main.py
- summary.csv
- results.jsonl
- README.md

Install dependencies: pip install numpy openai

Requirements: openAI key which is obtainable through the openai platform
- input openai key into where it says OPENAI_API_KEY="your_api_key_here"

run with
- python main.py

following inputs are needed:
- .json or .jsonl file for dataset
- sample size (+=1)
- self-evaluation repeats, default is 5
- k sampling, default is 5
- sampling temperature, default is 0.7, use 0.0 for P(T)
- specify model name, default is davinci-002 NOTE: this program is specifically for text models on openai, no other model might work without errors.
- output for .jsonl path, results.jsonl is default if skipped
- output for .csv path, summary.csv is default if skipped
- how many bins to seperate ece

results:

the program outputs accuracy P(T) and ECE
for example:
Processed: 750 examples.
Overall Accuracy: 0.6493
ECE: 0.2836


file outputs:
- results.jsonl
- summary.csv

