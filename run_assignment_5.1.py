import subprocess
import json
import matplotlib.pyplot as plt

# range to test
beam_sizes = range(1, 26)  # 1 to 25

# lists for BLEU scores and brevity penalties
bleu_scores = []
brevity_penalties = []

# loop through each beam size
for k in beam_sizes:
    # run inference command
    subprocess.run(f"python translate_beam.py --data data/en-fr/prepared --dicts data/en-fr/prepared --checkpoint-path assignments/03/baseline/checkpoints/checkpoint_best.pt --output assignments/05/baseline/translations_k_{k}.txt --beam-size {k}", shell=True)

    # run postprocess command
    subprocess.run(f"bash scripts/postprocess.sh assignments/05/baseline/translations_k_{k}.txt assignments/05/baseline/translations_k_{k}.p.txt en", shell=True)

    # run SacreBLEU evaluation
    result = subprocess.run(f"cat assignments/05/baseline/translations_k_{k}.p.txt | sacrebleu data/en-fr/raw/test.en", shell=True, capture_output=True, text=True)

    # Parse the SacreBLEU output
    sacrebleu_output = json.loads(result.stdout)
    print(sacrebleu_output)
    bleu_scores.append(sacrebleu_output['score'])
    brevity_penalties.append(float(sacrebleu_output['verbose_score'].split('BP = ')[1].split(' ')[0]))

print(bleu_scores)
print(brevity_penalties)

fig, ax1 = plt.subplots()

# plot BLEU scores
ax1.set_xlabel('Beam Size')
ax1.set_ylabel('BLEU Score', color='tab:blue')
ax1.plot(beam_sizes, bleu_scores, color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Brevity Penalty', color='tab:red')  # We already handled the x-label with ax1
ax2.plot(beam_sizes, brevity_penalties, color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()  # To ensure the labels don't get cut off
plt.show()
