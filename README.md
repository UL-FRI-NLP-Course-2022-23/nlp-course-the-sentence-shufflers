# Natural language processing course 2022/23: `The Sentence Shufflers`

Team members:
 * `Aljaž Grdadolnik`, `63160120`, `ag5319@student.uni-lj.si`
 * `Anže Mihevc`, `63170199`, `am8130@student.uni-lj.si`
 * `Luka Galjot`, `63160111`, `lg0775@student.uni-lj.si`
 
Group public acronym/name: `TSS`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## How to evaluate?
There are two scripts available `test_model.py` and `test_model_csv.py`.

First one is interactive and allows paraphrasing of any sentence given via console input.

The second one works on a csv file. Each line should be a sentence, the script then appends the file with a new column
of paraphrased sentences. Fist define input and output files in the script and the run it.

To evaluate the model(with metrics BLEU, ROUGE, BERTScore, WER, METEROR, Google BLEU, ParaScore) use calculate_scores.py script where you need to provide a path to the file which contains the paraphrase pairs(reference\tgenerated). The script crates a csv file with all the scores in a file with the same name as the input and added "-metrics" at the end.

## How to train the model?
Use the `train_model.py` script. It allows retraining of the model, just define the checkpoint name in load_model function.
Pick an appropriately structured dataset and put it into the `datasets` folder. Write down the name of that dataset 
on **line 20** in train script.

## Other scripts
There are other scripts that will help you create usable dataset. Each script contains a comment which tells you what it does.