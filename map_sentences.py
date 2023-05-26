from tqdm import tqdm

def create_sentence_dictionary(file1, file2):
    sentence_dict = {}
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        sentences1 = f1.readlines()
        sentences2 = f2.readlines()

        for i in range(len(sentences1)):
            sentence_dict[sentences1[i].strip()] = sentences2[i].strip()

    return sentence_dict


def process_files(file1, file2, file3, file4, result_file, dict):
    # sentence_dict = create_sentence_dictionary(file1, file2)

    with open(file3, 'r', encoding='utf-8') as f3, open(file4, 'r', encoding='utf-8') as f4, open(result_file, 'a', encoding='utf-8') as result:
        sentences3 = f3.readlines()
        sentences4 = f4.readlines()

        for sentence in tqdm(sentences3):
            current_sentence = sentence.strip()
            if current_sentence in dict:
                value = dict[current_sentence]
                translation = sentences4[sentences3.index(sentence)].strip()
                result.write(f"{value}\t{translation}\n")



# Specify the paths to your input files and the result file
file1 = ''
file2 = ''
result_file = 'subtitles.txt'
all_files_translated_english = [""]
all_files_translated_slovene = [""]

path = ''
sentence_dict = create_sentence_dictionary(file1, file2)

for i in range(len(all_files_translated_english)):
    all_files_translated_english[i] = f"{path}/{all_files_translated_english[i]}"
    all_files_translated_slovene[i] = f"{path}/{all_files_translated_slovene[i]}"

    
sentence_dict = create_sentence_dictionary(file1, file2)
#use enumerate
for i, file in enumerate(all_files_translated_english):
    process_files(file1, file2, all_files_translated_english[i],all_files_translated_slovene[i], result_file, sentence_dict)