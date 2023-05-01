# read a file with english and slovenian sentences in tab-separated value format and extract the english sentences
with open("MaCoCu-sl-en.txt", "r", encoding="utf-8") as f_in:
    with open("slovene.txt", "w", encoding="utf-8") as f_out:
        for line in f_in:
            english, slovenian = line.strip().split("\t")
            f_out.write(slovenian + "\n")