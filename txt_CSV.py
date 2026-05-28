import csv

def txt_to_csv(input_txt, output_csv):
    with open(input_txt, 'r', encoding='utf-8') as infile, open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Front', 'Back'])
        for line in infile:
            if '\t' in line:
                front, back = line.strip().split('\t', 1)
                writer.writerow([front, back])

if __name__ == "__main__":
    txt_to_csv("flashcards.txt", "flashcards.csv")
