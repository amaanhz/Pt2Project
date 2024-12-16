import csv


def reddit_tsv(name, numlines):
    with open("raw/soc-redditHyperlinks-body.tsv", "r") as file:
        reader = csv.reader(file, delimiter='\t')
        reader.__next__() # skip column titles
        for i, row in enumerate(reader):
            if i == numlines:
                break
            vector = row[-1].split(',')
            print(f"{row[0]} -> {row[1]}, Pos: {vector[18]}, Neg: {vector[19]}, Comp: {vector[20]}")

if __name__ == "__main__":
    reddit_tsv("reddit-hyperlinks", 100)