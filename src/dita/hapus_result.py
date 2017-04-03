import csv
with open('result1.csv', 'rb') as inp, open('predicted_result1.csv', 'wb') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[0] != "results":
            writer.writerow(row)