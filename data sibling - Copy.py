import csv

f = open('data.csv')
csv_f = csv.reader(f)

TRADING_CODE = []

for row in csv_f:
  TRADING_CODE.append(row[2])

print TRADING_CODE
