import pandas as pd
df = pd.read_csv('data.csv')
print(df[df['TRADING CODE'] == 'GREENDELT'])

#writer = csv.writer(open('Book1.csv'))
#for pd in csv.reader('data.csv'):
  #  if (df[df['TRADING CODE'] == 'GREENDELT']):
 #       writer.writerow(df)
#writer.close()

#import pandas as pd

# Create sample CSV-file (100x100)
#df = pd.DataFrame(np.arange(10000).reshape(100,100))
#df.to_csv('test.csv', index=False)
import csv

with open('Book1.csv', 'rb') as f:
  data = list(csv.reader(f))

import collections
counter = collections.defaultdict(int)
for row in data:
    counter[row[0]] += 1


writer = csv.writer(open("Book1.csv", 'w'))
for row in data:
    if counter[row[0]] >= 4:
        writer.writerow(row)
