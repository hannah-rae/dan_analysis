import csv

datfile = '/Users/hannahrae/data/dan_iki_reduced.csv'
outfile = '/Users/hannahrae/data/dan_iki_params.csv'

quantities = []
with open(datfile, 'rb') as csvfile:
    reader = csv.reader(open(datfile, 'rU'), dialect=csv.excel_tab)
    for row in reader:
        [site, sol, h2o_single, cl_single, p_single, h2o_double, cl_double, p_double] = row[0].rstrip().split(',')
        if float(p_single) > float(p_double):
            quantities.append([site, sol, float(h2o_single), float(cl_single)])
        else:
            quantities.append([site, sol, float(h2o_double), float(cl_double)])

with open(outfile, 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for q in quantities:
      writer.writerow(q)