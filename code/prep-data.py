#
# This script will handle pre-parsing the data into a form X-MAP likes
# The current "format" compatible with X-Map is as follows
# <userid>\t<itemid>\t<rating>\t<timestamp>


# For the json.gz files
import gzip
import json
import csv

import argparse
parser = argparse.ArgumentParser()

## CLI Args for this script
parser.add_argument("-i", "--infile", help="Input file to be parsed, should be json.gz")
parser.add_argument("-o", "--outfile", help="Output file to be parsed, should be tsv")
args = parser.parse_args()


if (not args.infile or not args.outfile):
    print("I need an input and output file!")

print("Infile: {} Outfile:{}".format(args.infile, args.outfile))

### At this point everything for the program is set up
def parseJSON(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def transform(obj):
  asin = obj['asin']
  userid = obj['reviewerID']
  review = obj['overall']
  time = obj['unixReviewTime']

  return([userid,asin,review,time])

with open(args.outfile, 'wt',) as outfile:
  print("Begining transformation. Each . represents 10,000 records")
  tsv_writer = csv.writer(outfile, delimiter='\t')
  for obj in parseJSON(args.infile):
    tsv_writer.writerow(transform(obj))
