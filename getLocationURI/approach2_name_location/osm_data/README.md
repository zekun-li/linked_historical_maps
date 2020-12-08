# These tsv files are obtained by:

## 1. download "planet-latest_geonames.tsv" from https://osmnames.org/download/

### 2. Use Data extract by bounding box (west, south, east, north)
- go to the folder having "planet-latest_geonames.tsv"
- enter the command for creating tsv file
- example command for creating tsv file

```
zcat planet-latest_geonames.tsv.gz | awk -F '\t' -v OFS='\t' 'NR == 1 || ($8 >= 32.75 && $8 <= 33.0 && $7 >= -115.75 && $7 <= -115.5)' > USGS-15-CA-brawley-e1957-s1957-p1961.jpg.tsv
```
- There are commands in getCommand_for_BBbox.ipynb


