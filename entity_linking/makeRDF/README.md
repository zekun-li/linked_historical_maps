## Folder
1. input: all csv files extracted from histrical maps
2. input_usgs: USGS csv files
3. input_2ndLayer: manually added annotation csv files

## Scripts
1. extractPhrase_allGT.ipynb
    - extracting phrases from input folder (all csv files)
     - output: map_phrase_allGT.json
2. extractPhrase_USGS(GT).ipynb
    - extracting phrases from input_usgs folder (USGS csv files)
    - output: map_phrase_USGS(GT).json
3. extractPhrase_adding2ndlayer.ipynb
    - extracting phrases from input and input_2ndLayer
    - output: map_phrase_gt_2ndlayer.json
4. makeRDFcode.ipynb
    - making RDF
    - input: map_phrase_allGT.json
    - output: mapRDF.ttl

