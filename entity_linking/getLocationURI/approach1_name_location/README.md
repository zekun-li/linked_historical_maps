### USGS entity matching

## 1. 01_getURIwithPhrases.ipynb
- input: map_phrase_USGS(GT).json
	- map_phrase_USGS(GT).json: ground truth phrases on USGS maps
	- output of extractPhrase_USGS(GT).ipynb in "makeRDF" folder
- get node/way uris having case-sensitve exact mathched phrases
	- intermediate_output: URIwithPhrase.pkl

## 2. 02_getLocationFromURI.ipynb
- input: URIwithPhrase.pkl from step 1
- get coordinates information
- intermediate_output:
	- name_loc_result.pkl: results have name and location
	- uri_failtogetLoc_list.txt: uris which fails to get location infromation

## 3. 03_entityMatching.ipynb
- input: name_loc_result.pkl.pkl, Historical Map Metadata - USGS.csv (coordinates information)
- filtering only uris within the map boundaries
- output
	- uri_exactmatching.pkl 
	- empty_uri_phrases.pkl