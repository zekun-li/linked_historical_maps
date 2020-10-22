1. getLocationURI1(sparql_query_USGS).ipynb
- input: map_phrase_USGS(GT).json
	- USGS ground truth phrase (lowercase) 
	- get by extractPhrase_USGS(GT).ipynb in make RDF folder
- get sparql query results based on node type 1 and 2
	- output: sparql_usgs_result1.pkl, sparql_usgs_result2.pkl

2. getLocationURI2(USGS)
- input: sparql_usgs_result1.pkl, sparql_usgs_result2.pkl, Historical Map Metadata - USGS.csv (coordinates information)
- filtering sqarql results based on coordinates
- output
	- uri_exactmatching.pkl: exact matching results (few)
	- empty_uri_phrases.pkl: phrases per map which doesn't match with location URI

3. Further consideration
- case-insenstive partial matching
- consider way, relation