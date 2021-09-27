# Entity linking for historical maps

## Purpose:
For the text labels contained in the map images, we would like to link those text labels to external knowlege bases such as LinkedGeoData and GeoNames. In this way, the associated metadata information from the external KB can help the machines better understand the historical map. 

## Dependencies:
You need to install the following packages:
```
pip install SPARQLWrapper
pip install rdflib rdflib
```

## Code Structure:

**getLocationURI**:
This folder contains script for getting uri from GeoNames and LinkedGeodata. 

**makeRDF**:
This folder contains script for generating RDF triples given the schema. To run scripts inside this repo, Apache Jena Fuseki server is required to be deployed on your machine. This folder contains a tutorial for settting up the Apache Jena Fuseki for the linked historical database. 
