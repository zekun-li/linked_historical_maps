# Entity linking for historical maps

## Purpose:
For the text labels contained in the map images, we would like to link those text labels to external knowlege bases such as LinkedGeoData and GeoNames. In this way, the associated metadata information from the external KB can help the machines better understand the historical map. 

## Code Structure:

**getLocationURI**:
This folder contains script for getting uri from GeoNames and LinkedGeodata. 

**makeRDF**:
This folder contains script for generating RDF triples given the schema. It also has a tutorial for deploying the Apache Jena Fuseki server for the linked historical database. 
