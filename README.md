# EventOrient:
## A System for Community-Scale Tempo-Contextual Network Analysis of Social Data Streams

#### Developed in <a href="http://sehir.edu.tr/">Istanbul Sehir University: Media Lab
  <p>
<p>
<img src="https://www.sehir.edu.tr/tr/Documents/kurumsal-kimlik/PNG_Formatinda_SEHIR_logo_1.png" width=200>
  </br>
<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/medialab.jpg?raw=true" width=200>
  </a>
</p>
<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/ss2.png?raw=true" width=500>
</p>

- <a href="https://github.com/AmmarRashed/EventOrient/blob/master/docs/IEEE_formatted_paper.pdf">Paper</a>
- <a href="https://github.com/AmmarRashed/EventOrient/blob/master/docs/final_report.pdf">Report</a>
- <a href="https://github.com/AmmarRashed/EventOrient/blob/master/docs/final_presentation.pdf">Slides</a>


## Abstract

We study the evolution of the social network of Istanbul Şehir University overtime, capturing direct, contextual and latent changes in the network structure. The university's story is embodied in the networks we construct, analyze and temporally monitor. Our system,*EventOrient*, stands on three components; Web Crawling, Networked Data Analysis and Data storytelling, making it a comprehensive system for community-scale tempo-contextual network analysis. Our goal is to render the social development of the university's community in a lucid and insightful manner.

## Data

- The datasets are compiled and published on <a href="http://datascience.sehir.edu.tr/main/datasets/"> Datascience Sehir</a>
- A notebook explaining the data used can be found in: <a href="https://github.com/AmmarRashed/EventOrient/blob/master/datasets/datasets.ipynb">datasets</a>


## System overview

<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/workflow.jpg?raw=true">
*EventOrient* provides a thorough pipeline for social network analysis. This pipeline consists of three main components; *Web Crawling*, *Networked Data Analysis* and *Data Storytelling*.

## Web Crawling

<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/crawler.jpg?raw=true" width=500>
- The crawler repository can be found in: <a href="https://github.com/ihsansecer/socialcrawler">socialcrawler</a>
- Twitter accounts are validated against Gmail contacts obtained from sehir.edu.tr domain
<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/accountValidating.jpg?raw=true" width=400>

## Network Analysis

- <a href="https://github.com/AmmarRashed/EventOrient/blob/master/notebooks/tracking_link_formation.ipynb">Tracking link-formation.</a> <p>Constructing a network from twitter connections. This notebook also has the script for filtering twitter accounts to obtain only the ones pertaining to <a href="http://sehir.edu.tr/">Sehir</a> community.</p>
- <a href="https://github.com/AmmarRashed/EventOrient/blob/master/notebooks/calculating_communities.ipynb">Calculating Communities</a><pr>Each node is labeled by the community detected by <a href="https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm"> Girvan-Newman algorithm</a>.
  - <a href="https://github.com/AmmarRashed/EventOrient/blob/master/notebooks/calculating_closures.ipynb"> Affiliation Network Analysis</a>
  <p> Institutional accounts are labled *foci* with which we build an affiliation network. Closures are detected and categorized accross different states of the network in different timestamps
    </br>
    <img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/focal.png?raw=true" width=300>
  <img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/member.png?raw=true" width=300>

</p>
  
## Application

- The source code for the Django application of the project can be found in <a href="https://github.com/AmmarRashed/EventOrient/tree/master/REST"> REST</a>

### Backend-Frontend communication

<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/Django.jpg?raw=true">

### Screenshots

<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/ss.png?raw=true">
<img src="https://github.com/AmmarRashed/EventOrient/blob/master/misc/pics/ss1.png?raw=true">
