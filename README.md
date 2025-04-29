# COFI

## Description
This is the GitHub repository for the creation of the China Overseas Finance Inventory Database (COFI) dataset.

The projects first cleans and extract information from already existing data sources about power plants and investments and then merges this information together by relying on fuzzy matching techniques.

## Technology Used
* Relational Database: for data storage and retrieval
* Fuzzy matching algorithms for text and numeric data
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN) for clustering commissioning years

## Libraries
The main libraries used are the following:
* pandas=2.2.1=pypi_0
* thefuzz=0.22.1=pypi_0
* scikit-learn=1.4.1.post1=pypi_0

For a full list of the libraries requirements, see here. # TODO: put link to requirements list (it's in the "11. Documentation" folder).

## Methodology

### Source Data
The data sources used are the following:

| Database Name | Abbreviation | Focus |
| - | - | - | 
| UDI World Electric Power Plants Data Base | WEPP | Power Plants |
| Global Power Plant Database | GPPD | Power Plants |
| China's Global Power Database | BU_CGP | Equity, Debt | 
| China's Global Energy Finance | BU_CGEF | Debt | 
| Chinese Loans to Africa Database | SAIS_CLA | Debt | 
| China-Latin America Finance Database | IAD_GEGI | Debt | 
| IJ Global Infrastructure Finance & Energy Transactions | IJ_Global | Equity, Debt | 
| Refinitiv Eikon M&A | REFINITIV_MA | Equity |
| fDi Markets | FDI_Markets | Equity |
| Refinitiv Eikon Loan | REFINITIV_LOAN | Debt |

### COFI Relational Database structure
The resulting COFI is structured as a relational database as follows:

<img src="Relational DB - final.png" alt="COFI Database Structure" title="Relational DB"> <!-- TODO: this picture is in the "11. Documentation" folder. -->

* **Power Plant**: Each row corresponds to a unique power plant.
* **City**: Contains data on cities, each of which is located in a province and country.
* **CITYKEY_BRIDGE_PP_C**: Table linking power plants to cities, allowing for multiple power plants in the same city.
* **Country**: Contains country-level information for each power plant.
* **Equity** and **Debt**: Each row represents an equity or debt investment in a power plant.
* **Transaction**: Records the contribution of individual companies to an investment.
* **Investor**: Information about investing companies.

### Data Extraction and Cleaning

### Data Matching and Merging

* **Incremental Approach**: each source is processed on its own and added to the existing database by checking for matches each time. The data from the most reliable source is prioritized according to these rankings: 
    * Debt Investments: SAIS_CLA = IAD_GEGI > BU_CGEF > IJ_Global > REFINITIV_LOAN > BU_CGP > Aiddata
    * Equity Investments: IJ_Global > REFINITIV_MA > FDI_Markets > BU_CGP
* **Exact ID Matching**: if available, matching using IDs shared among different data sources is prioritized.
* **Custom methodology**: two algorithms have been developed, one for power plant matching and one for investment matching. These algorithms can be found here. <!-- TODO: add the link to the code of the functions. --> They share the same structure:
    1. *Determining if there is enough information*: check if there is the information that is the minimum needed to identify the power plants and investments.
    2. *Running the matching*: There are three sub-steps:
        1. *Exact matching*: there must be an exact match on the most important columns (e.g., country).
        2. *Fuzzy matching*: the match on these columns is more loose: the numerical values are within a certain range from what is being matched; the texts are similar. 
        3. *Ranking*: of all the potential matches left, the best match is determined as the best ranked based on the similarities of its columns to what is being matched. 

## Repository structure

├── README.md
