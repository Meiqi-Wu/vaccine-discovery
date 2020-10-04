### ORIE 4741 Group Project
# COVID-19 Vaccine Discovery Project   
**Group members: Meiqi Wu(mw849), Yuwei Liu (yl3388), Jialiang Sun(js3553), Vaish Gajaraj (vg289)**  
  
We obtain our project inspiration and data from the following [this weibsite](https://www.kaggle.com/futurecorporation/epitope-prediction?select=input_covid.csv).

### 1. Project Goal:
We seek to identify the parts of COVID’s **antigen** protein that stimulate an immune response in the body. These specific regions are known as **B Cell epitopes**, and predicting their **protein sequence** is useful in the design and development of vaccines. These regions may induce an immune response in the human body capable of combating the coronavirus. A successful immune response from a vaccine that uses the designed B-cell will produce large amounts of antigen-specific antibodies. We would like to discover particular characteristics of a whole **antigen** protein in addition to the **target sequence**. A “vaccine” is simply a substance that can induce specific **antibodies** and that mimics the structure and function of an epitope. 

From a data set of COVID’s (our antigen of choice) protein sequences, we will parse out and output the sections that most likely are the epitopes. In doing so, we can consider the entirety of the antigen protein, and then medical researchers can take steps to create vaccines that target these specific regions. 


### 2. Relevance of the research question
2020 is an unusual year. COVID-19 spreaded out all over the world. Globally, as of October 2nd, there have been 34,079,542 confirmed cases of COVID-19, including 1,015,963 deaths, reported to WHO. There have been more than 7 million confirmed cases and over 205 thousand deaths due to COVID-19 in the United States alone. Even president Donald Trump and his wife have tested positive for COVID-19.

Since the first breakout of the notorious pandemic, there has been a worldwide shut down of companies, factories and schools, as well as substantial negative impact on the global economy. The shortage of medical resources and test kits once led to a huge increase in daily confirmed cases and massive panic among citizens. 

Currently, many countries, including the U.S., are still fighting hard to control COVID-19. Medical scientists from all over the world are developing COVID-19 vaccine, exhausting their abilities. An effective and efficient vaccine will be the key for the human being to win the battle against COVID-19. 

Hence, with reference to B Cell and SARS data, we look forward to undercovering the features of COVID-19’s antigen protein that stimulate an immune response in the body, and therefore to discover the significant regions that a COVID-19 vaccine can target. Although medical scientists have already finished this step and some vaccines are nearly put into use, we hope we can develop a further understanding of COVID-19 as well as machine learning methods through this research.


### 3. Describe the dataset:
A piece of peptide taken from a parent protein is applied to B cell culture, and we want to predict whether the piece of peptide will introduce a specific antibody on the surface of the B cell. The dataset contains features of the peptide and its parent protein, and the label whether antibody is introduced on the surface of B-cell. 
Data columns:
- *parent_protein_id*: Id of parent protein.   
- *protein _seq*: Sequence of parent protein.   
- *start_position*: Start position of the peptide in the parent protein.  
- *end_position*: End position of the peptide in the parent protein. 
- *peptide_seq*: Sequence of peptide.
- *chou_fasman*: A feature of the peptide describing beta return.
- *emini*: A feature of the peptide describing relative surface accessibility.
- *kolaskar_tongaokar*: A feature of the peptide describing antigenicity. 
- *parker*: A feature of the peptide describing hydrophobicity.
- *isoelectric_point*: A feature of the parent protein. 
- *aromaticity*: A feature of the parent protein.
- *hydrophobicity*: A feature of the parent protein.
- *stability*: A feature of the parent protein. 

### 4. How the dataset will help answer the question:
Our goal is to predict whether an amino acid peptide (epitope) would trigger antibody-inducing activity. Our data set contains all combinations of a total of 14362 epitopes and 757 proteins, as well as features of both the epitopes and proteins, which provides us with rich information to predict certain epitope’s performance. In our training data we also have labels corresponding to whether the epitope could stimulate antibody-inducing activity. Difference epitopes tend to have distinct features that contribute to its performance, therefore we are able to identify certain epitopes that have desired activity. 
