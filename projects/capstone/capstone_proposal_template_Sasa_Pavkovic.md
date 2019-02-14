# Machine Learning Engineer Nanodegree
## Capstone Proposal
Sasa Pavkovic
February 14th, 2019

## Proposal
_(approx. 2-3 pages)_
Predicting future sales from Time Series data using Machine learning approaches. It is a part of a bigger domain of predicting a continous variable by using regression, hence the idea can be generalized to other types of similar challenges in predicting customer moetary value in the future. The data for the project will be taken from the Predict Future Sales competition on Kaggle. The link is provided here:
https://www.kaggle.com/c/competitive-data-science-predict-future-sales


### Domain Background
_(approx. 1-2 paragraphs)_

The domain that i chose comes from business where usually one of the relevant problems is to be able to predict buying behaviour of the customers by being able to predict sales. This problem is usually solved using regression as we are predicting a continuous variable. The additional component here is the temporal component where we can use some feature engineering approaches to map the temporal relations. Also, my whish is to try out if a deep learning approach can be used sucessfully for this particular problem. 

The need in business for solving this type of problem is very high as many questions can be answered by it. How much revenue can we expect from diffferent segments, in which segments to invest most marketing budgets or how to balance the marketing budget to maxmize CLV. Trying out a novel approach with Deep Neural Networks is a challenge but also very interesting. 

Papers: https://arxiv.org/pdf/1708.05123.pdf

### Problem Statement
_(approx. 1 paragraph)_

The problem that will be solved is predicting future sales based on the historical observed temporal data. Main idea for solving the problem is to do feature engineering such that the temporal structure is transformed into features. Then standard regression algorithms can be used to make a prediction for the next period. We want to predict how much future sales in dollars will happen in the next year.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

The datasets will be taken from the related Kaggle competition. 

We have daily historical sales data.  Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

File descriptions
sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
sample_submission.csv - a sample submission file in the correct format.
items.csv - supplemental information about the items/products.
item_categories.csv  - supplemental information about the items categories.
shops.csv- supplemental information about the shops.
Data fields
ID - an Id that represents a (Shop, Item) tuple within the test set
shop_id - unique identifier of a shop
item_id - unique identifier of a product
item_category_id - unique identifier of item category
item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
item_price - current price of an item
date - date in format dd/mm/yyyy
date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
item_name - name of item
shop_name - name of shop
item_category_name - name of item category

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
