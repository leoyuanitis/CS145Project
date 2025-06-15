# CS145Project Spring 25

The specifics of implementation and environment config & setup could be found on [https://github.com/FrancoTSolis/CS145-RecSys](https://github.com/FrancoTSolis/CS145-RecSys) by Fang Sun at UCLA CS. Thank you Fang!


To run the models we presented: 
  1. make suree to clone the above repo by Fang completely and follow the setup steps and dependencies requirements;
  2. replace recommender_analysis_visualization.py in the given pipeline from the above repo with checkpoint1_code.py;
  3. in terminal, enter uv run checkpointX_code.py (checkpoint1 is on content based models; 2 is on sequence based models; 3 is on graph based model)
The best model in terms of discounted revenue could be found in submission.py; We actually inplemented around 10 versions of models that have the same performances in the [leaderboard](http://scai2.cs.ucla.edu:5431/), but for demonstration, we provide the least fancy model here. 

Some side notes:
- We advise you to avoid putting all check points code together and running a integrated recommender_analysis_visualization.py since it would take extremely long and break the workflow very often;
- You may encounter many warnings messages on terminal when running the code -- we tried very hard to supress them as much as possible and asked LLM to help us on it, but unfortunately there are still a lot of them... Rest assured, they won't affect the implementation of recommender classes!
