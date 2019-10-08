# tennis_betting
I.	Overview

In this summer-long project, we predicted the outcomes of men’s singles tennis matches. Using data from 2003-2017, we generated new features that we then passed into a neural network that predicted win percentages with probabilities of victory for each player. Comparing our predicted probabilities with the odds from various bookmakers, we came up with an “edge” that we believed we had on each match. Using this edge, we followed a fractional Kelly Criterion approach to simulate betting on matches from 2013-2017 with positive results showing a large increase in overall wealth. 



II.	Cleaning and Merging Data

Initially, we had two separate datasets containing different statistics for the same set of matches. Using the Python merge function, we were able to combine both datasets and use the statistics from both. We filled in missing data using average values from the rest of the data. In retrospect, a better solution would’ve been to use linear regression to predict the missing variables. 


III.	Data Exploration and Visualization


Our initial publicly available dataset contained a number of features about both players in the match, such as ATP ranking, age, and height. The link to the initial dataset is available here: https://www.kaggle.com/jordangoblet/atp-tour-20002016. Also included were a number of match-specific features such as date, tournament name, tournament round, surface, and location. We were also provided a set of odds from various bookmakers for each game (Bet365, Bet&Win, Ladbrokes, Pinnacles...). Most importantly, the statistics in the second dataset provided set scores, break points faced and won (per player), 1st serve percentages, aces, and more of the results of the match. 

Before trying to beat the bookmakers, we first wanted to explore how correct the bookmakers really are. First, we simply took the average odds for all the bookmakers who posted odds for each match. Odds in our dataset were listed as decimals (return per dollar bet). For example, a player with odds of 2.5 returns a total of $2.50 for every $1 bet should they win. It’s easy to see that a player with odds under 2.0 is favored and above is an underdog. However, these odds include a rake, or hold, (typically 3-7 percent) that the bookmaker’s use to generate profit. To calculate implied percentages of each player winning (according to the bookmaker’s), we took the inverse of each player’s odds and then adjusted for the rake. Let the bookmaker’s odds of player A winning be x. Let the bookmaker’s odds of player B winning be y. So, player A’s implied odds are (1/x)/((1/x) + (1/y)). Similarly, player B’s implied odds are (1/y) /((1/x) + (1/y)). In figure 1, we bucket matches by implied win percentages and graph this implied percentage vs actual win percentage. 

 

Figure 1. The bookmakers tend to be extremely accurate, although they may undersell large favorites.

IV.	Feature Generation.

In order to successfully predict tennis matches, we needed to use our existing dataset to generate meaningful input features. For each player, we generated a number of useful recent features including rank trend, close set win/loss percentages in the last 15 matches, games played in the last seven days, winning percentage on surface. For a full list and description of each feature, please refer to the appendix [1]. In addition to player specific features, we also used match features such as surface, stage of tournament, major/non-major, and more. We encoded all categorical data as one-hot vectors. For example, ‘clay’ might be stored as [1, 0, 0, 0] while ‘grass’ is [0,1,0,0]. All other numerical variables were normalized by scaling between -1 and 1. To account for bias in a player being listed first, all stats were reversed in order to artificially double rows. 

V.	Neural Network and Prediction

We tried a number of different techniques to generate predictions. Regardless of method used, we used the average prediction from inputting Player A’s stats first and then Player B’s first. This is consistent with the artificial doubling technique described earlier.

Our first approach made use of simple logistic regression. To train, we used Elastic Net regularization with l1 ratios [0.1, 0.1] and the ‘saga’ solver. A naïve way of measuring the accuracy of our predictions is to compare the percentage of time we are able to pick the winner compared to the bookmaker’s favorite winning the matches. In matches from 2013-2017, the bookmaker predicted the winner correct 70.01% of the time. The higher-ranked player won 65.89 % of the time. Our logistic regression classifier did slightly better, predicting the winner correct about 70.69 % of the time. 

A slightly improved approach made use of a neural network with 2 fully-connected Dense layers with ReLu activations. Our input was 24 dimensions with a single sigmoid activated output corresponding to the probability of a player winning the match. To train, we used a binary- cross entropy loss function and a RMS prop optimizer with 30 epochs and a batch size of 128. The neural network was by far the highest performing, with an accuracy of 72.13 %. Figure 2 shows the same graph from Figure 1 with our new predictions added in green. 

 
Figure 2: The bookmaker’s predictions are in blue, our predictions are in green, and red is the ideal predictive model.

Of course, the real test of our accuracy in this case is whether or not our predictions can beat the odds and make money. This testing is described in the following section.

I.	Simulated Betting and Kelly Criterion

We bet a percentage of our wealth on each match according to the Kelly criterion. We simply follow the formula to bet in accordance with the bookmaker’s odds and our perceived “edge” (predicted probability NN - implies odds of the bookmakers). Here is a link to the Kelly criterion: https://en.m.wikipedia.org/wiki/Kelly_criterion. Assuming no max bet size, following a fractional betting approach takes an initial wealth of 1 to 10^44 over 3 ½ years, astoundingly positive results. Figures 3 and 4 show the growth of wealth over time. 


 

Figure 2: Wealth over time using fractional Kelly betting and alpha of 0.69.

 
Figure 3: Wealth over time using fractional Kelly betting and alpha of 0.69 (log graph). 
I think the most interesting part of my study was investigating the optimal fractional Kelly. Kelly proved that if you know with exact certainty the probability of an event happening, then his formula provides the best percentage of your wealth to wager on each bet way to maximize your wealth over time without going broke. However, most probabilities and estimates in the real world are far from exact, so most practitioners rely on a fractional Kelly, which is basically the percentage of wealth Kelly recommends multiplied by a number between 0 and 1. This reduces variance and subsequently reduces the chances of going broke. While the optimal fraction (alpha) is still debated, most studies find that a number close to 1/2 works well and indeed gives better results than a full Kelly approach. Interestingly enough, in my simulations, I found the optimal value to be close to 0.69.  Figure 5 is a graph showing the ending wealth (y axis) for different values of alpha, or fraction, (x axis) of the Kelly formula that I use. I have also attached a paper that I think you may find of interest where the authors talk about their use of a fractional Kelly approach and show its advantages over a full Kelly when exact probabilities of an event occurring are unknown. 

-   Lack of I.I.D
-	Parameter uncertainty

 

Figure 5: Ending wealth for different values of alpha. In our model, the optimal value was about 0.69.


II.	Conclusion and Next Steps

Although our results were remarkably successful, there are a number of concerns before carrying out the same betting in the real world. First, we are unsure how long ago before the match these odds were recorded. Generally, market makers, are the first to post odds and be open to betting. During this time, however, max bet sizes are very small (in the range of a few hundred dollars). After the line has been so-called hammered into place by a number of professional sports bettors, then the betting limits are increased, and other retail bookmakers copy over the line form the market maker. If the odds in our dataset are the initial ones posted by the market maker, then our program is still untested against the lines we would actually be able to bet large amounts into. 

Secondly, even at retail bookmakers, our max bet size would have to be within reason - bookmakers usually have max bets in the thousands. This condition drastically limits our upside of wealth. 

Finally, bookmakers know that some people are smarter than them and are better at predicting outcome of games. Remember, all bookmaker’s care about (usually) is getting even action and taking the hold without risk. However, if they notice a customer with a consistent pattern of winning (especially one who is betting every game as we would do), they raise their alarms and start tracking the customer. The second the bookmaker figures out how much money we are making, we would not only be banned from their website but also be blacklisted from betting at any major bookmaker in Vegas and across the world. The biggest challenge of all is to find someone to keep taking our bets once we have shown a consistent pattern of success.

Obviously, the next steps are to try this out on future events. We are working on a new script that will scrape match statistics from the ATP website daily.  After adding these new matches to our dataset, we can re-run our algorithms to generate new predictions for tomorrow’s matches. All that’s left is for us to bet it and see how it goes – stay tuned. 

