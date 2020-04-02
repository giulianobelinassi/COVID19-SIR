# COVID-19 SIR Model Estimation
SIR model estimation on COVID-19 cases dataset. There is a blog post describing the detail of the SIR model and COVID-19 cases dataset.

- [https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html](https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html)

![japan](/Japan.png)

## Usage

All dependencies are resolved by [Pipenv](https://pipenv.kennethreitz.org/en/latest/)

```
$ pipenv shell
$ python solver.py
```

Option to run
```
usage: solver.py [-h] [--countries COUNTRY_CSV] [--download-data]
                 [--start-date START_DATE] [--prediction-days PREDICT_RANGE]
                 [--S_0 S_0] [--I_0 I_0] [--R_0 R_0]

optional arguments:
  -h, --help            show this help message and exit
  --countries COUNTRY_CSV
                        Countries on CSV format. It must exact match the data
                        names or you will get out of bonds error.
  --download-data       Download fresh data and then run
  --start-date START_DATE
                        Start date on MM/DD/YY format ... I know ...It
                        defaults to first data available 1/22/20
  --prediction-days PREDICT_RANGE
                        Days to predict with the model. Defaults to 150
  --S_0 S_0             S_0. Defaults to 100000
  --I_0 I_0             I_0. Defaults to 2
  --R_0 R_0             R_0. Defaults to 0
```


## Data Sources

The data used by this simulation is available in:

- [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19)

## Model

This program implements a modification of the SIR (Subject Infected Recovered)
model. SIR consists of the following system of equations:

![SIR](/equations/images/sir.gif)

Where:
- S(t) is the number of people subject to infection at time t.
- I(t) is the number of people infected at time t.
- R(t) is the number of people recovered from infection at time t.
- N is the population size.
- β is how intense the disease spreads.
- γ is how fast the people recovers from it.

Therefore, SIR model doesn't model the number of deaths because it counts it
as Recovered. Therefore, in Recovered we have the number of deaths and the
number of people that is cured from the disease.

Therefore, we can split the recovered R(t) into two other functions, and
calculate their rate of change over time:

![RCD](/equations/images/cd.gif)

Where C(t) is the number of people cured from the disease, and D(t) is the
number of people killed by the disease.

Since the death rate seems to be a linear fraction of the infected people
(1.5% in China, for instance), then we can assume that:

![GAB](/equations/images/gamma_a_b.gif)

Where *a* is the death rate, and *b* is the cure rate.

Therefore, the SIR-D model can be viewed as:

![SIRD](/equations/images/sird.gif)

However we can calculate C(t) and D(t) by using R(t), because

![D](/equations/images/D.gif)

Where K is a constant. Here we assume that k = 0 because we start with 0 deaths.
Finally, we can also compute C(t) by:

![D](/equations/images/C.gif)

