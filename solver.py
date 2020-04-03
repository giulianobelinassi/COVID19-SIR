#!/usr/bin/python
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import json
import ssl
import urllib.request
import pdb


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    country1="Brazil"

    if country1=="Brazil":
        date="3/3/20"
        s0=25000
        i0=27
        r0=-35

    if country1=="China":
        date="1/22/20"
        s0=170000
        i0=400
        r0=-80000

    if country1=="Italy":
        date="1/31/20"
        s0=160000
        i0=23
        r0=15

    if country1=="France":
        date="2/25/20"
        s0=95e3
        i0=250
        r0=-75

    if country1=="United Kingdom":
        date="2/25/20"
        s0=80000
        i0=22
        r0=-5

    if country1=="US":
        date="2/25/20"
        s0=470000
        i0=10
        r0=-50

    parser.add_argument(
        '--countries',
        dest='countries',
        type=str,
        default=country1)
    
    parser.add_argument(
        '--download-data',
        dest='download_data',
        default=True
    )

    parser.add_argument(
        '--start-date',
        dest='start_date',
        type=str,
        default=date)
    
    parser.add_argument(
        '--prediction-days',
        dest='predict_range',
        type=int,
        default=150)

    parser.add_argument(
        '--S_0',
        dest='s_0',
        type=int,
        default=s0)

    parser.add_argument(
        '--I_0',
        dest='i_0',
        type=int,
        default=i0)

    parser.add_argument(
        '--R_0',
        dest='r_0',
        type=int,
        default=r0)

    args = parser.parse_args()

    country_list = []
    if args.countries != "":
        try:
            countries_raw = args.countries
            country_list = countries_raw.split(",")
        except Exception:
            sys.exit("QUIT: countries parameter is not on CSV format")
    else:
        sys.exit("QUIT: You must pass a country list on CSV format.")

    return (country_list, args.download_data, args.start_date, args.predict_range, args.s_0, args.i_0, args.r_0)


def sumCases_province(input_file, output_file):
    with open(input_file, "r") as read_obj, open(output_file,'w',newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
               
        lines=[]
        for line in csv_reader:
            lines.append(line)    

        i=0
        ix=0
        for i in range(0,len(lines[:])-1):
            if lines[i][1]==lines[i+1][1]:
                if ix==0:
                    ix=i
                lines[ix][4:] = np.asfarray(lines[ix][4:],float)+np.asfarray(lines[i+1][4:] ,float)
            else:
                if not ix==0:
                    lines[ix][0]=""
                    csv_writer.writerow(lines[ix])
                    ix=0
                else:
                    csv_writer.writerow(lines[i])
            i+=1     


def download_data(url_dictionary):
    pass
    #Lets download the files
    #for url_title in url_dictionary.keys():
    #    urllib.request.urlretrieve(url_dictionary[url_title], "./data/" + url_title)


def load_json(json_file_str):
    # Loads  JSON into a dictionary or quits the program if it cannot.
    try:
        with open(json_file_str, "r") as json_file:
            json_variable = json.load(json_file)
            return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_file_str)


class Learner(object):
    def __init__(self, country, loss, start_date, predict_range,s_0, i_0, r_0):
        self.country = country
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0


    def load_confirmed(self, country):
        df = pd.read_csv('data/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_recovered(self, country):
        df = pd.read_csv('data/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_dead(self, country):
        df = pd.read_csv('data/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, death, healed, country, s_0, i_0, r_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SIR(t,y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        extended_healed = np.concatenate((healed.values, [None] * (size - len(healed.values))))

        sir = solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1))
        R = sir.y[2][0:len(death)]

        optimal = minimize(loss2, gamma*0.02, args=(gamma, recovered, healed, death),
                          bounds=[(0.00000001, gamma),])

        #I tried to change to odeint but I do not know how to make sir.y[] data structure
        #the answer of odeint is an array not a data frame
        #y=[res[:,0], res[:,1], res[:,2]]
        #just need to put it in sir as sir.y[0], sir.y[1], sir.y[2]

        print(optimal)

        a = optimal.x[0]
        b = gamma - a

        prediction_death = a*sir.y[2]/gamma
        prediction_healed = sir.y[2] - prediction_death

        return new_index, extended_actual, extended_recovered, extended_death, sir, prediction_death, prediction_healed, extended_healed

    def train(self):
        self.death = self.load_dead(self.country)
        self.healed = self.load_recovered(self.country)
        self.recovered = self.healed + self.death
        self.data = self.load_confirmed(self.country) - self.recovered

        optimal = minimize(loss, [0.001, 0.001], args=(self.data, self.recovered, self.s_0, self.i_0, self.r_0), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
        print(optimal)
        beta, gamma = optimal.x
        self.optimal_beta = beta
        self.optimal_gamma = gamma

    def plot(self):
        beta = self.optimal_beta
        gamma = self.optimal_gamma

        death = self.death
        healed = self.healed
        recovered = self.recovered
        data = self.data

        new_index, extended_actual, extended_recovered, extended_death, prediction, prediction_death, prediction_healed, extended_healed = self.predict(beta, gamma, data, recovered, death, healed, self.country, self.s_0, self.i_0, self.r_0)

        df = pd.DataFrame({'Infected data': extended_actual,
                            'Death data': extended_death,
                            'Susceptible': prediction.y[0],
                            'Infected': prediction.y[1],
                            'Predicted Recovered (Alive)': prediction_healed,
                            'Predicted Deaths': prediction_death,
                            'Recovered (Alive)': extended_healed},
                            index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(self.country + '.png')


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point
    def SIR(y,t):
    # def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]

    # solve ODE by solve_ivp    
    # solution = solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    # l1 = np.sqrt(np.mean((sir.y[1] - data)**2))
    # l2 = np.sqrt(np.mean((sir.y[2] - recovered)**2))

    # solve ODE by odeint
    y0=[s_0,i_0,r_0]
    tspan=np.arange(0, size, 1)
    res=odeint(SIR,y0,tspan)
    l1 = np.sqrt(np.mean((res[:,1] - data)**2))
    l2 = np.sqrt(np.mean((res[:,2] - recovered)**2))

    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2

def loss2(a, gamma, recovered, healed, death):
    size = len(recovered)
    b = gamma - a

    estimated_death = a*(recovered/gamma)
    estimated_healed = recovered-estimated_death

    l1 = np.sqrt(np.mean((estimated_death - death)**2))
    l2 = np.sqrt(np.mean((estimated_healed - healed)**2))

    alpha = 0.9
    return alpha*l1 + (1-alpha)*l2

def main():

    countries, download, startdate, predict_range , s_0, i_0, r_0 = parse_arguments()

    if download:
        data_d = load_json("./data_url.json")
        download_data(data_d)

    sumCases_province('data/time_series_19-covid-Confirmed.csv', 'data/time_series_19-covid-Confirmed-country.csv')
    sumCases_province('data/time_series_19-covid-Recovered.csv', 'data/time_series_19-covid-Recovered-country.csv')
    sumCases_province('data/time_series_19-covid-Deaths.csv', 'data/time_series_19-covid-Deaths-country.csv')

    for country in countries:
        learner = Learner(country, loss, startdate, predict_range, s_0, i_0, r_0)
        #try:
        learner.train()
        learner.plot()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')
           

if __name__ == '__main__':
    main()
