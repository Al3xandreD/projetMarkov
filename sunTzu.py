import utilsMarkov
import pandas as pd
import sqlite3
import numpy as np
from Country import Country

def extract_data(path):
    '''
    Extract data from the UCDP-PRIO database
    :param path: file path
    :return:
    '''

    return pd.read_excel(path)

def data_to_sql(data):
    '''
    Converts the xlsx file to sqlite database
    :param data:
    :return:
    '''

    conn = sqlite3.connect('database.db')  # opening
    data.to_sql('Conflicts', conn, if_exists='replace', index=False)  # transfering
    cur = conn.cursor()  # cursor for requests

    return cur


def newCreate_contries(data):
    '''

    :param data:
    :return:
    '''
    list_countries = []
    list_name_countries = []
    total_countriesA = data['side_a'].value_counts()  # side A  in the database
    total_countriesB = data['side_b'].value_counts()  # side B in the database

    for elementA in total_countriesA.index.tolist():
        country_lstring=elementA.split(",")  # split if contains multiple countries
        for countryA in country_lstring:
            if "Government" in countryA.split(" "): # check if government
                if countryA[0] == " " and countryA[1:] not in list_name_countries: # there is blank space at first and country not in list
                    list_name_countries.append(countryA[1:])
                if countryA not in list_name_countries:   # no blank space and country not in list
                    list_name_countries.append(countryA)

    for elementB in total_countriesB.index.tolist():
        country_lstring=elementB.split(",")  # split if contains multiple countries
        for countryB in country_lstring:
            if "Government" in countryB.split(" "): # check if government
                if countryB[0]==" " and countryB[1:] not in list_name_countries:
                    list_name_countries.append(countryB[1:])   # incrementing in the side A
                if countryB not in list_name_countries:
                    list_name_countries.append(countryB)

    # transfer to sql
    conn = sqlite3.connect('database.db')   # opening
    data.to_sql('Conflicts', conn, if_exists='replace', index=False)  # transfering
    cur = conn.cursor()  # cursor for requests

    # creating countries
    for country in list_name_countries:
        # fetching current state of intensity
        try:
            cur.execute(
                "SELECT intensity_level FROM Conflicts WHERE (side_a='" + country + "' OR side_b LIKE '%" + country + "%') ORDER BY year DESC LIMIT 1")
            res1=cur.fetchall()
            list_countries.append(Country(country, res1[0][0]))
        except IndexError as e:
            print("Nothing returned by the request")

    # closing database
    conn.close()

    return list_countries

def comp_nbConf(country, cur):
    '''
    Computes the number of conflicts in which a given country was involved
    :param country:
    :param data: Panda dataframe
    :return:
    '''
    #print(country.name)
    cur.execute("SELECT COUNT(DISTINCT conflict_id) AS nombre_total_de_conflits FROM Conflicts WHERE side_a = '"+country.name+"' OR side_b LIKE '%" + country.name + "%';")
    country.nb_conf=cur.fetchall()[0][0]

def comp_nbTransition(country, cur):
    '''

    :param country:
    :param cur:
    :return:
    '''

    cur.execute("SELECT COUNT(*) AS nombre_total_de_conflits FROM Conflicts WHERE side_a = '" + country.name + "' OR side_b LIKE '%" + country.name + "%';")
    country.nb_transition = cur.fetchall()[0][0]


def newCompProb_ab(country, cur, a, b):
    '''

    :param country:
    :param cur:
    :param a: starting state
    :param b: end state
    :return:
    '''

    count=0

    cur.execute("SELECT DISTINCT conflict_id FROM Conflicts WHERE side_a = '" + country.name+ "'")
    list_cId=cur.fetchall()

    for i in range(len(list_cId)): # for all conflicts where the country is involved
        cur.execute("SELECT * FROM CONFLICTS WHERE conflict_id = '"+str(list_cId[i][0])+"'") # toute les lignes du conflit
        list_lineC=cur.fetchall()

        for j in range(len(list_lineC)-1):
            if list_lineC[j][11]==a and list_lineC[j+1][11]==b: # test to see intensity level state between two years of conflict
                count+=1

    if country.nb_transition==0:
        return 0
    else:
        return count/country.nb_transition

def newCompA(country, cur):
    '''
    Computes A using the number of transitions
    :param country:
    :param cur:
    :return:
    '''

    country.A[0][0] = newCompProb_ab(country, cur, 0,0)
    country.A[0][1] = newCompProb_ab(country, cur, 0,1)
    country.A[0][2] = newCompProb_ab(country, cur, 0,2)
    country.A[1][0] = newCompProb_ab(country, cur, 1,0)
    country.A[1][1] = newCompProb_ab(country, cur, 1,1)
    country.A[1][2] = newCompProb_ab(country, cur, 1,2)
    country.A[2][0] = newCompProb_ab(country, cur, 2,0)
    country.A[2][1] = newCompProb_ab(country, cur, 2,1)
    country.A[2][2] = newCompProb_ab(country, cur, 2,2)

def compAaugmentation(country, cur):
    '''
    Computes A using the number of transitions and forcing stochastic on p10 and p20
    :param country:
    :param cur:
    :return:
    '''

    country.A[1][1] = newCompProb_ab(country, cur, 1, 1)
    country.A[1][2] = newCompProb_ab(country, cur, 1, 2)
    country.A[1][0] = 1 - country.A[1][1] - country.A[1][2]

    country.A[2][1] = newCompProb_ab(country, cur, 2, 1)
    country.A[2][2] = newCompProb_ab(country, cur, 2, 2)
    country.A[2][0] = 1 - country.A[2][1] - country.A[2][2]

    country.A[0][1] = country.A[1][1] + country.A[1][2]  # using the number of conflicts starting to intensity level 1
    country.A[0][2] = country.A[2][1] + country.A[2][2]  # using the number of conflicts starting to intensity level 2
    country.A[0][0] = 1 - country.A[0][1] - country.A[0][2]

    # country.A[0][1] = newCompProb_ab(country,cur,0,1)
    # country.A[0][2] = newCompProb_ab(country, cur, 0, 2)
    # country.A[0][0] = 1-country.A[0][1]-country.A[0][2]

def testStochastique(country):
    '''
    
    :param country: 
    :param cur: 
    :return: 
    '''
    if np.array_equal(np.sum(country.A, axis=1),np.array([1,1,1])):
        print(country.name, "True")

def compB():
    '''
    Computes the observation matrix
    :return:
    '''

    B=np.array([[0.9, 0.07, 0.03],
                [0.05, 0.9, 0.05],
                [0.03, 0.07, 0.9]])

    return B

def createSequenceDP(states):
    '''
    Creates all sequences possible between the states
    :param states:
    :return:
    '''
    # states=[0, 1, 2]
    l_seq=[]  # list of sequence

    for state1 in states:
        for state2 in states:
            for state3 in states:
                for state4 in states:
                    for state5 in states:
                        X=[state1, state2, state3, state4, state5]
                        l_seq.append(X)

    return l_seq

def compProbSequence(sequence, observation, A, B, pi):
    '''
    Computes the probabilty for a given sequence to happen
    :param sequence:
    :return:
    '''
    # sequence=[0,1,2,2,0]

    prob=pi[sequence[0]]*B[sequence[0]][observation[0]] *A[sequence[0]][sequence[1]]*B[sequence[1]][observation[1]] *A[sequence[1]][sequence[2]]*B[sequence[2]][observation[2]] *A[sequence[2]][sequence[3]]*B[sequence[3]][observation[3]] *A[sequence[3]][sequence[4]]*B[sequence[4]][observation[4]]
    return prob

def solDP(l_seq, observation, A, B, pi):
    '''
    Computes the best sequence of states according to Dynamic Programming solution
    :param l_seq:
    :return:
    '''

    l_prob=[]

    for seq in l_seq:
        prob = compProbSequence(seq,observation, A, B, pi)
        l_prob.append(prob)

    prob_max=max(l_prob)
    index=l_prob.index(prob_max)

    return l_seq[index], l_prob, prob_max

def dispResDP(l_seq, l_prob):
    '''
    Displays the probabilty of realisation for each sequence
    :param l_seq:
    :param l_prob:
    :return:
    '''

    print("Sequence", "                  ", "Probabilité")
    for k in range(len(l_seq)):
        print(l_seq[k], "                   ",l_prob[k])


if __name__ == "__main__":

    # TODO HMM
    # TODO: baum welsh retrouver le modèle

    excel_path = "UcdpPrioConflict_v23_1.xlsx"
    data = extract_data(excel_path)
    list_countries = newCreate_contries(data)
    cur = data_to_sql(data)  # sqlite cursor


    Augmentation=True

    transition=True # on calcul les proba en se basant sur le nombre de transition

    for country in list_countries:
        if Augmentation and transition:
            print('#######################')
            comp_nbTransition(country, cur)
            compAaugmentation(country, cur)
            testStochastique(country)
            print(country.toString())
            print('#######################')

        if Augmentation and not transition:
            comp_nbConf(country, cur)
            compAaugmentation(country, cur)
            testStochastique(country)
            print(country.toString())

        if not Augmentation:
            comp_nbConf(country, cur)
            newCompA(country,cur)
            print(country.toString())

    my_country=list_countries[0]
    # verification de la propriété d'ergodicité
    pi_n=utilsMarkov.isErgodique(my_country.A, 0.1)
    print("Valeur final du vecteur transition", pi_n)

    # simulation du HMM
    starting_state=0
    l_states=utilsMarkov.newSimulation(my_country.A,starting_state,5)
    print("Sequence d'etat après simulation", l_states)

    # estimation d'une sequence d'états à partir d'observation
    Y=[0,1,2,2,0]   # sequence d'observation des etats de la région
    pi_0=np.array([0.7,0.2,0.1])  # distribution initiale

    # solution DP
    l_seq=createSequenceDP([0, 1, 2])
    X_best, l_prob, prob_max=solDP(l_seq,Y, my_country.A,compB(), pi_0)
    print("La sequence la plus semblable au sens de la programmation dynamique est: ", X_best, " avec une probabilité de réalisation de: ",prob_max )
    #print("Solution par forward algorithm", utilsMarkov.forwardAlgorithm(Y, my_country.A,compB(),pi_0))
    dispResDP(l_seq,l_prob)

    # solution HMM