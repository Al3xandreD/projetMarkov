#import utilsMarkov
import pandas as pd
import sqlite3
from Country import Country

def extract_data(path):
    '''
    Extract data from the UCDP-PRIO database
    :param path: file path
    :return:
    '''

    return pd.read_excel(path)

def create_country(data, nb_countries):
    '''
    Creates the interest countries included in the dataframe
    :param data: Panda dataframe
    :param nb_countries: number of interest countries
    :return: list_country: list of countries
    '''
    list_countries = []
    total_countriesA=data['side_a'].value_counts()  # side A  in the database
    total_countriesB=data['side_b'].value_counts()  # side B in the database

    # counting total number of years in conflicts in both sides
    for elementB in total_countriesB.index.tolist():
        if "Government" in elementB.split(" "): # check if government
            government_name=elementB.split(" ")[2]    # government name in side B
            if "Government"+government_name in total_countriesA.index.tolist(): # country already is in the side A
                total_countriesA['Government of '+government_name]+=1    # incrementing in the side A
            else: # country not in side A
                total_countriesA['Government of '+government_name]=1

    # transfer to sql
    conn = sqlite3.connect('database.db')   # opening
    data.to_sql('Conflicts', conn, if_exists='replace', index=False)  # transfering
    cur = conn.cursor()  # cursor for requests

    print(total_countriesA)

    # creating countries
    for country in total_countriesA.index.tolist():
        # fetching current state of intensity
        cur.execute("SELECT type_of_conflict FROM Conflicts WHERE side_a=='" + country + "' AND year==(SELECT MAX(year) FROM Conflicts WHERE side_a=='" + country + "')")
        res1=cur.fetchall()
        list_countries.append(Country(country.split(" ")[2], res1))

    # closing database
    conn.close()

    return list_countries[:nb_countries+1]

def comp_nbConf(country, data):
    '''
    Computes the number of conflicts in which a given country was involved
    :param country:
    :param data: Panda dataframe
    :return:
    '''

    conn = sqlite3.connect('database.db')
    data.to_sql('Conflicts', conn, if_exists='replace', index=False)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT COUNT('conflict_id') FROM Conflicts WHERE 'side_a'==country.name OR 'side_b'==country.name")
    country.nb_conf=cur.fetchall()

    conn.close()

def comp_p00(country, data):
    '''
    Computes the probability to pass from  intensity level 0 to 0
    :param country:
    :param data:
    :return:
    '''

    # connexion to database
    conn = sqlite3.connect('database.db')
    data.to_sql('Conflicts', conn, if_exists='replace', index=False)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT 'conflict_id' FROM Conflicts WHERE 'side_a'==country.name AND 'side_b'==country.name")
    all_conflicts=cur.fetchall()
    for conflict in all_conflicts:    # on verifie pour chaque conflit impliquant le pays

        # year n
        cur.execute("SELECT 'year' FROM Conflicts WHERE ('conflict_id'==conflict)")
        year=cur.fetchall()

        # year n+1
        year+1

        # number of conflicts transitioning
        cur.execute("SELECT COUNT('conflict_id') FROM Conflicts WHERE ('side_a'==country.name OR 'side_b'==country.name) AND 'intensity_level_0'==0 AND"
                    "")

def comp_p01(country, data):
    '''
    Computes the probability to pass from  intensity level 0 to 1
    :param country:
    :param data:
    :return:
    '''

def comp_p02(country, data):
    '''
    Computes the probability to pass from  intensity level 0 to
    :param country:
    :param data:
    :return:
    '''

def comp_p11(country, data):
    '''
    Computes the probability to pass from  intensity level 1 to 1
    :param country:
    :param data:
    :return:
    '''

def comp_p12(country, data):
    '''
    Computes the probability to pass from  intensity level 1 to 2
    :param country:
    :param data:
    :return:
    '''

def comp_p10(country, data):
    '''
    Computes the probability to pass from  intensity level 1 to 0
    :param country:
    :param data:
    :return:
    '''

def comp_p22(country, data):
    '''
    Computes the probability to pass from  intensity level 2 to 2
    :param country:
    :param data:
    :return:
    '''

def comp_p21(country, data):
    '''
    Computes the probability to pass from  intensity level 2 to 1
    :param country:
    :param data:
    :return:
    '''

def comp_p20(country, data):
    '''
    Computes the probability to pass from  intensity level 2 to0
    :param country:
    :param data:
    :return:
    '''

def compA(country, data):
    '''
    Computes the transition matrix for a given country
    :param country:
    :param data: Panda dataframe
    :return: A transition matrix
    '''


    conn = sqlite3.connect('database.db')
    data.to_sql('Conflicts', conn, if_exists='replace', index=False)
    cur = conn.cursor()







if __name__ == "__main__":

    # TODO: calculer les probabilités de transition
    # TODO: pour chaque pays, déterminer la matrice de transition

    # TODO: simuler l'effet de voisinage par observation dynamique
    # TODO: ajouter les autres variables d'observation et matrice d'observation

    # TODO: simulation du HMM
    # TODO: sequence d'observation, solution DP, HMM
    # TODO: baum welsh retrouver le modèle

    excel_path="UcdpPrioConflict_v23_1.xlsx"
    data=extract_data(excel_path)
    #print(data)
    list_countries=create_country(data,3)
    for country in list_countries:
        description=country.toString()
        print(description)