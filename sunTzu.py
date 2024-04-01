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


def create_countries(data, nb_countries):
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

    # creating countries
    for i in range(len(total_countriesA.index.tolist())):
        country=total_countriesA.index.tolist()[i]
        # fetching current state of intensity
        try:
            #cur.execute("SELECT type_of_conflict FROM Conflicts WHERE (side_a=='" + country + "' OR side_b=='"+ country+ "') ORDER BY year DESC LIMIT 1")

            cur.execute(
                "SELECT type_of_conflict FROM Conflicts WHERE (side_a='" + country + "' OR side_b LIKE '%" + country + "%') ORDER BY year DESC LIMIT 1")

            res1=cur.fetchall()
            list_countries.append(Country(country, res1[0][0]))
        except IndexError as e:
            print("Nothing returned by the request")

    # closing database
    conn.close()

    return list_countries#[:nb_countries]

def comp_nbConf(country, cur):
    '''
    Computes the number of conflicts in which a given country was involved
    :param country:
    :param data: Panda dataframe
    :return:
    '''

    cur.execute("SELECT COUNT(*) AS nombre_total_de_conflits FROM Conflicts WHERE side_a = '"+country.name+"' OR side_b = '"+country.name+"';")
    country.nb_conf=cur.fetchall()[0][0]

def comp_p00(country, cur):
    '''
    Computes the probability to pass from  intensity level 0 to 0
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 0 AND c2.intensity_level = 0 AND c2.year < c1.year) AS subquery;")

    if country.nb_conf==0:
        p00=0
    else:
        p00=cur.fetchall()[0][0]/country.nb_conf
    #print(p00)
    return p00

def comp_p01(country, cur):
    '''
    Computes the probability to pass from  intensity level 0 to 1
    :param country:
    :param data:
    :return:
    '''

    cur.execute("SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '"+country.name+"' AND c2.side_a = '"+country.name+"') AND c1.intensity_level = 1 AND c2.intensity_level = 0 AND c2.year < c1.year) AS subquery;")

    if country.nb_conf==0:
        p01=0
    else:
        p01= cur.fetchall()[0][0] / country.nb_conf
    #print(p01)
    return p01

def comp_p02(country, cur):
    '''
    Computes the probability to pass from  intensity level 0 to
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 2 AND c2.intensity_level = 0 AND c2.year < c1.year) AS subquery;")

    if country.nb_conf==0:
        p02=0
    else:
        p02= cur.fetchall()[0][0] / country.nb_conf
    return p02

def comp_p11(country, cur):
    '''
    Computes the probability to pass from  intensity level 1 to 1
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 1 AND c2.intensity_level = 1 AND c2.year < c1.year) AS subquery;")
    if country.nb_conf==0:
        p11=0
    else:
        p11= cur.fetchall()[0][0]/country.nb_conf
    return p11

def comp_p12(country, cur):
    '''
    Computes the probability to pass from  intensity level 1 to 2
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 2 AND c2.intensity_level = 1 AND c2.year < c1.year) AS subquery;")
    if country.nb_conf==0:
        p12=0
    else:
        p12= cur.fetchall()[0][0]/country.nb_conf
    return p12

def comp_p10(country, cur):
    '''
    Computes the probability to pass from  intensity level 1 to 0
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 0 AND c2.intensity_level = 1 AND c2.year < c1.year) AS subquery;")
    if country.nb_conf==0:
        p10=0
    else:
        p10= cur.fetchall()[0][0]/country.nb_conf
    return p10

def comp_p22(country, cur):
    '''
    Computes the probability to pass from  intensity level 2 to 2
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 2 AND c2.intensity_level = 2 AND c2.year < c1.year) AS subquery;")

    if country.nb_conf==0:
        p22=0
    else:
        p22= cur.fetchall()[0][0]/country.nb_conf
    return p22

def comp_p21(country, cur):
    '''
    Computes the probability to pass from  intensity level 2 to 1
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 1 AND c2.intensity_level = 2 AND c2.year < c1.year) AS subquery;")

    if country.nb_conf==0:
        p21=0
    else:
        p21=cur.fetchall()[0][0]/country.nb_conf
    return p21

def comp_p20(country, cur):
    '''
    Computes the probability to pass from  intensity level 2 to0
    :param country:
    :param data:
    :return:
    '''

    cur.execute(
        "SELECT COUNT(*) AS nombre_de_conflits FROM (SELECT c1.conflict_id FROM Conflicts c1 INNER JOIN Conflicts c2 ON c1.conflict_id = c2.conflict_id WHERE (c1.side_a = '" + country.name + "' AND c2.side_a = '" + country.name + "') AND c1.intensity_level = 0 AND c2.intensity_level = 2 AND c2.year < c1.year) AS subquery;")
    if country.nb_conf==0:
        p20=0
    else:
        p20= cur.fetchall()[0][0]/country.nb_conf
    return p20

def compA(country, cur):
    '''
    Computes the transition matrix for a given country
    :param country: country at stake
    :param cur: sqlite cursor
    :return:
    '''

    country.A[0][0]=comp_p00(country, cur)
    country.A[0][1]=comp_p01(country, cur)
    country.A[0][2]=comp_p02(country, cur)
    country.A[1][0]=comp_p10(country, cur)
    country.A[1][1]=comp_p11(country, cur)
    country.A[1][2]=comp_p12(country, cur)
    country.A[2][0]=comp_p20(country, cur)
    country.A[2][1]=comp_p21(country, cur)
    country.A[2][2]=comp_p22(country, cur)



if __name__ == "__main__":

    # TODO: Regler problème de redondance dans le calcul des probabilités

    # TODO: simuler l'effet de voisinage par observation dynamique
    # TODO: ajouter les autres variables d'observation et matrice d'observation

    # TODO: simulation du HMM
    # TODO: sequence d'observation, solution DP, HMM
    # TODO: baum welsh retrouver le modèle

    excel_path="UcdpPrioConflict_v23_1.xlsx"
    data=extract_data(excel_path)
    #print(data)
    list_countries=create_countries(data,3) # creating countries
    cur=data_to_sql(data)   # sqlite cursor

    # computing total number of conflicts for every country
    for country in list_countries:
        comp_nbConf(country, cur)   # computing total number of conflicts for every country
        compA(country, cur) # computing transition matrix
        print(country.toString())