import numpy as np

def isErgodique(A):
    '''
    Checks if the Markov chain is ergodique, if so return pi_etoile,
    the limit to infinity of the transition vector
    :param A:
    :return:
    '''

    # check if ergodique
    n=10000000 # nombre d'itérations

    pi1=np.random.random((1,3)) # vecteur aléatoire
    pi2=np.random.random((1,3))

    pi1=pi1@np.linalg.matrix_power(A,n)
    pi2=pi2@np.linalg.matrix_power(A,n)

    print("Vecteur aléatoire n°1 après"+str(n)+"itérations", pi1)
    print("Vecteur aléatoire n°2 après" + str(n) + "itérations", pi2)


    eigenvalues,eigenvectors = np.linalg.eig(A)
    index = None  # Index of the searched eigenvector
    tol = 1e-10  # Tolerance for considering a value close to 1

    for k in range(len(eigenvalues)):
        if np.isclose(eigenvalues[k], 1.0, atol=tol):
            index = k
            break

    if index is None:
        print("No eigenvalue close to 1 found.")
        return None
    return eigenvectors[:,index]

def simulation(A,N,i_Xn):
    '''
    Réalise une simulation d'une etape d'une chaine de Markov
    :param A: matrice transition
    :param N: nombre d'états
    :param i_Xn: indice de la ligne de l'état courant du système dans A
    :return:
    '''

    nb_voisins=0
    l_jState=[] # liste des indices des etats voisins
    l_iState=[]  # etats ayant survecu a la premiere vague de selection

    # création des blocs
    ligne=A[i_Xn]   # ligne associée à l'état courant
    for j in range(ligne.shape[0]): # pour tous les etats voisins
        if ligne[j]!=0: # si les 2 états sont reliés
            nb_voisins+=1   # nombre de blocs reliés
            l_jState.append(j)
    if nb_voisins//2==0:    # test de parité
        nb_bloc=nb_voisins//2   # nombre de blocs d'états
    else:
        nb_bloc=int(nb_voisins//2)

    # première vague de selection
    for i in range(0,nb_bloc,2):    # pour tous les blocs à construire, par pas de 2
        prob1=ligne[l_jState[i]]    # simule les blocs en prenant leurs probabilités
        prob2=ligne[l_jState[i+1]]
        val=np.random.random()    # tirage aléatoire
        if abs(prob1-val)>abs(prob2-val):   # prob1 a plus de chance de se réaliser
            l_iState.append(l_jState[i]) # on garde l'état du bloc
        else:
            l_iState.append(l_jState[i+1])  # prob2 a plus de chance de se réaliser

    while len(l_iState)>1:
        l_iState=recurComp(l_iState, ligne)

    return l_iState[0]


def recurComp(l_State, ligne):
    '''
    Effectue une comparaison pour une simulation de manière recursive
    :return:
    '''
    for i in range(0,len(l_State),2):    # pour tous les blocs à construire, par pas de 2
        prob1=ligne[l_State[i]]    # simule les blocs en prenant leurs probabilités
        prob2=ligne[l_State[i+1]]
        val=np.random.random    # tirage aléatoire
        if abs(prob1-val)>abs(prob2-val):   # prob1 a plus de chance de se réaliser
            l_State.pop(i+1)    # on supprime l'etat

    return l_State

def newSimulation(A, etat_initial, n):
    '''
    Simulates a process on a Markov chain
    :param A:
    :param etat_initial:
    :param n: nombre d'iteration
    :return:
    '''
    l_states=[etat_initial] # list of all states
    etat_actuel=etat_initial

    for k in range(n-1):
        probabilites_entieres = (A[etat_actuel] * 100).astype(int)
        probabilites_normalisees = probabilites_entieres / np.sum(probabilites_entieres)
        etat_suivant=np.random.choice([0,1,2], p=probabilites_normalisees)
        etat_actuel=etat_suivant
        l_states.append(etat_actuel)

    return l_states


def forwardAlgorithm(Y, A, B, v_pi):
    '''
    Solution par un problème 1
    :param Y:
    :param A:
    :param B:
    :param v_pi:
    :return:
    '''
    N=A.shape[0]
    T=len(Y)
    tab_alpha=np.zeros((N,T))
    # initialisation
    for i in range(N):
        tab_alpha[i][0]=v_pi[i]*B[i][Y[0]]
    # recurrence
    for t in range(1,T):
        for i in range(N):
            tab_alpha[i][t]=np.sum(tab_alpha[:,t-1]*A[:,i])*B[i][Y[t]]

    pylbd=np.sum(tab_alpha[:,-1])

    return pylbd, tab_alpha

def backwardAlgorithm(Y, A, B):
    '''
    Solution pour un problème 1
    :param Y:
    :param A:
    :param B:
    :return:
    '''
    N=A.shape[0]
    T=len(Y)
    tab_beta=np.zeros((N,T))

    # initialisation
    for i in range(N):
        tab_beta[i][-1]=1
    for t in range(T-2,0,-1):
        for i in range(N):
            tab_beta[i][t]=np.sum(A[i,:]*B[:,Y[t+1]]*tab_beta[:,t+1])
    pylbd=np.sum(tab_beta[:,0])

    return pylbd, tab_beta

def baum_welch(Y, N, M, max_iter=100, tol=1e-6):
    '''

    :param Y:
    :param N: number of states
    :param M: number of observations types
    :param max_iter:
    :param tol:
    :return:
    '''
    T = len(Y)

    A = np.random.rand(N, N)
    A /= np.sum(A, axis=1, keepdims=True)
    B = np.random.rand(N, M)
    B /= np.sum(B, axis=1, keepdims=True)
    pi = np.random.rand(N)
    pi /= np.sum(pi)

    for iter in range(max_iter):
        # Etape d'espérance
        pylbd_f, alpha = forwardAlgorithm(Y, A, B, pi)
        pylbd_b, beta = backwardAlgorithm(Y, A, B)
        gamma = alpha * beta / pylbd_f

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    xi[i, j, t] = alpha[i, t] * A[i, j] * B[j, Y[t + 1]] * beta[j, t + 1] / pylbd_f

        # Etape de maximisation
        new_pi = gamma[:, 0]
        new_A = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1, keepdims=True)

        new_B = np.zeros((N, M))
        for k in range(M):
            mask = (Y == k)
            new_B[:, k] = np.sum(gamma[:, mask], axis=1) / np.sum(gamma, axis=1)

        diff_A = np.linalg.norm(new_A - A)
        diff_B = np.linalg.norm(new_B - B)
        diff_pi = np.linalg.norm(new_pi - pi)

        if diff_A < tol and diff_B < tol and diff_pi < tol:
            break

        A = new_A
        B = new_B
        pi = new_pi

    return A, B, pi

# def viterbiAlgorithm(Y, A, B, v_pi):
#     '''
#     Permet de calculer la solution au sens de la programmation dynamique pour un problème 2
#     :param Y:
#     :param A:
#     :param B:
#     :param v_pi:
#     :return: solution optimale X_etoile
#     '''
#
#     N = A.shape[0]  # nombre d'etat
#     T = Y.shape[0]  # nombre d'observations
#     tab_delta=np.zeros((N,T))
#
#     # initialisation
#     for i in range(N):
#         tab_delta[i][0]=v_pi[i]*B[i][Y[0]]
#
#     for t in range(1, T):
#         for i in range(0, N):
#             psi_t=np.argmax(tab_delta[i][t-1]@A)
#             tab_delta[i][t]=np.max(tab_delta[i][t-1]@A[i,:]@B[:,Y[t]])
#
#     X_etoile=np.argmax(tab_delta[])
#
#     return X_etoile

