import numpy as np

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
    T=Y.shape[0]
    tab_alpha=np.zeros((N,T))
    # initialisation
    for i in range(N):
        tab_alpha[i][0]=v_pi[i]*B[i][Y[0]]
    # recurrence
    for t in range(1,T):
        for i in range(N):
            tab_alpha[i][t]=np.sum(tab_alpha[:,t-1]*A[:,i])*B[i][Y[t]]

    pylbd=np.sum(tab_alpha[:,-1])

    return pylbd

def backwardAlgorithm(Y, A, B):
    '''
    Solution pour un problème 1
    :param Y:
    :param A:
    :param B:
    :return:
    '''
    N=A.shape[0]
    T=Y.shape[0]
    tab_beta=np.zeros((N,T))

    # initialisation
    for i in range(N):
        tab_beta[i][-1]=1
    for t in range(T-2,0,-1):
        for i in range(N):
            tab_beta[i][t]=np.sum(A[i,:]*B[:,Y[t+1]]*tab_beta[:,t+1])
    pylbd=np.sum(tab_beta[:,0])

    return pylbd

def viterbiAlgorithm(Y, A, B, v_pi):
    '''
    Permet de calculer la solution au sens de la programmation dynamique pour un problème 2
    :param Y:
    :param A:
    :param B:
    :param v_pi:
    :return: solution optimale X_etoile
    '''

    N = A.shape[0]  # nombre d'etat
    T = Y.shape[0]  # nombre d'observations
    tab_delta=np.zeros((N,T))

    # initialisation
    for i in range(N):
        tab_delta[i][0]=v_pi[i]*B[i][Y[0]]

    for t in range(1, T):
        for i in range(0, N):
            psi_t=np.argmax(tab_delta[i][t-1]@A)
            tab_delta[i][t]=np.max(tab_delta[i][t-1]@A[i,:]@B[:,Y[t]])

    X_etoile=np.argmax(tab_delta[])

    return X_etoile

def limPi(pi, seuil, A):
    '''
    Permet de calculer la limite du vecteur pi.
    :param pi: vecteur transition
    :param seuil: seuil à partir du quel la valeur n'evolue plus
    :param A: matrice de transition
    :return:
    '''

    pi_n=pi
    pi_n1=pi@A
    while np.abs(pi_n1-pi_n)<seuil:
        pi_n=pi_n1
        pi_n1=pi_n@A

    return pi_n1

