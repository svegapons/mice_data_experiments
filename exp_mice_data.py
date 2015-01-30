import os
import numpy as np
import sklearn.metrics.pairwise as skpw
from scipy.io import loadmat
from statsmodels.distributions.empirical_distribution import ECDF
from kernel_two_sample_test import MMD2u, compute_null_distribution
import matplotlib.pyplot as plt
import pdb


def load_structural_data(path_b6, path_btbr):
    """
    Load the structural connectivity matrixes from the .mat file.
    Parameters:
    ----------
    path_b6: string
              Path to the .mat file with the data of B6 class
    path_btbr: string
              Path to the .mat file with the data of BTBR class
    Returns:
    ------
    norm_b6: ndarray
            Array...
    norm_btbr: ndarray
            Array...
    """
    print "Loading structural connectivity matrixes"
    struc_b6 = []
    #Loading the 3D matrix
#    mat = np.log(loadmat(path_b6)['AssocMatrixC57'] + 1)
    mat = loadmat(path_b6)['AssocMatrixC57']

    for i in range(mat.shape[-1]):
        struc_b6.append(mat[:,:,i].reshape(-1))    
    print 'Number of subjects in class B6: %s' %(len(struc_b6))
    struc_b6 = np.array(struc_b6)
    
    struc_btbr = []
    #Loading the 3D matrix
    mat = np.log(loadmat(path_btbr)['AssocMatrixBTBR'] + 1)
    for i in range(mat.shape[-1]):
        struc_btbr.append(mat[:,:,i].reshape(-1))
    print 'Number of subjects in class BTBR: %s' %(len(struc_btbr))
    struc_btbr = np.array(struc_btbr)
    
    struc_X = np.vstack((struc_b6, struc_btbr))

    #Computing the empirical cumulative distribution function using all 
    #non-zero edge weights.
    ecdf = ECDF(struc_X.reshape(-1)[struc_X.reshape(-1) > 0])
    norm_b6 = np.zeros(np.array(struc_b6).shape)
    norm_btbr = np.zeros(np.array(struc_btbr).shape)
    #Representing each edge weight by its distribution value.
    for i in range(len(struc_b6)):
        norm_b6[i] = ecdf(struc_b6[i])
    for i in range(len(struc_btbr)):
        norm_btbr[i] = ecdf(struc_btbr[i])
        
    return norm_b6, norm_btbr
    
    
    
def load_functional_data(path_b6, path_btbr):
    """
    Load the functional connectivity matrixes from the .mat file.
    Parameters:
    ----------
    path_b6: string
              Path to the .mat file with the data of B6 class
    path_btbr: string
              Path to the .mat file with the data of BTBR class
    Returns:
    ------
    norm_b6: ndarray
            Array...
    norm_btbr: ndarray
            Array...             
    """
    print "Loading functional connectivity matrixes"
    func_b6 = []
    #Unfolding and concatenating the data for all subjects
    for f in os.listdir(path_b6):
        mat = loadmat(os.path.join(path_b6, f))['nw_re_arranged']
        func_b6.append(mat.reshape(-1))
    print 'Number of subjects in class B6: %s' %(len(func_b6))
    func_b6 = np.array(func_b6)
    #Removing the edges with negative correlation values
    func_b6 = np.where(func_b6 > 0, func_b6, 0)
    #Using as edge weights the absolute values 
#    func_b6 = np.abs(func_b6)

    #Creating graphs for class 1
    func_btbr = []
    #Unfolding and concatenating the data for all subjects
    for f in os.listdir(path_btbr):
        mat = loadmat(os.path.join(path_btbr, f))['nw_re_arranged']
        func_btbr.append(mat.reshape(-1))
    print 'Number of subjects in class BTBR: %s' %(len(func_btbr))
    func_btbr = np.array(func_btbr)
    #Removing the edges with negative correlation values
    func_btbr = np.where(func_btbr > 0, func_btbr, 0)
    #Using as edge weights the absolute values 
#    func_btbr = np.abs(func_btbr)
    
    func_X = np.vstack((func_b6, func_btbr))

    #Computing the empirical cumulative distribution function using all 
    #non-zero edge weights.
    ecdf = ECDF(func_X.reshape(-1)[func_X.reshape(-1) > 0])
    norm_b6 = np.zeros(func_b6.shape)
    norm_btbr = np.zeros(func_btbr.shape)
    #Representing each edge weight by its distribution value.
    for i in range(len(func_b6)):
        norm_b6[i] = ecdf(func_b6[i])
    for i in range(len(func_btbr)):
        norm_btbr[i] = ecdf(func_btbr[i])
        
    return norm_b6, norm_btbr
    
    
    
def compute_kernel_matrix(struc_b6, struc_btbr, func_b6, func_btbr, kernel='linear', normalized=True, plot=True, **kwds):
    """
    Computes the kernel matrix for all graphs (structural and functional)
    represented in the common space.
    
    Parameters:
    ----------
    struc_b6: array like
    struc_btbr: array like
    func_b6: array like
    func_btbr: array like
    kernel: string
            Kernel measure. The kernels implemented in sklearn are allowed.
            Possible values are ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, 
            ‘linear’, ‘cosine’.
    normalized: boolean
                Whether to normalize the kernel values by
                k_normalized(a,b) = k(a,b)/np.sqrt(k(a,a)*k(b,b))
    **kwds: optional keyword parameters
            Any further parameters are passed directly to the kernel function.
    Returns:
    ------
    k_mat: ndarray
           Kernel matrix
    """
    vects = np.vstack((struc_b6, struc_btbr, func_b6, func_btbr))    
    k_mat = skpw.pairwise_kernels(vects, vects, metric = kernel, **kwds)
    if normalized:
        k_norm = np.zeros(k_mat.shape)
        for i in range(len(k_mat)):
            for j in range(i, len(k_mat)):
                k_norm[i, j] = k_norm[j, i] = k_mat[i, j] / np.sqrt(k_mat[i, i] * k_mat[j, j])   
        k_mat = k_norm
        
    if plot:
        fig = plt.figure()
        iplot = plt.imshow(k_norm)
        iplot.set_cmap('spectral')
        plt.colorbar()
        plt.title('Similarity matrix')
        plt.show() 
    
    return k_mat
    

def compute_mmd_struc_func(k_mat, struc_b6, struc_btbr, func_b6, func_btbr, iterations=100000):
    """
    Computes the mmd values for the structural and functional problems and plot
    them with the null distributions.
    
    Parameters:
    ----------
    k_mat: ndarray
           Kernel matrix
    struc_b6: array like
           Structural vectors for B6 class
    struc_btbr: array like
           Structural vectors for BTBR class
    func_b6: array like
           Functional vectors for B6 class
    func_btbr: array like
           Functional vectors for BTBR class
    """
    #Computing the number of samples belonging to structural data in order
    #to split the kernel matrix.
    l_struc = len(struc_b6) + len(struc_btbr)
    
    #Computing MMD values
    struc_mmd = MMD2u(k_mat[:l_struc][:,:l_struc], len(struc_b6), len(struc_btbr))
    func_mmd = MMD2u(k_mat[l_struc:][:,l_struc:], len(func_b6), len(func_btbr))
    print "struc_mmd = %s, func_mmd = %s" %(struc_mmd, func_mmd) 
    
    #Computing the null-distribution
    mmd2u_null_all = compute_null_distribution(k_mat, struc_b6.shape[0]+func_b6.shape[0], struc_btbr.shape[0]+func_btbr.shape[0], iterations, seed=123, verbose=False)
    #Computing the p-value
    struc_p_value = max(1.0/iterations, (mmd2u_null_all > struc_mmd).sum() / float(iterations))
    print("struc_p-value ~= %s \t (resolution : %s)" % (struc_p_value, 1.0/iterations))
    func_p_value = max(1.0/iterations, (mmd2u_null_all > func_mmd).sum() / float(iterations))
    print("func_p-value ~= %s \t (resolution : %s)" % (func_p_value, 1.0/iterations))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    prob, bins, patches = plt.hist(mmd2u_null_all, bins=50, normed=True)
    ax.plot(struc_mmd, prob.max()/30, 'w*', markersize=15, markeredgecolor='k', markeredgewidth=2, label="$Structural MMD^2_u = %s$" % struc_mmd)
    ax.plot(func_mmd, prob.max()/30, 'w^', markersize=15, markeredgecolor='k', markeredgewidth=2, label="$Functional MMD^2_u = %s$" % func_mmd)
    plt.xlabel('$MMD^2_u$')
    plt.ylabel('$p(MMD^2_u)$')
    plt.title('$MMD^2_u$: null-distribution and observed values')
    
    ax.annotate('p-value: %s' %(struc_p_value), xy=(float(struc_mmd), 4.),  xycoords='data',
                    xytext=(-105, 30), textcoords='offset points',
                    bbox=dict(boxstyle="round", fc="1."),
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                    )
                    
    ax.annotate('p-value: %s' %(func_p_value), xy=(float(func_mmd), 4.),  xycoords='data',
                xytext=(10, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="1."),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )
                    
    plt.legend(numpoints=1)
    
    
    
def compute_distance_mmd(k_mat, struc_b6, struc_btbr, func_b6, func_btbr, iterations=100000):
    """
    Computes the distance of structural and functional mmd values and compares
    it with the null distribution.
    
    Parameters:
    ----------
    k_mat: ndarray
           Kernel matrix
    struc_b6: array like
           Structural vectors for B6 class
    struc_btbr: array like
           Structural vectors for BTBR class
    func_b6: array like
           Functional vectors for B6 class
    func_btbr: array like
           Functional vectors for BTBR class
    """
    #Computing the number of samples belonging to structural data in order
    #to split the kernel matrix.
    l_struc = len(struc_b6) + len(struc_btbr)
    
    #Computing dist mmd
    struc_mmd = MMD2u(k_mat[:l_struc][:,:l_struc], len(struc_b6), len(struc_btbr))
    func_mmd = MMD2u(k_mat[l_struc:][:,l_struc:], len(func_b6), len(func_btbr))
    dist_mmd = struc_mmd - func_mmd
    
    #Computing null distribution
    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = np.random.permutation(len(k_mat))
        k_perm = k_mat[idx][:,idx]
        s_mmd = MMD2u(k_perm[:l_struc][:,:l_struc], len(struc_b6), len(struc_btbr))
        f_mmd = MMD2u(k_perm[l_struc:][:,l_struc:], len(func_b6), len(func_btbr))
        mmd2u_null[i] = s_mmd - f_mmd
    
    #Computing p-value
    dist_p_value = max(1.0/iterations, (mmd2u_null > dist_mmd).sum() / float(iterations))
    print "Dist p-value ~= %s \t (resolution : %s)" % (dist_p_value, 1.0/iterations)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    prob, bins, patches = plt.hist(mmd2u_null, bins=50, normed=True)
    ax.plot(dist_mmd, prob.max()/30, 'w*', markersize=15, markeredgecolor='k', markeredgewidth=2, label="$Dist_MMD^2_u = %s$" % dist_mmd)

    plt.xlabel('$MMD^2_u$')
    plt.ylabel('$p(MMD^2_u)$')
    plt.legend(numpoints=1)
    
    ax.annotate('p-value: %s' %(dist_p_value), xy=(float(dist_mmd), 1.),  xycoords='data',
            xytext=(10, 30), textcoords='offset points',
            bbox=dict(boxstyle="round", fc="1."),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=10"),
            )        
   
        
if __name__ == "__main__":
                
    path_struc_b6 = "./MatriciRoiLaterali/MatriciB6"
    path_struc_btbr = "./MatriciRoiLaterali/MatriciBTBR"
    path_func_b6 = "./interhemispherical_correlation_matrices_50_regions-1/WT_inter_hemispherical_correlation_matrices"
    path_func_btbr = "./interhemispherical_correlation_matrices_50_regions-1/BTBR_inter_hemispherical_correlation_matrices"

    #Loading structural data
    struc_b6, struc_btbr = load_structural_data(path_struc_b6, path_struc_btbr)    
    #Loading functional data
    func_b6, func_btbr = load_functional_data(path_func_b6, path_func_btbr)
    #Computing the kernel matrix
    k_mat = compute_kernel_matrix(struc_b6, struc_btbr, func_b6, func_btbr, kernel='linear', normalized=True, plot=True)
    
    iters = 10000
    compute_mmd_struc_func(k_mat, struc_b6, struc_btbr, func_b6, func_btbr, iters)   
        
    compute_distance_mmd(k_mat, struc_b6, struc_btbr, func_b6, func_btbr, iters)   
    
    
    
    
    
    
    