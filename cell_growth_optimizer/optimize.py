import numpy as np
import pandas as pd
import re
import libsbml
from . import processSBML
import importlib
import sys
from types import FunctionType
from pathlib import Path
importlib.reload(processSBML)


def optimizer(model, method="optimum-growth", initlist=[]):
    
    model = Path(model)
    
    
    def classify_data():
        # classify the species and set initial values
        for idx, row in df_species.iterrows():  # set initial species values
            if row['sbo'] == 'SBO:0000247':
                model_params['metabolites'].append(idx)
            elif row['sbo'] == 'SBO:0000252':
                model_params['metabolic_proteins'].append(idx)
            elif row['sbo'] == 'SBO:0000250':
                model_params['ribosomes'].append(idx)
            if row['hasOnlySubsUnits']:
                if row['isInitialConc']:
                    model_params[idx] = row['initialConc'] * \
                        model_params[row['compartment']]
                else:
                    model_params[idx] = row['initialAmount']
            else:
                if row['isInitialConc']:
                    model_params[idx] = row['initialConc']
                else:
                    model_params[idx] = row['initialAmount'] / \
                        model_params[row['compartment']]
        model_params['enzymes'] = model_params['metabolic_proteins'] + \
            model_params['ribosomes']

        # dynamically generate rate functions for each reaction using variable scope of model_params, classify the reactions
        for idx, row in df_reactions.iterrows():
            localParams = ' '
            for lp in row['localParams']:
                # add any local parameters
                localParams += '{} = {}; '.format(lp[0], lp[-1])
            code = compile('def flux(): ' + localParams +' return ' + row['kineticLaw'], "<string>", "exec")
            model_params[idx] = FunctionType(code.co_consts[0], model_params, idx)
            if not row['substrates'] and df_species['sbo'][row['products'][0][0]] == 'SBO:0000247':
                model_params['meta_syn_reactions'].append(idx)
                reactions2modifiers[idx] = row['modifiers'][0][0]
            elif not row['products']:
                model_params['dil_reactions'].append(idx)
            elif df_species['sbo'][row['products'][0][0]] == 'SBO:0000252' or df_species['sbo'][row['products'][0][0]] == 'SBO:0000250':
                products2reactions[row['products'][0][0]] = idx
                reactions2products[idx] = row['products'][0][0]
        for enzyme in model_params['enzymes']:
            model_params['prot_syn_reactions'].append(products2reactions[enzyme])

        # dynamically generate ODE functions for each species. assessing globals dictionary
        N = df_N.values
        ode = []
        for i in range(N.shape[0]):
            sum_odes = ''
            for j in range(N.shape[1]):
                if N[i, j]:
                    if len(sum_odes) > 0:
                        sum_odes += ' + '
                    sum_odes += 'rate_functions[{}]() * ({})'.format(j,N[i, j])
            code = compile('def ode(): return ' + sum_odes, "<string>", "exec")
            ode.append(FunctionType(code.co_consts[0], globals(), 'ode'))
        df_species['function'] = ode

        # dynamically create lambda functions defined under ListOfFunctions (e.g.: lambda(Y, cell, k1, k1 * Y / cell))
        func = []
        for id, row in df_functions.iterrows():
            m = re.search(r'lambda\((.*)\)', row['formula'].strip())
            if m:
                items = m.group(1).split(',')
            params = ''
            # first n-1 parameters are arguments, last parameter defines the function
            for i in np.arange(len(items)-1):
                if i > 0:
                    params += ', '
                params += items[i].strip()
            code = compile('def ' + id + '(' + params + '): return ' +items[-1].strip(), "<string>", "exec")
            func.append(FunctionType(code.co_consts[0], model_params, id))
        df_functions['function'] = func

        # dynamically generate rule functions
        func = []
        for idx, row in df_rules.iterrows():
            code = compile('def flux(): return ' + row['formula'], "<string>", "exec")
            func.append(FunctionType(code.co_consts[0], model_params, idx))
            exec(compile('def '+idx+'(): return ' +row['formula'], "<string>", "exec"))
        df_rules['function'] = func

        # dynamically generate initial assignment functions
        func = []
        for idx, row in df_initAss.iterrows():
            code = compile('def flux(): return ' + row['formula'], "<string>", "exec")
            func.append(FunctionType(code.co_consts[0], model_params, idx))
        df_initAss['function'] = func

        # generate model parameters in model_params
        for idx, row in df_parameters.iterrows():
            if idx.startswith('alpha'):  # setting all alphas to 1
                model_params[idx] = 1
            else:
                model_params[idx] = row['value']

        for idx, row in df_compartments.iterrows():
            if idx.startswith('cell'):
                model_params[idx] = 1
            else:
                model_params[idx] = row['size']

        for idx, row in df_initAss.iterrows():  # 1st run: overwrite compartment sizes, where applicable
            # usually has no infoeternal word vs
            model_params[idx] = row['function']()

        for idx, row in df_initAss.iterrows():  # 2nd run: overwrite initial quantities, where applicable
            model_params[idx] = row['function']()

        for idx, row in df_functions.iterrows():
            model_params[idx] = row['function']
        return

        # dynamically create the ODEs for the metabolites

    dict_sbml = processSBML.sbml2dfs(model)

    if 'error' in dict_sbml:
        print(dict_sbml['error'])
        sys.exit("Error: file invalid or file processSBML failed")
    else:
        df_reactions = dict_sbml["reactions"]
        df_species = dict_sbml["species"]
        s_modelInfo = dict_sbml["modelInfo"]
        df_parameters = dict_sbml["parameters"]
        df_unitdefs = dict_sbml["unitDefinitions"]
        df_compartments = dict_sbml["compartments"]
        df_objectives = dict_sbml["objectives"]
        df_rules = dict_sbml["rules"]
        df_initAss = dict_sbml["initialAssignment"]
        df_functions = dict_sbml["functions"]
        df_N = dict_sbml["stoichMatrix"]

# determination of states of balanced growth (using metabolic protein fractions)

    def muphi(c_s, phi_m, model):      # Eq. 21: µ^-1 = 1e^T diag(G^-1) diag(le) c_e/c_r
        phi_e = np.append(phi_m ,1.0)
        return 1.0/sum(1.0/Gr(c_s) * phi_e)   # checked and correct

    def muphigrad(c_s, phi_m, model): # d µ/ d c_A = -µ**2 * (l_T phi_T d(1/G_T)/d c_A) + l_V phi_V d(1/G_V)/dc_A)) ..)
        phi_e = np.append(phi_m ,1.0)
        return -muphi(c_s, phi_m, model)**2 * phi_e.dot(Grinvjac(c_s))

    def hphi(c_s, phi_m, model): # Eq. 24 h_s = S diag(phi_m) Fm - µ P c_e
        return S.dot(Fm(c_s) * phi_m) - muphi(c_s, phi_m, model)*model['caa_pr']

    def Jphi(c_s, phi_m, model):  # Eq. 25 J_ss = S diag(phi_m) Fmjac - P c_e x mugrad - no metabolite dilution
        return S.dot(Fmjac(c_s) * phi_m.reshape(-1,1)) \
            - muphigrad(c_s, phi_m, model) * model['caa_pr'].reshape(-1,1)

    def find_nroots_phi(func, x0, fprime, phi_m, model, maxiter=50, atol=1.48e-08, rtol=0.0):
        lb = 0.0; ub = 1000.0  # upper and lower bounds for metabolite protein fractions 
        model['caa_pr'] = -P.dot(np.r_[phi_m,1]) # metabolite concentration included in proteins
        x = x0
        for i in range(maxiter):
            dx     = np.linalg.inv(fprime(x, phi_m, model)).dot(func(x, phi_m, model))        
            violate     = (x - dx)  < lb   # identify xi, that after update would become negative
            dx[violate] = 0.9*x[violate]   #    in such use a new delta xi that will reduce xi by 90% 
            violate     = (x - dx)  > ub   # identfy xi, that after update would exceed upper bound
            dx[violate] = 0.9*(x[violate] - ub) #    in such case new delta xi will reduce distance to ub by 90%       
            x = x - dx
            if(np.linalg.norm(dx)   <  atol): break                         # break when using absolute tolerance
            if(np.linalg.norm(dx)/np.linalg.norm(x) < rtol): break         # break when using relative tolerance
        i +=1
        converged = True if i < maxiter else False
        return x, {'converged' : converged, 'iterations':i , 'result': x}

    # determination of states of balanced growth (using protein concentrations)

    def mu(c_s, c_e, model):      # Eq. 21: µ^-1 = 1e^T diag(G^-1) diag(le) c_e/c_r
        return 1/sum(1/Gr(c_s) * c_e/c_e[-1])   # checked and correct

    def mugrad(c_s, c_e, model): # d µ/ d c_A = -µ**2 * (l_T phi_T d(1/G_T)/d c_A) + l_V phi_V d(1/G_V)/dc_A)) ..)
        return -mu(c_s, c_e, model)**2.0 * (c_e/c_e[-1]).dot(Grinvjac(c_s))

    def hcpr(c_s, c_e, model, m_dilution): # Eq. 24 h_s = S diag(c_s) Fm - µ ( P c_e + c_s)
        return S.dot(Fm(c_s) * c_e[:-1]) \
            + mu(c_s, c_e, model) * (P.dot(c_e) - m_dilution*c_s )

    def Jcpr(c_s, c_e, model, m_dilution):  # Eq. 25 J_ss = S diag(c_s) Fmjac - P c_e x mugrad - µ diag(1)
        return S.dot(Fmjac(c_s) * c_e[:-1].reshape(-1,1)) \
            + mugrad(c_s, c_e, model) * (P.dot(c_e) - m_dilution*c_s).reshape(-1,1) \
            - mu(c_s, c_e, model) * np.eye(len(c_s))
    
    def find_nroots_cpr(func, x0, fprime, c_e, model, m_dilution=0.0, maxiter=50, atol=1.48e-08, rtol=0.0):
        lb = 0.0; ub = 1000.0  # upper and lower bounds for concentrations (could also be a parameter)
        x = x0
        for i in range(maxiter):
            dx     = np.linalg.inv(fprime(x, c_e, model, m_dilution)).dot(func(x, c_e, model, m_dilution))
            violate     = (x - dx)  < lb   # identify xi, that after update would become negative
            dx[violate] = 0.9*x[violate]   #    in such use a new delta xi that will reduce xi by 90% 
            violate     = (x - dx)  > ub   # identfy xi, that after update would exceed upper bound
            dx[violate] = 0.9*(x[violate] - ub) #    in such case new delta xi will reduce distance to ub by 90%
            x -= dx
            if(np.linalg.norm(dx)   <  atol): break                         # stop, based on absolute tolerance
            if(np.linalg.norm(dx)/np.linalg.norm(x) < rtol): break         # stop, based on relative tolerance
        i +=1
        converged = True if i < maxiter else False
        return x, {'converged' : converged, 'iterations':i , 'result': x}

    # optimization for balanced growth using gradient update method
    def tau(c_s, phi_m, model):      # Eq. 59: tau = 1e^T diag(G^-1) diag(l_e) phi_e
        phi_e = np.append(phi_m ,1.0)
        return sum(phi_e/Gr(c_s))    # returns scalar

    def tau_grad_m(c_s, model):             # marginal cost to increase a specific protein (i.e. time to produce 1 unit)
        return 1.0 / Gr(c_s)[:-1]   # returns vector of size m

    def tau_grad_s(c_s, phi_m, model):
        phi_e = np.append(phi_m ,1.0)
        return (phi_e).dot(Grinvjac(c_s))      # returns vector of size s

    def hsm(c_s, phi_m, model): # Eq. 64 h_s = S diag(Fm) phi_m - Q * dial(l_e) * phi_e / tau
        phi_e = np.append(phi_m ,1.0)                                   # returns vector of size s
        return S.dot(Fm(c_s) * phi_m) + P.dot(phi_e)/tau(c_s, phi_m, model)

    def hsm_grad_s(c_s, phi_m, model):
        phi_e = np.append(phi_m ,1.0)
        return   S.dot(Fmjac(c_s)*phi_m.reshape(-1,1)) \
            - P.dot(phi_e).reshape(-1,1)*tau_grad_s(c_s, phi_m, model)\
                /tau(c_s, phi_m, model)**2

    def hsm_grad_m(c_s, phi_m, model):
        phi_e = np.append(phi_m ,1.0)
        return    S*Fm(c_s) \
            + P[:,:-1]/tau(c_s, phi_m, model) \
            - P.dot(phi_e/tau(c_s, phi_m, model)**2.0).reshape(-1,1)*tau_grad_m(c_s, model)

    def dtaudphi(c_s, phi_m, model): # Eq. 73: dtau/dphi_m = tau_grad_m - tau_grad_s * inv(hsm_grad_s) * hsm_grad_m
        return tau_grad_m(c_s, d_model) - tau_grad_s(c_s, phi_m, d_model).dot(
                    np.linalg.inv(hsm_grad_s(c_s, phi_m, d_model))).dot(
                    hsm_grad_m(c_s, phi_m, d_model))

    # c_s0: start concentration vector, phi_m0: start metabolic protein ratio vector
    # eta: learning rate
    def opt_growth_rate(c_s0, phi_m0, model, eta=0.0001, maxiter=50, rtol=0.0, atol=1.48e-08):
        lb = 0.0; ub = 100.0  # upper and lower bounds metabolic protein fractions
        x     = phi_m0.copy()
        c_s   = c_s0.copy()
        for i in range(maxiter):
            dx = eta * dtaudphi(c_s, x, d_model)
            violate     = (x - dx)  < lb   # identify xi, that after update would become negative
            dx[violate] = x[violate]       # in such case use a dx to move xi after update to zero
            violate     = (x - dx)  > ub   # identfy xi, that after update would exceed upper bound
            dx[violate] = 0.9*(x[violate] - ub) #  in such case new delta xi will reduce distance to ub by 90%  
            x -= dx
            c_s, r = find_nroots_phi(hsm, c_s, hsm_grad_s, x, d_model, maxiter=100) # update c_s with root finding algorithm for updated phi_m
            if(np.linalg.norm(dx)   <  atol): break                       # stop, based on absolute tolerance
            if(np.linalg.norm(dx)/np.linalg.norm(x) < rtol): break    # stop, based on relative tolerance
        i +=1
        converged = True if i < maxiter else False
        return c_s, x, {'converged' : converged, 'iterations':i , 'opt_c_s': c_s, 'opt_phi_m': x}
    
    # update model_params with the given array using dct for indices
    def update_params(val_array):
        for ind, val in enumerate(val_array):
            model_params[model_params['metabolites'][ind]] = val

    # initialize

    model_params = {'ribosomes': [], 'metabolites': [], 'enzymes': [], 'metabolic_proteins': [], 'meta_syn_reactions': [
    ], 'prot_syn_reactions': [], 'dil_reactions': []}       # variable scope that holds model parameters and functions
    
    products2reactions = dict(); reactions2products = dict(); reactions2modifiers = dict();
    
    classify_data()

    df_S = pd.DataFrame(np.zeros(shape=(len(model_params['metabolites']), len(model_params['meta_syn_reactions']))), 
                        index=model_params['metabolites'], columns=model_params['meta_syn_reactions'])
    df_P = pd.DataFrame(np.zeros(shape=(len(model_params['metabolites']), len(model_params['enzymes']))), 
                        index=model_params['metabolites'], columns=model_params['prot_syn_reactions'])

    for rxn in model_params['meta_syn_reactions']:
        for species, coeff in df_N[rxn].iteritems():
            if species in model_params['metabolites']:
                df_S[rxn][species]=coeff
    df_S=df_S.rename(columns=reactions2modifiers)
    
    for rxn in model_params['prot_syn_reactions']:
        for species, coeff in df_N[rxn].iteritems():
            if species in model_params['metabolites']:
                df_P[rxn][species]=coeff
    df_P=df_P.rename(columns=reactions2products)
    
    dx = 6e-6
    c_s0 = np.array([model_params[metabolite] for metabolite in model_params['metabolites']])
    ribosome = model_params['enzymes'][-1]
    modifier_for_meta_rxn = [df_reactions['modifiers'][rxn][0][0]
                             for rxn in model_params['meta_syn_reactions']]

    def Fm(c_s):
        answer = np.zeros(len(model_params['meta_syn_reactions']))
        update_params(c_s)
        mod_conc = np.array([model_params[x] for x in modifier_for_meta_rxn])
        for ind, rxn in enumerate(model_params['meta_syn_reactions']):
            #print(rxn,ind)
            answer[ind] = model_params[rxn]()
        return answer/mod_conc

    def Fmjac(c_s):
        answer = np.zeros(
            shape=(len(model_params['meta_syn_reactions']), len(c_s)))
        for ind, val in enumerate(c_s):
            c_s_plusdx = np.copy(c_s)
            c_s_plusdx[ind] += dx
            c_s_minusdx = np.copy(c_s)
            c_s_minusdx[ind] -= dx
            answer[:, ind] = Fm(c_s_plusdx)-Fm(c_s_minusdx)
        return answer/2/dx

    def Gr(c_s):
        answer = np.zeros(len(model_params['prot_syn_reactions']))
        update_params(c_s)
        for ind, rxn in enumerate(model_params['prot_syn_reactions']):
            answer[ind] = model_params[rxn]()
        return answer/model_params[ribosome]

    def Grjac(c_s):
        update_params(c_s)
        answer = np.zeros(
            shape=(len(model_params['prot_syn_reactions']), len(c_s)))
        for ind, val in enumerate(c_s):
            c_s_plusdx = np.copy(c_s)
            c_s_plusdx[ind] += dx
            c_s_minusdx = np.copy(c_s)
            c_s_minusdx[ind] -= dx
            answer[:, ind] = Gr(c_s_plusdx, P)-Gr(c_s_minusdx, P)
        return answer/2/dx

    def Grinvjac(c_s):
        update_params(c_s)
        answer = np.zeros(
            shape=(len(model_params['prot_syn_reactions']), len(c_s)))
        for ind, val in enumerate(c_s):
            c_s_plusdx = np.copy(c_s)
            c_s_plusdx[ind] += dx
            c_s_minusdx = np.copy(c_s)
            c_s_minusdx[ind] -= dx
            answer[:, ind] = 1/Gr(c_s_plusdx)-1/Gr(c_s_minusdx)
        return answer/dx/2

    d_model = {}
    
    S = df_S.values
    P = df_P.values

    no_of_metabolic_proteins = len(df_P.columns)-1

    if method == "optimum-growth":
        phi_m = 0.1*np.ones(no_of_metabolic_proteins)
        c_s, phi_m, r = opt_growth_rate(c_s0, phi_m, d_model, eta=0.00001, maxiter=1000)
        print('max growth rate: {:.3f} 1/h'.format(1/tau(c_s, phi_m, d_model)*3600))
        print('phi_m: ', phi_m) 
        print('c_s:', c_s, ', log10: ', np.log10(c_s))
    elif method == "optimize-for-protein-conc":
        if isinstance(initlist, list):
            if len(initlist) == no_of_metabolic_proteins+1:
                if all(isinstance(element, float) or isinstance(element, int) for element in initlist):
                    c_e = np.array(initlist)
                    phi_m = c_e[:-1]/c_e[-1]
                    c_s, r = find_nroots_cpr(hcpr, c_s0, Jcpr, c_e, d_model, maxiter=50, m_dilution=True)
                    mu_result = mu(c_s, c_e/c_e[-1], d_model)
                    print('growth rate: {:.4e} 1/s, {:.4f} 1/h'.format(mu_result, mu_result * 3600))
                    print('phi_m: ', phi_m)
                    print('c_s: ', c_s,'   log10: ', np.log10(c_s))
                else:
                    print("Error: must give a list of enzyme concentrations [1]")
            else:
                print("Error: must give a list of enzyme concentrations [2]")
        else:
            print("Error: must give a list of enzyme concentrations [3]")
    elif method == "optimize-for-protein-ratios":
        if isinstance(initlist, list):
            if len(initlist) == no_of_metabolic_proteins:
                if all(isinstance(element, float) or isinstance(element, int) for element in initlist):
                    phi_m = np.array(initlist)
                    c_s, r = find_nroots_phi(hphi, c_s0, Jphi, phi_m, d_model, maxiter=50)
                    mu_result = muphi(c_s, phi_m, d_model)
                    print('growth rate: {:.4e} 1/s, {:.4f} 1/h'.format(mu_result, mu_result * 3600))
                    print('phi_m: ', phi_m)
                    print('c_s:   ', c_s, '   log10:', np.log10(c_s))
                else:
                    print("Error: must give a list of ratios of all metabolic proteins to ribosomes [1]")
            else:
                print("Error: must give a list of ratios of all metabolic proteins to ribosomes [2]")
        else:
            print("Error: must give a list of ratios of all metabolic proteins to ribosomes [3]")
    else:
        print("Error: method must be optimum-growth / optimize-for-protein / optimize-for-protein-ratios")

