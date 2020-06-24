# processSBML.py

# helper routines to process SBML
# convert SBML to pandas DataFrame
# extract ODE function from SBML


# Reads in SBML documents (up to SBML Level 3 Relase 2, including FBC and GROUPS Packages)
# returns information in python dict with pandas DataFrame
# - returns MIRIAM annotations
# - returns functions in string representation
# - converts mathml operators/functions to python operators and numpy functions where possible
# - extracts stoichiometric matrix (could be returned in sparse presentation)
# - basic error handling, file not found, XMLerrors
# - unit handling improved: add built in units to UnitDefinition, add units to compartments
# - error correcion: base unit 'mole' accepted as substance Unit for particleFactor calculation (26.03.2020)

## Peter Schubert, HHU-Duesseldorf, Computational Cell Biology. December 2019
# 2019-12-12

import numpy as np
import pandas as pd
import os
import os.path
import sys
import string
import re
import libsbml
from types import FunctionType


# mapping table from mathml to python/numpy functions
# some functions cannot be mapped directly, if required, define functions in python
# no mapping for mathml: implies(), root(), quotient(), piecewise(), arcsec(), arccsc(), arccot(), 
#    arcsech(), arccsch(), arccoth()
map_mathml2numpy = ( 
# arithmetic operators
    ('abs' , 'np.absolute'), ('exp' , 'np.exp'), ('sqrt' , 'np.sqrt'), ('sqr' , 'np.square'),
    ('ln' , 'np.log'), ('log10' , 'np.log10'), ('floor' , 'np.floor'), ('ceil' , 'np.ceil'),     
    ('factorial' , 'np.math.factorial'), ('rem' , 'np.fmod'), 
# relational operators
    ('eq' , 'np.equal'), ('neq' , 'np.not_equal'), ('gt' , 'np.greater'), 
    ('lt' , 'np.less'), ('geq' , 'np.greater_equal'), ('leq' , 'np.less_equal'),
# logical operators
    ('and' , 'np.logical_and'), ('or' , 'np.logical_or'), ('xor' , 'np.logical_xor'), ('not' , 'np.logical_not'), 
    ('and' , 'np.logical_and'), ('or' , 'np.logical_or'), ('xor' , 'np.logical_xor'), ('not' , 'np.logical_not'), 
# trigonometric operators
    ('sin' , 'np.sin'), ('cos' , 'np.cos'), ('tan' , 'np.tan'), ('sec' , '1/np.cos'), ('csc' , '1/np.sin'),
    ('cot' , '1/np.tan'), ('sinh' , 'np.sinh'), ('cosh' , 'np.cosh'), ('tanh' , 'np.tanh'), ('sech' , '1/np.cosh'),
    ('csch' , '1/np.sinh'), ('coth'  , '1/np.tanh'), ('asin' , 'np.arcsin'), ('acos' , 'np.arccos'),
    ('atan' , 'np.arctan'), ('arcsinh' , 'np.arcsinh'), ('arccosh' , 'np.arccosh'), ('arctanh' , 'np.arctanh'),
)


############################
###### function for sbml2dfs

# no support of nested qualifiers yet
def getMiriamAnnotation(element):
    metaid  = element.getMetaId()
    cvterms = element.getCVTerms()           
    annotation = []
    for cv in cvterms:
        qual_type = cv.getQualifierType(); sub_type = ""
        if qual_type == libsbml.BIOLOGICAL_QUALIFIER:
                qual_type = "Biology"
                sub_type  = libsbml.BiolQualifierType_toString(cv.getBiologicalQualifierType())
        if qual_type == libsbml.MODEL_QUALIFIER:
                qual_type = "Model"
                sub_type  = libsbml.ModelQualifierType_toString(cv.getModelQualifierType())
        for r_idx in range(0, cv.getNumResources()):
            rsrc = cv.getResourceURI(r_idx)
            annotation.append([metaid, qual_type, sub_type, 
                               os.path.basename(os.path.dirname(rsrc)), os.path.basename(rsrc)])
    return np.array(annotation)

def getSpecies(model):
    ss          = []
    default_unit = model.getSubstanceUnits() if model.isSetSubstanceUnits() else 'substance'

    for s in model.getListOfSpecies():         
        subst_units  = s.getSubstanceUnits() if s.isSetSubstanceUnits() else default_unit
        chem_Formula = s.getPlugin("fbc").getChemicalFormula() if s.isPackageEnabled("fbc") else ""
        charge       = s.getPlugin("fbc").getCharge()          if s.isPackageEnabled("fbc") else ""    

        annotations = getMiriamAnnotation(s)
        if len(annotations) > 0: annotations = annotations[:,-2:]
        bigg_name = ""; metanetx_ids = []; kegg_ids = []; chebi_ids = []; other_ids= []
        for res, id in annotations:
            if    re.match("bigg", res):     bigg_name = id
            elif  re.match("metanetx", res): metanetx_ids.extend([id])
            elif  re.match("kegg", res):     kegg_ids.extend([id])
            elif  re.match("chebi", res):    chebi_ids.extend([re.sub("CHEBI:","",id)])
            else: other_ids.extend([[res,id]])            
            
        ss.append([s.getIdAttribute(), s.getMetaId(), s.getName(), s.getCompartment(), s.getConstant(), 
                   s.getBoundaryCondition(), s.getHasOnlySubstanceUnits(), s.isSetInitialConcentration(), 
                   s.getInitialConcentration(), s.getInitialAmount(), subst_units, 
                   s.getSBOTermID(), chem_Formula, charge, 
                   bigg_name, metanetx_ids, kegg_ids, chebi_ids, other_ids ])        
    cols = ["id", "metaid", "name", "compartment", "isConst", "isBoundCond", "hasOnlySubsUnits",
            "isInitialConc", "initialConc", "initialAmount", "subsUnits", "sbo", "chemFormula", "charge", 
            "biggName", "metanetxIds", "keggIds", "chebiIds", "otherIds"]    
    return pd.DataFrame(ss, columns=cols).set_index(keys='id')


def getReactions(model):
    rs          = []
    for r in model.getListOfReactions():
        # collect list of substrates, products, modifiers with stoichiometry for this reaction
        substrates = []; products   = []; modifiers  = []
        for i in range(0, r.getNumReactants()): 
            species_ref = r.getReactant(i)
            substrates.append([species_ref.getSpecies(), species_ref.getStoichiometry(), species_ref.getConstant()])
        for i in range(0, r.getNumProducts()):
            species_ref = r.getProduct(i)
            products.append([species_ref.getSpecies(), species_ref.getStoichiometry(), species_ref.getConstant()])        
        for i in range(0, r.getNumModifiers()):
            species_ref = r.getModifier(i)
            modifiers.append([species_ref.getSpecies()])            
        
        mformula = ""; pformula = ""; localParams=[]
        if r.isSetKineticLaw():
            if r.getKineticLaw().isSetMath():
                mformula = libsbml.formulaToL3String(r.getKineticLaw().getMath())
                pformula = mathml2numpy(mformula)
            if r.getLevel() <= 2:
                df_localParams = getParameters(r.getKineticLaw().getListOfParameters()).drop(labels=["constant"],axis=1)
            else:
                df_localParams = getParameters(r.getKineticLaw().getListOfLocalParameters()).drop(labels=["constant"],axis=1)
            localParams = df_localParams.reset_index().values.tolist()
        
        lower_fluxBound = ""; upper_fluxBound = ""
        gps = []; gps_str =""; is_complex = False; is_isoenzyme = False; 
        if r.isPackageEnabled("fbc"):
            lower_fluxBound = r.getPlugin("fbc").getLowerFluxBound()
            upper_fluxBound = r.getPlugin("fbc").getUpperFluxBound()
            gpa             = r.getPlugin("fbc").getGeneProductAssociation()
            if gpa:
                fbc_ass = gpa.getAssociation()
                if fbc_ass.isGeneProductRef():       # case of just a single gene in GPA
                    gps     = [fbc_ass.getGeneProduct()]
                    gps_str = fbc_ass.getGeneProduct()
                else:                                # case of multiple genes in GPA
                    gps, gps_str, is_isoenzyme, is_complex = parse_gpa(fbc_ass)  
                    
        annotations = getMiriamAnnotation(r)
        if len(annotations) > 0: annotations = annotations[:,-2:]
        bigg_name = ""; ec_codes = []; metanetx_ids = []; kegg_ids = []; other_ids= []
        for res, id in annotations:
            if    re.match("bigg", res):     bigg_name = id
            elif  re.match("ec-code", res):  ec_codes.extend([id])
            elif  re.match("metanetx", res): metanetx_ids.extend([id])
            elif  re.match("kegg", res):     kegg_ids.extend([id])
            else: other_ids.extend([[res,id]])            
                
        rs.append([r.getIdAttribute(), r.getMetaId(), r.getName(), r.getReversible(), r.getFast(), 
                   lower_fluxBound, upper_fluxBound, r.getSBOTermID(), substrates, products, modifiers, 
                   list(set(gps)), gps_str, is_complex, is_isoenzyme, pformula, mformula, localParams,
                   bigg_name, ec_codes, metanetx_ids, kegg_ids, other_ids])
    cols = ["id", "metaid", "name", "isReversible", "isFast", "lowerFluxBound", "upperFluxBound", "sbo", 
            "substrates", "products", "modifiers", "genes", "gpa", "isComplex", "isIsoenzyme",
            "kineticLaw","kineticLaw_mathml","localParams", "biggName", "ecNumbers", "metanetxIds", 
            "keggIds", "otherIds"]    
    return pd.DataFrame(rs, columns=cols).set_index("id")

def getParameters(lps):
    param = []
    for p in lps:
        param.append([p.getIdAttribute(), p.getName(), p.getConstant(), p.getUnits(), p.getValue()])            
    return pd.DataFrame(param, columns=["id","name", "constant", "units", "value"]).set_index(keys="id")

def getUnitDefinitions(model):
    unit_def = []; unit_ids=set()
    for ud in model.getListOfUnitDefinitions():
        id    = ud.getIdAttribute()
        unit_ids.add(id)
        units = []
        for u in ud.getListOfUnits():
            units.append([libsbml.UnitKind_toString(u.getKind()), u.getMultiplier(), u.getScale(), u.getExponent()])
        unit_def.append([id, ud.getName(), units])     
    return pd.DataFrame(unit_def, columns=["id","name", "unitsKMSE"]).set_index(keys='id')

def getCompartments(model):
    comp = []
    for c in model.getListOfCompartments():
        comp.append([c.getIdAttribute(), c.getName(), c.getConstant(), c.getUnits(),c.getSize(), 
                     c.isSetSpatialDimensions(), c.getSpatialDimensions()])   
        cols = ["id","name", "constant", "units", "size","isSpaceDim", "spaceDim"]
    return pd.DataFrame(comp, columns=cols).set_index(keys="id")

def getModelAnn(model):
    cols         = ["metaid", 'qualType', 'subType', "resource", "Id"]
    return pd.DataFrame(getMiriamAnnotation(model), columns=cols).set_index(keys="metaid")

def getModelInfo(model):
    plugins=[]
    for idx in range(model.getNumPlugins()):
        if model.getPlugin(0).getPackageName() == 'layout':
            if model.getPlugin(0).getNumLayouts() == 0: continue
        plugins.append(model.getPlugin(idx).getPackageName())
    model_summary =[model.getId(), model.getName(), "[L{}V{}]".format(model.getLevel(), model.getVersion()), plugins,
                    model.getNumReactions(), model.getNumSpecies(), model.getNumCompartments(), 
                    model.getNumUnitDefinitions(), model.getNumParameters(), model.getNumInitialAssignments(), 
                    model.getNumFunctionDefinitions(), model.getNumRules(), model.getNumEvents(),
                    model.getNumConstraints(),  model.getNumSpeciesTypes(), model.getNumCompartmentTypes()]
    cols = ["id", "name", "level", "plugins", "reactions", "species", "compartments", "units", "parameters", "initAss", 
            "functions", "rules", "events", "constraints",  "specTypes", "compTypes"]
    return pd.Series(model_summary, index = cols)

def getStoichMatrix(df_species, df_reactions, sparse = False):
    df_N = pd.DataFrame(np.zeros((len(df_species),len(df_reactions))), index=df_species.index, columns=df_reactions.index)
    for idx, reaction in df_reactions.iterrows():
        for substrate in reaction["substrates"]: df_N.at[substrate[0],idx] -= substrate[1]
        for product in reaction["products"]:     df_N.at[product[0],idx]   += product[1] 
    if sparse:  return df_N.astype(pd.SparseDtype("float", 0.0))
    else:       return df_N

    
###########################################
### parameters related to kinetic modelling

def mathml2numpy(mformula):
    pformula = ' ' + mformula  # prefix temporarily a space character to support processing
    pformula = pformula.replace('^', '**')               # treat '^' converstion separately
    pformula = pformula.replace(' && ', ' & ')               # treat '^' converstion separately
    pformula = pformula.replace(' || ', ' | ')               # treat '^' converstion separately
    for mathmlFunction, numpyFunction in map_mathml2numpy:
        if re.search(r'(?<=\W)' + mathmlFunction + '\(', pformula):   #check it is actually the specific function
            pformula = re.sub(r'(?<=\W)' + mathmlFunction + '\(', numpyFunction + '(' , pformula)
    return pformula.strip()

def getFunctions(model):
    functions = []
    for fd in model.getListOfFunctionDefinitions():
        mformula = ""; pformula = ""        
        if fd.isSetMath():
            mformula = libsbml.formulaToL3String(fd.getMath())
            pformula = mathml2numpy(mformula)
        functions.append([fd.getId(), fd.getMetaId(), fd.getName(), pformula, mformula])
    return pd.DataFrame(functions, columns = ["id", "metaId", "name", "formula", "formula_mathml"]).set_index(keys='id')

def getRules(model):
    rules = []
    for r in model.getListOfRules():
        mformula = ""; pformula = ""        
        if r.isSetMath():
            mformula = libsbml.formulaToL3String(r.getMath())
            pformula = mathml2numpy(mformula)
        rules.append([r.getVariable(),libsbml.SBMLTypeCode_toString(r.getTypeCode(), r.getPackageName()),
                      pformula, mformula])
    return pd.DataFrame(rules, columns = ["variable", "ruleType", "formula", "formula_mathml"]).set_index(keys='variable')

def getInitAss(model):
    init_ass = []
    for ia in model.getListOfInitialAssignments():
        mformula = ""; pformula = ""        
        if ia.isSetMath():
            mformula = libsbml.formulaToL3String(ia.getMath())
            pformula = mathml2numpy(mformula)
        init_ass.append([ia.getSymbol(), pformula, mformula])
    return pd.DataFrame(init_ass, columns = ["symbol", "formula", "formula_mathml"]).set_index(keys='symbol')

def getEvents(model):
    events = []
    for e in model.getListOfEvents():
        trig_initValue = False; trig_persitent = False; 
        trig_mformula = ""; trig_pformula = ""
        if e.isSetTrigger():
            et = e.getTrigger()
            trig_initValue = et.getInitialValue()
            trig_persitent = et.getPersistent()
            if et.isSetMath(): 
                trig_mformula = libsbml.formulaToL3String(et.getMath())
                trig_pformula = mathml2numpy(trig_mformula)
        prio_mformula = ""; prio_pformula = ""
        if e.isSetPriority():
            ep = e.getPriority()
            if ep.isSetMath(): 
                prio_mformula = libsbml.formulaToL3String(ep.getMath())
                prio_pformula = mathml2numpy(prio_mformula)
        delay_mformula = ""; delay_pformula = ""
        if e.isSetDelay():
            ed = e.getDelay()
            if ed.isSetMath(): 
                delay_mformula = libsbml.formulaToL3String(ed.getMath())
                delay_pformula = mathml2numpy(delay_mformula)
        eas = []
        for ea in e.getListOfEventAssignments():
            eventAssign_mformula = ""; eventAssign_pformula = ""
            if ea.isSetMath(): 
                eventAssign_mformula = libsbml.formulaToL3String(ea.getMath())
                eventAssign_pformula = mathml2numpy(eventAssign_mformula)
            eas.append([ea.getVariable(), eventAssign_pformula, eventAssign_mformula ])
        events.append([e.getName(), e.getUseValuesFromTriggerTime(), trig_initValue, trig_persitent, 
                       trig_pformula, prio_pformula, delay_pformula,
                       trig_mformula, prio_mformula, delay_mformula, eas ])
    cols = ["name", 'useValFromTrigTime', 'trigIVal', 'trigPers', 
            'trigFormula', 'prioFormula', "delayFormula", 
            'trigFormula_mathml', 'prioFormula_mathml', "delayFormula_mathml",'assignments']
    return pd.DataFrame(events, columns=cols).set_index(keys='name')

def getConstaints(model):
    constraints = []
    for c in model.getListOfConstraints():
        mformula = ""; pformula = "" ; message=""        
        if c.isSetMath():    
            mformula = libsbml.formulaToL3String(c.getMath())
            pformula = mathml2numpy(mformula)
        if c.isSetMessage(): 
            message = c.getMessage().getChild("p").getChild(0).toString().strip()            
        constraints.append([pformula, mformula, message])
    return pd.DataFrame(constraints, columns = ["formula", "formula_mathml", "message"])


#########################################################
### paramters related to FBA, i.e. FBC Package parameters

def parse_gpa(fbc_ass):
    is_complex   = fbc_ass.isFbcAnd()
    is_isoenzyme = fbc_ass.isFbcOr()
    if fbc_ass.isFbcOr() : operator = " | "
    if fbc_ass.isFbcAnd(): operator = " & "

    gps = []; gps_str = ""    
    for i in range(0, fbc_ass.getNumAssociations()):
        if i == 0: gps_str += "("
        ass = fbc_ass.getAssociation(i)
        if ass.isGeneProductRef():
            gps_str += ass.getGeneProduct()
            gps.extend([ass.getGeneProduct()])
        else:
            more_gps, more_str, is_iso, is_compl = parse_gpa(ass)
            gps.extend(more_gps)
            gps_str += more_str
            is_complex   = is_complex   or is_compl
            is_isoenzyme = is_isoenzyme or is_iso
        if    i < fbc_ass.getNumAssociations()-1: gps_str += operator
        else: gps_str += ")"
    return gps, gps_str, is_isoenzyme, is_complex

def getGeneProduct(model):
    gps            = []
    fbc_plugin = model.getPlugin("fbc")
    if fbc_plugin:
        annotations = []
        for gp in fbc_plugin.getListOfGeneProducts():
            annotations = getMiriamAnnotation(gp)
            if len(annotations) > 0: annotations = annotations[:,-2:]
                
        uniprot_ids = []; ecogene_ids = []; ncbigene_ids = []; synonyms = []; other_ids= []
        for res, id in annotations:
            if    re.match("uniprot", res):        uniprot_ids.extend([id])
            elif  re.match("ecogene", res):        ecogene_ids.extend([id])
            elif  re.match("ncbigene", res):       ncbigene_ids.extend([id])
            elif  re.match("refseq_synonym", res): synonyms.extend([id])
            else: other_ids.extend([[res,id]])            
            gps.append([gp.getIdAttribute(), gp.getMetaId(), gp.getLabel(), gp.getName(), gp.getSBOTermID(), 
                        uniprot_ids, ecogene_ids, ncbigene_ids, synonyms, other_ids])
    cols   = ["id", "metaid", "label", "name", "sbo", "uniprotIds", "ecogeneIds", 
              "ncbigeneIds", "synonyms", "otherIds"]
    return pd.DataFrame(gps, columns=cols).set_index(keys='id')  


def getObjectives(model):
    objectives = []   
    if model.isPackageEnabled("fbc"):
        lobjs = model.getPlugin("fbc").getListOfObjectives()
        for i in range(0, lobjs.getNumObjectives()):
            obj = lobjs.get(i)
            obj_act  = (lobjs.getActiveObjective() == obj.getId() or False)
            obj_flux = []
            for fobj in obj.getListOfFluxObjectives():
                obj_flux.append([fobj.getReaction(), fobj.getCoefficient()])
            objectives.append([obj.getIdAttribute(), obj_act, obj.getType(), obj_flux])          
    return pd.DataFrame(objectives, columns = ["id", "isActive", "type", "fluxObjectives"]).set_index(keys='id')

###########################
# GROUPS package parameters
def getGroupings(model):
    group_members=[]
    groups_plugin = model.getPlugin("groups")
    if groups_plugin:
        for g in groups_plugin.getListOfGroups():
            for m in g.getListOfMembers():
                group_members.append([g.getIdAttribute(), g.getName(), m.getIdRef()])            
    return pd.DataFrame(group_members, columns=["id", "name", "memberId"]).set_index(keys='id')


##########################
# additional converstion functions
# for each species check if its unit contains 'mole', in which case add Avogadro number to particleFactor vector
#   otherwise add value 1. particleFactors are required for analysis to converte concentrations to particles
def getParticleFactor(df_species, df_unitdefs):
    CONST_AVOGADRO = 6.02214086e23   # mol-1
    particleFactor = []
    for subsUnits in df_species['subsUnits']:
        factor = 1
        if subsUnits in df_unitdefs.index:             # check if substanceUnit is a defined unit
            for unit in df_unitdefs['unitsKMSE'].loc[subsUnits]:
                if(unit[0] == 'mole') & (unit[3] == 1):
                    factor = CONST_AVOGADRO * unit[1] * 10**unit[2]  
        else:                                          # unit is a base Unit
            if subsUnits == 'mole':
                factor = CONST_AVOGADRO
        particleFactor.append(factor)
    return particleFactor

def addPredefinedUnits(df_unitdefs):    
    predefined_units = [['substance', 'substance', [['mole'        , 1.0, 0, 1]]],
                        ['volume'   , 'volume'   , [['litre'       , 1.0, 0, 1]]],
                        ['area'     , 'area'     , [['square metre', 1.0, 0, 1]]],
                        ['length'   , 'length'   , [['metre'       , 1.0, 0, 1]]],
                        ['time'     , 'time'     , [['second'      , 1.0, 0, 1]]]]
    df_predefined_units = pd.DataFrame(predefined_units, columns=["id","name", "unitsKMSE"]).set_index(keys='id')
    return df_unitdefs.append(df_predefined_units.loc[df_predefined_units.index.difference(df_unitdefs.index)])

def addDefaultCompUnits(df_compartments):
    default_comp_units = ["","length", "area", "volume" ]
    for idx, rows in df_compartments.iterrows():
        if rows['units'] == '':
            if rows['isSpaceDim']: df_compartments['units'] = default_comp_units[rows['spaceDim']]
            else:                  df_compartments['units'] = 'volume'
    return df_compartments

#####################################
# sbml2dfs                          #
#####################################
def sbml2dfs(model_path):
    
    if not os.path.exists(model_path):
        return {"error"    : "Model file not found: {}".format(model_path)}
    
    reader  = libsbml.SBMLReader()
    sbmldoc = reader.readSBML(model_path)
    if sbmldoc.getNumErrors() > 0: 
        print('XMLError:', sbmldoc.getError(0).getMessage())
        if (sbmldoc.getError(0).getSeverity() >= libsbml.LIBSBML_SEV_ERROR):
            return {"error"    : "XMLError: {}".format(sbmldoc.getError(0).getMessage())} 

    model        = sbmldoc.getModel()
    df_compartments = addDefaultCompUnits(getCompartments(model))
    df_unitdefs     = addPredefinedUnits(getUnitDefinitions(model))
    df_reactions    = getReactions(model)
    df_species      = getSpecies(model)
    df_species['particleFactor'] = getParticleFactor(df_species, df_unitdefs)
    
    return {"modelAnnot"        : getMiriamAnnotation(model),
            "modelInfo"         : getModelInfo(model),
            "species"           : df_species, 
            "reactions"         : df_reactions,
            "geneProducts"      : getGeneProduct(model),
            "groupings"         : getGroupings(model),
            "unitDefinitions"   : df_unitdefs,
            "compartments"      : df_compartments,
            "parameters"        : getParameters(model.getListOfParameters()),
            "objectives"        : getObjectives(model),
            "functions"         : getFunctions(model),
            "rules"             : getRules(model),
            "initialAssignment" : getInitAss(model),
            "events"            : getEvents(model),
            "constraints"       : getConstaints(model),
            "stoichMatrix"      : getStoichMatrix(df_species, df_reactions, sparse = False)
           }

def value2particles(vals, params, species):
    particles = np.zeros(len(species))
    for i in range(len(species)):
        metabolite = species[i,0]
        if species[i,1]: # hasOnlySubstanceUnits' - amounts
            particles[i] = vals[i] * species[i,2]    # times particleFactor
        else:                                  #  - concentrations  
            particles[i] = vals[i] * params[species[i,3]] * species[i,2] #also x compartment size
    return particles

def patricles2value(X, params, species):
    values = np.zeros(len(X))
    for i in range(len(species)):
        metabolite = species[i,0]
        if species[i,1]: # hasOnlySubstanceUnits' - amounts
            values[i] = X[i] / species[i,2]    # times particleFactor
        else:                                   #  - concentrations  
            values[i] = X[i] / (params[species[i,3]] * species[i,2]) #also x compartment size
    return values

unit_prefix = {9:'G', 6:'M', 3:'k', 0:'', -3:'m', -6:'µ', -9:'n', -12:'p', -15:'f'}
unit_short = {'second':'s', 'mole':'mol', 'litre':'l', 'metre':'m' , 'item':'#', 
              'square metre':'m2', 'gram':'g', 'kilogram':'kg' }
def getUnitsShort(units, df_unitdefs):
    units_str = ''   
    if units in unit_short:
        units_str = unit_short[units]
    else:      # units are defined       
        for unit in df_unitdefs['unitsKMSE'].loc[units]: 
            short  = unit_short[unit[0]]  if unit[0] in unit_short.keys()  else ''
            prefix = unit_prefix[unit[2]] if unit[2] in unit_prefix.keys() else ''
            exp    = str(unit[3])         if unit[3] != 1   else ''
            mult   = str(unit[1])+'*'     if unit[1] != 1.0 else ''
            units_str = ' '.join([units_str, mult + prefix + short + exp])
    return units_str.strip()

## not used ??
def getInitialParticles(df_species, df_compartments, df_unitdefs):
    CONST_AVOGADRO = 6.02214086e23   # mol-1
    initialParticles = []
    for idx, row in df_species.iterrows():
        particle_factor = 1
        if subsUnits in df_unitdefs.index:             # check if substanceUnit is a defined unit
            for unit in df_unitdefs['unitsKMSE'].loc[row['subsUnits']]:
                if(unit[0] == 'mole') & (unit[3] == 1):
                    particle_factor = CONST_AVOGADRO * unit[1] * 10**unit[2]
        else:
            if subsUnits == 'mole':
                particle_factor = CONST_AVOGADRO                       
        if(row['isInitialConc']):
            particles = particle_factor * row['initialConc'] * df_compartments['size'].loc[row['compartment']]   
        else:
            particles = particle_factor * row['initialAmount']
        initialParticles.append(particles)
    return initialParticles


