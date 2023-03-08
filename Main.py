"""
@author: Harshil Bhatia
"""
def createRandomShapeList(*args):
    shape_lists = []
    if(len(args) == 3):
        num_shapes,num_instances, classValue = args
        for _ in range(num_instances):
            list_temp = [f'{classValue}{i}' for i in random.sample(range(config.args['max_num_shapes']),num_shapes)]
            shape_lists.append(list_temp)

    if(len(args) == 2):
        num_shapes,num_instances = args
        for _ in range(num_instances):
            list_temp = [f'{i}{j}' for i,j in zip(random.sample(range(config.args['max_num_shapes']),num_shapes),random.sample(range(config.args['max_num_shapes']),num_shapes))]
            shape_lists.append(list_temp)

    return shape_lists

def parseArguments():
    parser = argparse.ArgumentParser(description="Change Config File")
    
    #QPU Arguments
    parser.add_argument("--qpu",default=False,action = 'store_true')
    parser.add_argument("--num_reads",type = int,default = 200)
    parser.add_argument("--num_sweeps",type = int,default = 5)
    # parser.add_argument("--embeddingpath",default = 'EmbeddingOld.p')

    #Instance Arguments
    parser.add_argument("--num_shapes",type = int,default = 10)
    parser.add_argument("--num_instances",type = int,default = 1)
    parser.add_argument("--dataset",default='FAUST',choices = ['FAUST','TOSCA','SMAL'])
    parser.add_argument("--classvalue",default= 3,help = "Class")
    parser.add_argument("--interclass",default = False,action='store_true',help = "inter Class Matchings")
    parser.add_argument("--all",default=False,action = 'store_true',help = 'Perform for All FAUST Instances')

    #Algorithm Arguments
    parser.add_argument("--nrWorst",type = int,default = 80,help = "Number of Worst Vertices")
    parser.add_argument("--steps",type = int,default= 100,help = "Steps")
    parser.add_argument("--dropvars",default='True',action = 'store_false')

    parser.add_argument("--LRLabels",default=True,action='store_false',help = "Not using Left Right labels")
    parser.add_argument('--noise',default= False,action = 'store_true')
    parser.add_argument('--noise_variance',type = float,default = 0.02)
    parser.add_argument('--variableMaster',default= False,action= 'store_true')
    #Storage Arguments
    parser.add_argument("--saveDir",default = '/',help = "Save Directory")
    parser.add_argument("--computeEnergy",default = True,action = "store_false", help = 'remove computing energy')
    parser.add_argument("--loadcorr",default=False,action = 'store_true')
    parser.add_argument("--loadcorrpath",default = '')

    #Testing
    parser.add_argument("--test",default = False,action = 'store_true',help = "test the code")


    args = parser.parse_args()
    args_config = vars(args)

    if(args_config['loadcorr'] == 1):
        if(args_config['loadcorrpath'] == ''):
            print("No path specified")

        args_config['usedescriptors'] = 0 
    else:
        args_config['usedescriptors'] = 1

    if args_config['dataset'] == 'FAUST':
        args_config['vertices'] = 502
        args_config['max_num_shapes'] = 10
    if args_config['dataset'] == 'SMAL':
        _cls = args_config['classvalue']
        if _cls == 'cat':
            args_config['num_shapes'] = 9
        elif _cls =='dog':
            args_config['num_shapes'] = 16
        elif _cls =='horse':
            args_config['num_shapes'] = 10
        elif _cls =='cow':
            args_config['num_shapes'] = 8
        elif _cls =='hippo':
            args_config['num_shapes'] = 6

    config.args = args_config

def createSaveDir(exp_number):
    dataset = config.args['dataset']
    config.args['saveDir'] = f'{dataset}/'
    

    classvalue = config.args['classvalue']
    config.args['saveDir'] += f'{classvalue}/'

    if config.args['interclass'] == 1:
        config.args['saveDir'] += f'interclass/'
    
    if config.args['noise'] == 1:
        config.args['saveDir'] += 'var0.02/'

    if config.args['test'] == 1:
        config.args['saveDir'] = '/workspace/data/finalalgo/testdump/'

    utils.CreatePaths(config.args['saveDir'])
    utils.CreatePaths(config.args['saveDir'] + 'ITR/')

def main():

    matching_framework = MatchingFramework.MatchingFramework()
    if config.args['dataset'] == 'FAUST':
        if config.args['all']:
            config.args['shape_lists'] = [[f'{i:02}' for i in range(100)]]
            config.args['num_shapes'] = len(config.args['shape_lists'][0])  
        else:
            if config.args['interclass'] == 1:
                config.args['shape_lists'] = createRandomShapeList(config.args['num_shapes'],config.args['num_instances'])
            else:
                config.args['shape_lists'] = []                    
                temp_shape_list = createRandomShapeList(config.args['num_shapes'],config.args['num_instances'],config.args['classvalue'])
                config.args['shape_lists'].append(temp_shape_list[0])

        for exp_number,shape_list in enumerate(config.args['shape_lists']):

            createSaveDir(exp_number)
            print(config.args)
            config.args['num_shapes'] = len(shape_list)
            
            saveDir = config.args['saveDir']
            pickle.dump(config.args,open(f'{saveDir}args.p','wb'))

            start_time = time.perf_counter()
            matching_framework.match(shape_list)
            end_time = time.perf_counter()

            print('Total Time:',end_time-start_time)

    if config.args['dataset'] == 'TOSCA' or config.args['dataset'] == 'SMAL':

        createSaveDir(0)
        print(config.args)
        
        saveDir = config.args['saveDir']
        pickle.dump(config.args,open(f'{saveDir}args.p','wb'))

        start_time = time.perf_counter()
        matching_framework.match()
        end_time = time.perf_counter()
        print(f'Total Time:',end_time-start_time)
        

if __name__ == '__main__':
    
    import argparse
    import config 
    import pickle
    parseArguments()
    import MatchingFramework
    import utils
    import time
    import random 
    main()

