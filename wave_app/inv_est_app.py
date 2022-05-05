'''
Inverse Optimization App. Find X minimizing/maximizing y.

reference:
- GPyOpt API : https://gpyopt.readthedocs.io/en/latest/index.html

Not yet impremented (todo)
- Constraint
- Choose X data type. 'continuous' or 'discrete'
'''

import os
import functools

import h2o_wave
from h2o_wave import main, app, Q, ui, on, handle_on, data
import driverlessai

import pandas as pd
import numpy as np

from GPyOpt.methods import BayesianOptimization

print('h2o_wave version : {}'.format(h2o_wave.__version__))
print('driverless ai version : {}'.format(driverlessai.__version__))


''' List of "q.client" 
dai_connection : DAI Setting (ex: True) 
model_selected : DAI Setting (ex: True) 
initialized : DAI Setting (ex: True) 
error : DAI Setting (ex: None) 
experiment_key : DAI Setting (ex: '2f48f7fc-ffb5-11eb-9071-0242ac110002')
target_col : Target col of the model (ex: 'LIMIT_BAL') 
drop_cols : Cols dropped in the experiment (ex: ['ID', 'MARRIAGE', 'AGE']) 
X_columns: Cols used for the experiment (ex: ['SEX', 'EDUCATION', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'])
df_domain: Domains and datatype for GPyOpt
                col        min       max  unique  missing   data_type
        0      CRIM    0.00632   88.9762     504    False  continuous
        1        ZN    0.00000  100.0000      26    False  continuous
        2     INDUS    0.46000   27.7400      76    False  continuous
        ....
cat_cols : Categorical columns (ex: ['SEX', 'EDUCATION']) 
categorical_levels : levels of cat columns in dict (ex: {'SEX': ['M', 'F'], 'EDUCATION': ['university', 'graduate school', 'high school', nan, 'others'],..})
problem_type: Problem type of experiment. 'classification' or 'regression'
x_picker_name_list : "picker_" is added to q.client.cat_cols. They will be q.client items. (ex: ['picker_SEX', 'picker_EDUCATION',..)
x_domain_name_list : "domein_" is added to q.client.X_columns. They will be q.client items. (ex: ['domain_SEX', 'domain_EDUCATION', 'domain_BILL_AMT1', 'domain_BILL_AMT2', 'domain_BILL_AMT3', ...])
y_min_max : max or min for GpyOpt (ex: 'maximization')
cat_cols_key_list: Mapping categorical colms levels between GPyOpt(Can only handle int) and DAI. (ex. [{0: 'F'}, {0: 'university', 1: 'nan', 2: 'others'}, {0: 'married', 1: 'single', 2: 'married', 3: 'married'}])
domain : list(dict) format of q.client.df_domain (ex: [{'name': 'domain_SEX', 'type': 'continuous', 'domain': (0.01239, 0.01239)}, {'name': 'domain_EDUCATION', 'type': 'continuous', 'domain': (0.01239, 0.01239)}, ...]) 
acquisition_type : Acquisition type of GPyOpt (ex: 'EI')
max_iter : GPyOpt setting (ex: '5') 
max_time : GPyOpt setting (ex: '600') 
result_x_opt: Result of optimization
        SEX   EDUCATION      BILL_AMT1  ...      BILL_AMT4      BILL_AMT5      BILL_AMT6
        0  F  university     697569.666707  ...  310561.714805  101538.526625  355576.130631
result_fx_opt : Optimal y (ex: -600459.1) 
server_path_learning_hist_img : (ex: '/_f/f9c48ba6-45f4-4e0a-8b8e-13b6faefcc58/tmp_convergence.png')
server_path_learning_hist_data: (ex: '/_f/1b4ee1e7-5940-41ec-bec0-5880ee447cd2/tmp_hist_data.csv')
'''

@app('/app_inv')
async def serve(q: Q):
    """This function will route the user based on how they have interacted with the application."""
    print('>>>>>>>>>> App Start <<<<<<<<<<')

    # Set up the application
    if not q.client.initialized:
        await initialize_app_for_new_client(q)

    await handle_on(q)

    print("q.app --> ", q.app)   # Check app status (app level)
    print("q.user --> ", q.user)   # Check user status (browser level)
    print("q.client --> ", q.client) # Check client status (browser tab level)
    print("q.args --> ", q.args)   # Check action from browser

    print('>>>>>>>>>> App End <<<<<<<<<<')
    await q.page.save()


async def initialize_app_for_new_client(q: Q):
    """Setup this Wave application for each browser tab by creating a page layout and setting any needed variables"""
    print('INIT FUNCTION initialize_app_for_new_client')

    if not q.user.initialized:
        await initialize_app_for_new_user(q)

    q.page['meta'] = ui.meta_card(
        box='',
        title='Inverse Estimation App',
        theme='light',    # 'light' or 'neon'
    )

    q.page['header'] = ui.header_card(
        box='1 1 10 1',    # loc_x loc_y width height
        title='Inverse Estimation of trained Driverless AI model',
        subtitle='Estimate optimal X minimizing/maximizing y.',
        icon='Variable',
    )

    q.client.dai_connection = False
    q.client.model_selected = False
    render_sidebar_content(q)

    q.client.initialized = True


##### -------------------- Initializer -------------------- #####

async def initialize_app_for_new_user(q: Q):
    """Setup this Wave application for a new user for the first time"""
    print('INIT FUNCTION initialize_app_for_new_user')
    if not q.app.initialized:
        await initialize_app(q)
    q.user.initialized = True


async def initialize_app(q: Q):
    """Setup this Wave application for the first time"""
    print('INIT FUNCTION initialize_app')
    q.app.initialized = True


##### -------------------- Contents Render -------------------- #####

def render_sidebar_content(q: Q):
    """ 
    [Side Bar] contents
    """

    if not q.client.dai_connection:    # if not connected to DAI
        sidebar_items = get_dai_configure_items(q)
    elif not q.client.model_selected:  # after DAI connection, but model is not selected
        sidebar_items = get_model_selection_items(q)
    else:                              # after a model is selected
        sidebar_items = get_model_info(q)

    q.page['sidebar'] = ui.form_card(
        box='1 2 2 14',    # loc_x loc_y width height
        items=[
                  ui.stepper(name='almost-done-stepper', items=[
                      ui.step(label='Connect', done=q.client.dai_connection),
                      ui.step(label='Select', done=q.client.model_selected),
                  ])
              ] + sidebar_items
    )


def render_opt_setting_content(q: Q):
    """ 
    [Optimization Settings] contents
    """
    print('MAIN CONTENT RENDER FUNCTION render_opt_setting_content')

    get_data_info(q)

    q.page['opt_setting'] = ui.form_card(
        box='3 2 4 14',    # loc_x loc_y width height
        items=[
            ui.text_xl('Optimization Settings'),
            ui.separator('Target Variable (y)'),
            ui.text_l('Model target variable: {}'.format(q.client.target_col)),
            ui.choice_group(name='y_min_max', label='Select Minimization or Maximization', value='minimization', required=True, choices=[ui.choice('minimization', 'Minimization'), ui.choice('maximization', 'Maximization')]),
            ui.separator('Input Variables (X)'),
            ui.text_l('<b>Categorical columns</b>'),
            ui.text_m('Select levels of categorical columns to search. ([nan] is missing value in training data.)'),
            ] + get_X_cat_items(q) + [
            ui.text_l('<b>Numerical columns</b>'),
            ui.text_m('Set input domains of numerical columns. (Range to search optimal points.)'),
            ui.text_s('Default is min and max values of training data.'),
            ] + get_X_domain_items(q) + [
            ui.separator('Other settings'),
            ui.dropdown(name='acquisition_type', label='acquisition type', value='EI', choices=[ui.choice('EI', 'Expected Improvement'), ui.choice('MPI', 'Maximum Probability of Improvement'), ui.choice('LCB', 'GP Confidence Bound')]),
            ui.textbox(name='max_iter', label='max iteration', value='15'),
            ui.textbox(name='max_time', label='max run time (sec)', value='600'),
            ui.text_xs('max iteration and max run time exclude 5 initial random scorings'),
            ui.buttons([ui.button(name='run_optimization_button', label='Optimize', primary=True)], justify='center'),]
    )


def render_result_content(q: Q):
    """ 
    [Optimization Result] contents
    """
    print('MAIN CONTENT RENDER FUNCTION render_result_content')

    # formatting result(optimal X)
    df_res_X = q.client.result_x_opt    # pandas.DataFrame
    df_res_X = df_res_X.transpose().reset_index()
    df_res_X.columns = ['col', 'val']
    # positive/negative adjustment of optimal y 
    if q.client.y_min_max=='maximization':
        res_y_val = -1 * q.client.result_fx_opt
    else: # if q.client.y_min_max==''minimization'
        res_y_val = q.client.result_fx_opt
    
    opt_setting_X_display = []
    for i, col in enumerate(q.client.X_columns):
        if col in q.client.cat_cols:    # don't show categorical cols
            continue
        opt_setting_X_display.append(q.client.domain[i])

    q.page['opt_result'] = ui.form_card(
        box='7 2 4 14',    # loc_x loc_y width height
        items=[
            ui.text_xl('Optimization Result'),
            ui.separator('Result'),
            ui.text_l('Optimal X: '),
            ui_table_from_df(df=df_res_X, name='Optimal X', downloadable=True, height='400px'),
            ui.text_l('Value of y: {}'.format(res_y_val)),
            ui.separator('Optimization History'),
            ui.text('![learning_history]({})'.format(q.client.server_path_learning_hist_img)),
            ui.link(label="【Download Historical Result】", download=True, path=q.client.server_path_learning_hist_data),
            ]
    )


##### -------------------- Button Actions -------------------- #####

@on('dai_connect_button')    # [Side Bar] DAI "Connect" button action
async def handle_dai_connection(q: Q):
    print('HANDLE FUNCTION handle_dai_connection')

    q.user.dai_url = q.args.dai_url
    q.user.dai_username = q.args.dai_username
    q.user.dai_password = q.args.dai_password

    _, q.client.error = create_dai_connection(q)   # 'Client' class of DAI
    if q.client.error is None:
        q.client.dai_connection = True

    render_sidebar_content(q)


@on('select_model_button')    # [Side Bar] model "Select" button action
async def handle_model_selection(q: Q):
    print('HANDLE FUNCTION handle_model_selection')

    q.client.experiment_key = q.args.experiment_dropdown    # DAI model id
    q.client.model_selected = True

    render_sidebar_content(q)
    render_opt_setting_content(q)


@on('run_optimization_button')    # [Optimization Settings] "Optimization" button action
async def handle_optimization_setting(q: Q):
    print('HANDLE FUNCTION handle_optimization_setting')

    # assign optimization settings on q.client
    q.client.y_min_max = q.args.y_min_max   # BayesianOptimization parameter
    #(todo) if column type is 'discrete' ('continuous' or 'discrete')
    q.client.df_domain['data_type'] = pd.Series(q.client.df_domain['data_type']).map({'real':'continuous', 'int':'continuous', 'str':'categorical'})  # Replacing DAI type to GPyOpt data type (dai type:gpyopt type)

    # Create categorical colms levels mapping dict
    cat_cols_key_list = []
    for cat_col in q.client.x_picker_name_list:
        cat_cols_key_list.append({k:v for k,v in enumerate(q.args[cat_col])})
    q.client.cat_cols_key_list = cat_cols_key_list

    # Create a list of domains of input variables. Passed to BayesianOptimization(domain=). 
    domain_list = []
    for col,dcol in zip(q.client.X_columns, q.client.x_domain_name_list):
        if col in q.client.cat_cols:
            n_levels = len(q.args['picker_'+str(col)])
            domain_list.append({'name':dcol, 'type':'categorical', 'domain':[i for i in range(n_levels)]})
        else:
            domain_list.append({'name':dcol, 'type':'continuous', 'domain':q.args[dcol]})
    q.client.domain = domain_list

    q.client.acquisition_type = q.args.acquisition_type   # BayesianOptimization parameter
    q.client.max_iter = q.args.max_iter   # BayesianOptimization.run_optimization parameter
    q.client.max_time = q.args.max_time   # BayesianOptimization.run_optimization parameter

    run_optimization(q)
    q.client.server_path_learning_hist_img, = await q.site.upload(['./tmp_convergence.png'])
    q.client.server_path_learning_hist_data, = await q.site.upload(['./tmp_hist_data.csv'])

    render_result_content(q)


##### -------------------- Handle DAI -------------------- #####

def create_dai_connection(q):
    """ 
    Connect DAI and return a 'Client' object. 
    http://docs.h2o.ai/driverless-ai/pyclient/docs/html/client.html
    """
    try:
        conn = driverlessai.Client(q.user.dai_url, q.user.dai_username, q.user.dai_password, verify=False)
        return conn, None
    except Exception as ex:
        return None, str(ex)


def get_dai_configure_items(q: Q):
    """ 
    [Side Bar] sub content
    step1: Get config to connect DAI
    """
    if q.client.error is not None:
        dai_message = ui.message_bar(type='error', text=f'Connection Failed: {q.client.error}')
    elif q.client.dai_conn is None:
        dai_message = ui.message_bar(type='warning', text='This app is not connected to DAI!')
    else:
        dai_message = ui.message_bar(type='success', text='This app is connected to DAI!')

    dai_connection_items = [
        ui.separator('Connect to DAI'),
        dai_message,
        ui.textbox(name='dai_url', label='Driverlss AI URL', value=q.user.dai_url, required=True),
        ui.textbox(name='dai_username', label='Driverless AI Username', value=q.user.dai_username, required=True),
        ui.textbox(name='dai_password', label='Driverless AI Password', value=q.user.dai_password, required=True,
                   password=True),
        ui.buttons([ui.button(name='dai_connect_button', label='Connect', primary=True)], justify='center')
    ]
    return dai_connection_items


def get_model_selection_items(q: Q):
    """
    [Side Bar] sub content
    step2: Select model from connected DAI
    """
    dai, error = create_dai_connection(q)   # 'Client' class of DAI

    ui_choice_experiments = [ui.choice(d.key, d.name) for d in dai.experiments.list()]
    model_selection_items = [
        ui.separator('Select a Model'),
        ui.dropdown(name='experiment_dropdown', label='Driverless AI Models', required=True,
                    choices=ui_choice_experiments, value=q.client.experiment_key),
        ui.buttons([ui.button(name='select_model_button', label='Select', primary=True)], justify='center')
    ]
    return model_selection_items


def get_model_info(q: Q):
    """
    [Side Bar] sub content 
    step3(all enter is done): Show model info
    """
    dai, error = create_dai_connection(q)   # 'Client' class of DAI
    experiment = dai.experiments.get(q.client.experiment_key)

    model_info = [
        ui.separator('Model is selected'),
        ui.text_l('Experiment: {}'.format(experiment)),   # Experiment name and id
        ui.text_l('Training data: {}'.format(experiment.datasets['train_dataset'])),   # Training data name and id
    ]
    return model_info


def get_data_info(q):
    '''
    Assign training data info on q.client.
    q.client.target_col, q.client.X_columns, q.client.df_domain, q.client.cat_cols, q.client.categorical_levels
    '''
    print('DAI FUNCTION get_data_info')

    dai, error = create_dai_connection(q)   # 'Client' class of DAI
    experiment = dai.experiments.get(q.client.experiment_key)   # 'Experiment' class of DAI
    
    ## get target column name and input column names
    # get target column of the model
    q.client.target_col = experiment.settings['target_column']
    # get droped columns list
    try:
        q.client.drop_cols = experiment.settings['drop_columns']     # dropped columns in a list
    except KeyError:
        q.client.drop_cols = []     # if no dropped column
    # make input columns list
    data_cols = experiment.datasets['train_dataset'].columns   # all columns of chosen experiment
    col_no_show = q.client.drop_cols.copy()
    col_no_show.append(q.client.target_col)
    col_show = []
    for col in data_cols:
        if col in col_no_show: 
            continue
        col_show.append(col)
    q.client.X_columns = col_show    # input columns used for the model (no dropped and no target columns)

    ## make pandas.DataFramme of inpt column name, minimum and maximum of training data. Used for optimization setting
    dataset = dai.datasets.get(key=experiment.datasets['train_dataset'].key)   # 'Dataset' class of DAI

    min_list = [col_summary.min for col_summary in dataset.column_summaries()[col_show]]
    max_list = [col_summary.max for col_summary in dataset.column_summaries()[col_show]]
    uniq_list = [col_summary.unique for col_summary in dataset.column_summaries()[col_show]]
    miss_list = [col_summary.missing>0 for col_summary in dataset.column_summaries()[col_show]]
    type_list = [col_summary.data_type for col_summary in dataset.column_summaries()[col_show]]

    q.client.df_domain = pd.DataFrame({'col':col_show, 'min':min_list, 'max':max_list, 'unique':uniq_list, 'missing':miss_list, 'data_type':type_list})

    q.client.cat_cols = q.client.df_domain[q.client.df_domain['data_type']=='str']['col'].values.tolist()

    # if there is any categorial column, get actual class names for each categorical columns
    if len(q.client.cat_cols) != 0:
        dataset.download(dst_file='tmp_traindata.csv', overwrite=True)
        traindata = pd.read_csv('tmp_traindata.csv')
        categorical_levels = {}
        for col in q.client.cat_cols:
            categorical_levels[col] = traindata[col].unique().tolist()
        if os.path.exists('tmp_traindata.csv'):
            os.remove('tmp_traindata.csv')
        q.client.categorical_levels = categorical_levels
    else:  # if no categorical variable
        q.client.categorical_levels = {}
    
    # Problem type of experiment. 'classification' or 'regression'
    q.client.problem_type = experiment.settings['task']


def get_X_domain_items(q):
    '''
    [Optimization Settings] sub content 
    Create a list of ui.range_slider() 
    '''
    print('DAI FUNCTION get_X_domain_items')

    domain_list = []
    x_domain_name_list = []
    for _, row in q.client.df_domain.iterrows():
        colm = row[0]
        x_domain_name = 'domain_' + colm
        x_domain_name_list.append(x_domain_name)
        if colm in q.client.cat_cols:   # Skip if categorical col
            continue
        def_min = row[1]                  # deafault low value
        def_max = row[2]                  # deafault high value
        x_range = row[2] - row[1]
        #(todo) if column type is 'discrete'
        x_step = x_range / 100
        x_min = def_min - (x_range / 4)   # lowest limit
        x_max = def_max + (x_range / 4)   # highest limit
        domain_list.append(ui.range_slider(name=x_domain_name, label=colm, min=x_min, max=x_max, step=x_step, min_value=def_min, max_value=def_max))
        
    q.client.x_domain_name_list = x_domain_name_list

    return domain_list


def get_X_cat_items(q):
    '''
    [Optimization Settings] sub content 
    Create picker for categorical variables
    '''
    print('DAI FUNCTION get_X_cat_items')

    cat_level_list = []
    x_picker_name_list = []
    for col,lvls in q.client.categorical_levels.items():
        level_name = 'picker_'+col
        x_picker_name_list.append(level_name)
        #(todo) if want to replace 'nan' to be other word. lvls may include np.nan
        cat_level_list.append(ui.picker(name=level_name, label=col, required=True, values=[str(i) for i in lvls], choices=[ui.choice(name=str(i), label=str(i)) for i in lvls]))

    q.client.x_picker_name_list = x_picker_name_list

    if len(q.client.cat_cols)==0:
        #return [ui.text_m('-> There is no categorial column. No need to set..' '<font color="orange">There is no categorial column. No need to set..</font>')]
        return [ui.text_m('<font color="orange">There is no categorial column. No need to set..</font>')]
    
    return cat_level_list


def run_optimization(q):
    '''
    run GPyOpt using DAI model
    require q.client.X_columns), q.client.y_min_max and q.client.domain
    '''
    print('DAI FUNCTION run_optimization')

    dai, error = create_dai_connection(q)
    experiment = dai.experiments.get(q.client.experiment_key)   # 'Experiment' class

    def daimodel(x: np.array, dai: driverlessai._core.Client, experiment: driverlessai._experiments.Experiment, cat_cols: list, cat_cols_key_list: dict) -> np.array:
        ''' 
        This function does DAI Scoring. And we want to find optimal x to minimize or maximize daimodel(x)
        Only regression model (classification model is not implemented)
        '''
        #print(type(x))
        #print(x.shape)
        #print(x)
        # Create a scoring data for DAI
        df = pd.DataFrame(np.array([x[:,i] for i in range(len(q.client.X_columns))]).reshape(x.shape[0], len(q.client.X_columns)), columns=q.client.X_columns)
        #print('***** check df in daimodel function *****', df)
        # Replace int to str, which is a level of categorical variable in DAI
        # ex. cat_cols_key_list:[{0: 'F'}, {0: 'university', 1: 'nan', 2: 'others'}, {0: 'married', 1: 'single', 2: 'married', 3: 'married'}]
        for i,cat_col in enumerate(cat_cols):
            df[cat_col] = df[cat_col].map(cat_cols_key_list[i])
        #print('***** check df in daimodel function after replacing *****', df)
        
        # Upload it to DAI
        df.to_csv('tmp_pred.csv', index=False)
        data_to_predict = dai.datasets.create(data='./tmp_pred.csv', data_source='upload', name='tmp_pred.csv', force=True)
        # Scoring on the new dataset and download it
        dai_predictions = experiment.predict(dataset=data_to_predict, include_columns=data_to_predict.columns)
        data_to_predict.delete()
        # Download the scored data and return the prediction
        dai_predictions.download(dst_dir='', dst_file='tmp_res.csv', overwrite=True)
        df_res = pd.read_csv('tmp_res.csv')
        print('---------- Done DAI Scoring ----------')
        if q.client.problem_type == 'classification':
            return np.array(df_res[experiment.settings['target_column']+'.1'])    # return only prediction(probability of 1)
        else:   # if 'regression'
            return np.array(df_res[experiment.settings['target_column']])    # return only prediction

    daimodel_gpyopt = functools.partial(daimodel, dai=dai, experiment=experiment, cat_cols=q.client.cat_cols, cat_cols_key_list=q.client.cat_cols_key_list)

    if q.client.y_min_max=='maximization':
        maximize = True
    else: # if q.client.y_min_max=='minimization'
        maximize = False
    
    myBopt = BayesianOptimization(f=daimodel_gpyopt, domain=q.client.domain, constraints=None, acquisition_type=q.client.acquisition_type, maximize=maximize)  # if maximize=True, solve to minimize -f 
    #(todo) constraints
    myBopt.run_optimization(max_iter=int(q.client.max_iter), max_time=int(q.client.max_time))

    # result of optimal X
    df_opt_x = pd.DataFrame(myBopt.x_opt.reshape(1,len(q.client.X_columns)), columns=q.client.X_columns)
    # replacing int to str, which is a level of categorical variable in DAI
    for i,cat_col in enumerate(q.client.cat_cols):
        df_opt_x[cat_col] = df_opt_x[cat_col].map(q.client.cat_cols_key_list[i])
    q.client.result_x_opt = df_opt_x
    q.client.result_fx_opt = myBopt.fx_opt    # Value of optimal y. y = f(optimal X)
    
    myBopt.plot_convergence('./tmp_convergence.png')
    
    # historical data of optimization calculation
    df_X = pd.DataFrame(myBopt.X, columns=q.client.X_columns)
    # replacing int to str, which is a level of categorical variable in DAI
    for i,cat_col in enumerate(q.client.cat_cols):
        df_X[cat_col] = df_X[cat_col].map(q.client.cat_cols_key_list[i])

    if q.client.y_min_max=='maximization':
        df_y = pd.DataFrame(myBopt.Y * -1, columns=[q.client.target_col])
    else: # if q.client.y_min_max==''minimization'
        df_y = pd.DataFrame(myBopt.Y, columns=[q.client.target_col])
    
    pd.concat([df_X, df_y], axis=1).to_csv('./tmp_hist_data.csv', index=False)


##### -------------------- Utility Functions -------------------- #####

def ui_table_from_df(
    df: pd.DataFrame,
    name: str = 'table',
    sortables: list = None,
    filterables: list = None,
    searchables: list = None,
    min_widths: dict = None,
    max_widths: dict = None,
    multiple: bool = False,
    groupable: bool = False,
    downloadable: bool = False,
    link_col: str = None,
    height: str = '100%'
) -> ui.table:

    ''' utility function to display pandas.DataFrame '''
    #print(df.head())

    if not sortables:
        sortables = []
    if not filterables:
        filterables = []
    if not searchables:
        searchables = []
    if not min_widths:
        min_widths = {}
    if not max_widths:
        max_widths = {}

    columns = [ui.table_column(
        name=str(x),
        label=str(x),
        sortable=True if x in sortables else False,
        filterable=True if x in filterables else False,
        searchable=True if x in searchables else False,
        min_width=min_widths[x] if x in min_widths.keys() else None,
        max_width=max_widths[x] if x in max_widths.keys() else None,
        link=True if x == link_col else False
    ) for x in df.columns.values]

    try:
        table = ui.table(
            name=name,
            columns=columns,
            rows=[
                ui.table_row(
                    name=str(i),
                    cells=[str(df[col].values[i]) for col in df.columns.values]
                ) for i in range(df.shape[0])
            ],
            multiple=multiple,
            groupable=groupable,
            downloadable=downloadable,
            height=height
        )
    except Exception:
        print(Exception)
        table = ui.table(
            name=name,
            columns=[ui.table_column('x', 'x')],
            rows=[ui.table_row(name='ndf', cells=[str('No data found')])]
        )

    return table
