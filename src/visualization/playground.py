import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
import sys
import datetime
from datetime import date
sys.path.append('src/visualization')
sys.path.append('src/data')
from update_plot_playground import update_plot_playground
def playground():
    #Figure placeholder
    fig = plt.figure(figsize=(10, 6))

    # Create DatePicker widgets for start and end dates
    text_above_dates = widgets.HTML(value='<h3>Select the dates for the analysis (training + test set). No dates earlier than 2000-01-01</h3>')
    start_date_widget = widgets.DatePicker(description='Start Date', disabled=False, value = datetime.date(2000, 1, 1))
    end_date_widget = widgets.DatePicker(description='End Date', disabled=False, value = datetime.date(2023,12,8))
    dates_vbox = widgets.VBox([text_above_dates, start_date_widget, end_date_widget])
    standard_start_date = datetime.date(2000, 1, 1)
    standard_end_date = datetime.date(2023,12,8)
    current_date = datetime.date.today()

    # Create three models with initial parameters
    models = {
        'Decision Tree Classifier': {'active': True,'long_only': False, 'max_depth': 10},
        'Random Forest Classifier': {'active': True,'long_only': False, 'trees': 10, 'max_depth': 10, 'leaves': 10},
        'Support Vector Machine': {'active': True,'long_only': False},
    }

    # Create a dropdown widget to select the model
    text_above_dropdown = widgets.HTML(value='<h3>Select which models are active and model-specific parameters</h3>')
    model_dropdown = widgets.Dropdown(
        options=list(models.keys()),
        value='Decision Tree Classifier',
        description='Select Model:'
    )
    dropdown_vbox = widgets.VBox([text_above_dropdown, model_dropdown])

    #Create widget and box for SP500
    text_above_sp500 = widgets.HTML(value='<h3>Choose whether to include S&P500 as a predictive variable</h3>')
    sp500_widget = widgets.Checkbox(value=True, description='Include S&P500')
    sp500_vbox = widgets.VBox([text_above_sp500, sp500_widget])

    #Create widget and box for model lag
    text_above_lag = widgets.HTML(value='<h3>Choose the prediction lag of the models</h3>')
    lag_widget = widgets.FloatSlider(description='Lag:', min=1, max=20, step=1, value=1)
    lag_vbox = widgets.VBox([text_above_lag, lag_widget])

    #Create widget and box for training set size
    text_above_training_size = widgets.HTML(value='<h3>Choose the relative training data set size (5-95%) </h3>')
    training_size_widget = widgets.FloatSlider(description='Training size:', min=0.05, max=0.95, step=0.05, value=0.75)
    training_size_vbox = widgets.VBox([text_above_training_size, training_size_widget])

    #Create widget and box for random seed selection
    text_above_random_seed = widgets.HTML(value='<h3>Choose whether to use a random seed for the training of the models</h3>')
    random_seed_widget = widgets.Checkbox(value=False, description='Randomize seed')
    random_seed_vbox = widgets.VBox([text_above_random_seed, random_seed_widget])

    #Create currency-pair widgets
    usdeur_widget = widgets.Checkbox(value=True, description='USDEUR')
    usdjpy_widget = widgets.Checkbox(value=True, description='USDJPY')
    usdgbp_widget = widgets.Checkbox(value=True, description='USDGBP')
    usdchf_widget = widgets.Checkbox(value=True, description='USDCHF')
    usdaud_widget = widgets.Checkbox(value=True, description='USDAUD')
    usdcad_widget = widgets.Checkbox(value=True, description='USDCAD')
    usdnzd_widget = widgets.Checkbox(value=True, description='USDNZD')
    usdsek_widget = widgets.Checkbox(value=True, description='USDSEK')
    usdsgd_widget = widgets.Checkbox(value=True, description='USDSGD')
    usdnok_widget = widgets.Checkbox(value=True, description='USDNOK')
    currency_widgets = [usdeur_widget,usdjpy_widget,usdgbp_widget,usdchf_widget,
                        usdaud_widget,usdcad_widget,usdnzd_widget,usdsek_widget,
                        usdsgd_widget,usdnok_widget]

    #Create three Horizontal boxes to hold currency-pair widgets and a vertical box to hold the horizontal boxes
    currencies = ['USDEUR', 'USDJPY', 'USDGBP', 'USDCHF', 'USDAUD', 'USDCAD', 
              'USDNZD', 'USDSEK', 'USDSGD', 'USDNOK']
    text_above_currencies = widgets.HTML(value='<h3>Select the currencies to be used as predictive variables</h3>')
    currency_hbox1 = widgets.HBox([usdeur_widget, usdjpy_widget, usdgbp_widget, usdchf_widget])
    currency_hbox2 = widgets.HBox([usdaud_widget, usdcad_widget, usdnzd_widget, ])
    currency_hbox3 = widgets.HBox([usdsek_widget, usdsgd_widget, usdnok_widget])
    currency_vbox = widgets.VBox([ text_above_currencies, currency_hbox1, currency_hbox2, currency_hbox3])
    currency_vbox.add_class('checkbox-box')
    # Add custom CSS to create a box around the checkboxes
    style = """
    .checkbox-box {
        border: 2px solid #000;
        padding: 10px;
        margin: 10px;
    }
    """
    # Apply the CSS style
    widgets.HTML(value=f'<style>{style}</style>')


    #Output widget to control the output under dropdown list
    output_widget = widgets.Output()

    #Widgets 'unique' to their model
    dtc_active_widget = widgets.Checkbox(value=True, description='Activated')
    dtc_long_only_widget = widgets.Checkbox(value=False, description='Long only')
    dtc_max_depth_widget = widgets.FloatSlider(description='Max depth:', min=1, max=20, step=1, value = 10)

    rfc_active_widget = widgets.Checkbox(value=True, description='Activated')
    rfc_long_only_widget = widgets.Checkbox(value=False, description='Long only')
    rfc_trees_widget = widgets.FloatSlider(description='Trees:', min=1, max=100, step=1, value = 30)
    rfc_max_depth_widget = widgets.FloatSlider(description='Max depth:', min=1, max=20, step=1, value = 10)
    rfc_leaves_widget = widgets.FloatSlider(description='Leaves:', min=1, max=20, step=1, value = 10)

    svm_active_widget = widgets.Checkbox(value=True, description='Activated')
    svm_long_only_widget = widgets.Checkbox(value=False, description='Long only')

    #Button widget
    submit_button_widget = widgets.Button(description = 'Submit')

    #Widget for plot
    plot_output_widget = widgets.Output()

    #Accuracies widgets and box
    dtc_accuracy_widget = widgets.Output(value = None)
    rfc_accuracy_widget = widgets.Output(value = None)
    svm_accuracy_widget = widgets.Output(value = None)
    accuracy_vbox = widgets.VBox([dtc_accuracy_widget, rfc_accuracy_widget, svm_accuracy_widget])


    # Display widgets
    with output_widget:
        display(dtc_active_widget, dtc_long_only_widget, dtc_max_depth_widget)

    display(dates_vbox,sp500_vbox, lag_vbox, training_size_vbox, random_seed_vbox, currency_vbox, dropdown_vbox, output_widget, submit_button_widget,plot_output_widget)
    def update_widgets(change):
        with output_widget:
            # Clear the previous output
            selected_model_name = model_dropdown.value
            selected_model_params = models[selected_model_name]
            if selected_model_name == 'Decision Tree Classifier':   
                dtc_active_widget.value = selected_model_params['active']
                dtc_long_only_widget.value = selected_model_params['long_only']
                dtc_max_depth_widget.value = selected_model_params['max_depth']
                clear_output(wait=True) 
                display(dtc_active_widget,dtc_long_only_widget,dtc_max_depth_widget)     
            elif selected_model_name == 'Random Forest Classifier':
                rfc_active_widget.value = selected_model_params['active']
                rfc_long_only_widget.value = selected_model_params['long_only']
                rfc_trees_widget.value = selected_model_params['trees']
                rfc_max_depth_widget.value = selected_model_params['max_depth']
                rfc_leaves_widget.value = selected_model_params['leaves']
                clear_output(wait=True)
                display(rfc_active_widget,rfc_long_only_widget,rfc_trees_widget,rfc_max_depth_widget,rfc_leaves_widget)
            elif selected_model_name == 'Support Vector Machine':   
                svm_active_widget.value = selected_model_params['active']
                svm_long_only_widget.value = selected_model_params['long_only']

                clear_output(wait=True)
                display(svm_active_widget,svm_long_only_widget)
            else:
                print('Error when selecting widget in dropdown list.')       
    

    # Attach the update function to the dropdown's change event
    model_dropdown.observe(update_widgets, names='value')


    # Add a callback to update the models dictionary when a parameter changes
    def update_model_params(change):
        selected_model_name = model_dropdown.value
        if selected_model_name == 'Decision Tree Classifier':
            models[selected_model_name]['active'] = dtc_active_widget.value
            models[selected_model_name]['long_only'] = dtc_long_only_widget.value
            models[selected_model_name]['max_depth'] =  dtc_max_depth_widget.value
        elif selected_model_name == 'Random Forest Classifier':
            models[selected_model_name]['active'] = rfc_active_widget.value
            models[selected_model_name]['long_only'] = rfc_long_only_widget.value
            models[selected_model_name]['trees'] = rfc_trees_widget.value
            models[selected_model_name]['max_depth'] =  rfc_max_depth_widget.value
            models[selected_model_name]['leaves'] = rfc_leaves_widget.value
        elif selected_model_name == 'Support Vector Machine':   
            models[selected_model_name]['active'] = svm_active_widget.value
            models[selected_model_name]['long_only'] = svm_long_only_widget.value
        else:
                print('Error when selecting widget in dropdown list.')   

    def button_clicked(b,fig, output_widget):
        #Put together the data from our widgets and update the plot
        currencies = []
        for currency in currency_widgets:
            if currency.value == True:
                currencies.append(currency.description)
        include_sp500 = sp500_widget.value
        lag = int(lag_widget.value)
        train_size = training_size_widget.value
        random_seed = random_seed_widget.value
        dtc_active = models['Decision Tree Classifier']['active']
        rfc_active = models['Random Forest Classifier']['active']
        svm_active = models['Support Vector Machine']['active']
        dtc_long_only = models['Decision Tree Classifier']['long_only']
        rfc_long_only = models['Random Forest Classifier']['long_only']
        svm_long_only = models['Support Vector Machine']['long_only']
        dtc_max_depth = int(models['Decision Tree Classifier']['max_depth'])
        rfc_max_depth = int(models['Random Forest Classifier']['max_depth'])
        rfc_trees = int(models['Random Forest Classifier']['trees'])
        rfc_leaves = int(models['Random Forest Classifier']['leaves'])
        start_date = start_date_widget.value
        end_date = end_date_widget.value
        if not (dtc_active or rfc_active or svm_active):
            print('At least one model must be set to active, setting Decision Tree Classifier to Active')
            dtc_active = True
            dtc_active_widget.value = True
            models['Decision Tree Classifier']['active'] = True
        if start_date >= end_date :
            print('Start date later than or the same day as end date. Reverting to standard dates.')
            start_date_widget.value = standard_start_date
            end_date_widget.value = standard_end_date
            start_date = standard_start_date
            end_date = standard_end_date
        if start_date < datetime.date(2000,1,1) :
            print('Start date too early, reverting to standard dates.')
            start_date_widget.value = standard_start_date
            end_date_widget.value = standard_end_date
            start_date = standard_start_date
            end_date = standard_end_date
        if end_date > current_date :
            print('End date is in the future, reverting to standard dates.')
            start_date_widget.value = standard_start_date
            end_date_widget.value = standard_end_date
            start_date = standard_start_date
            end_date = standard_end_date
        with plot_output_widget:
            # Clear the previous output
            clear_output(wait=True)
            accuracies = update_plot_playground(currencies, include_sp500, lag, train_size, random_seed, dtc_active, rfc_active, svm_active,
                                   dtc_long_only, rfc_long_only, svm_long_only, dtc_max_depth, rfc_max_depth, rfc_trees,
                                   rfc_leaves, fig, plot_output_widget, start_date, end_date)
            print(accuracies)




        


        

    dtc_active_widget.observe(update_model_params, names='value')
    dtc_long_only_widget.observe(update_model_params, names='value')
    dtc_max_depth_widget.observe(update_model_params, names='value')

    rfc_active_widget.observe(update_model_params, names='value')
    rfc_long_only_widget.observe(update_model_params, names='value')
    rfc_trees_widget.observe(update_model_params, names='value')
    rfc_max_depth_widget.observe(update_model_params, names='value')
    rfc_leaves_widget.observe(update_model_params, names='value')

    svm_active_widget.observe(update_model_params, names='value')
    svm_long_only_widget.observe(update_model_params, names='value')
    submit_button_widget.on_click(lambda b: button_clicked(b, fig, output_widget))

