from bokeh.io import curdoc
import joblib
from bokeh.layouts import gridplot, row, column, layout
from bokeh.models import Select, ColumnDataSource, Div, DateRangeSlider, Slider, HoverTool
from bokeh.plotting import figure
from bokeh.models.widgets import Button, CheckboxGroup
from bokeh.events import ButtonClick
from bokeh.transform import factor_cmap, linear_cmap
from os.path import dirname, join
from src.ulearning import *


# Layout constants
column_width = 700
plot_height = 600

# Create text elements in the app
scatter_div = Div(text=open(join(dirname(__file__), "static/scatter.html")).read(), sizing_mode="stretch_width", width=column_width, height=100)
evaluation_div = Div(text=open(join(dirname(__file__), "static/evaluation.html")).read(), sizing_mode="stretch_width", width=column_width, height=100)
training_div = Div(text=open(join(dirname(__file__), "static/training.html")).read(), sizing_mode="stretch_width", width=column_width, height=100)



df = pd.read_csv('data/top50.csv', encoding='latin-1')
df = clean_columns(df)

unsup_cols = ['genre', 'beatsperminute', 'energy', 'danceability', 'loudnessdb',
              'liveness', 'valence', 'length', 'acousticness', 'speechiness', 'popularity']

clusters = [0 for i in range(len(df))]

# 2D Scatter plot
scat_tooltips = [
    ('x', "@x"),
    ('y', "@y"),
    ('Track name', "@key"),
    ('Cluster', "@cluster")
]

scat_source = ColumnDataSource(data=dict(x=[], y=[], key=[], cluster=[]))
scat_plot = figure(title='2D Scatter plot', plot_width=column_width, plot_height=plot_height, tooltips=scat_tooltips, sizing_mode='scale_width', min_border_right=30)
color_map = linear_cmap(field_name='cluster', palette='Category20_20', low=0, high=1)
scat_plot.scatter(x='x', y='y', marker='circle', size=15, alpha=0.5, source=scat_source, color=color_map)

# Inertia plot
inertia_source = ColumnDataSource(data=dict(n_cluster=[1,2,3], inertia=[4,5,6]))
inertia_plot = figure(title='Inertia plot', plot_width=column_width, plot_height=plot_height, sizing_mode='scale_width')
inertia_plot.line(x='n_cluster', y='inertia', source=inertia_source)
inertia_plot.xaxis.axis_label = 'Number of clusters'


def update():
    dd, nums, cats = generate_dataset(df, keys=['trackname', 'artistname'], keep_columns=unsup_cols, remove_columns=['unnamed:0'])
    #dd, nums, cats = generate_dataset(df, keys=[data_key_select_1.value, data_key_select_2.value], keep_columns=unsup_cols,
    #                                 remove_columns=['unnamed:0'])
    print(dd.columns)
    scat_source.data = dict(
        x=dd[x_select.value],
        y=dd[y_select.value],
        key=dd.index,
        cluster=clusters
    )
    x_select.options = nums
    y_select.options = nums
    color_map['transform'].high = max(clusters)

    scat_plot.xaxis.axis_label = x_select.value
    scat_plot.yaxis.axis_label = y_select.value

    print('Updated')


def calculate_inertia():
    dd = create_analytics_dataframe(df, keys=['trackname', 'artistname'], keep_columns=unsup_cols, remove_columns=['unnamed:0'])
    ncols, inertias = find_optimal_clusters(dd)
    inertia_source.data = dict(
        n_cluster=ncols,
        inertia=inertias,
    )

    print('Updated inertia')


def fit_model():
    model = KMeans(n_clusters=n_cluster_slider.value)
    if data_key_select_2 == 'None':
        keys = [data_key_select_1.value]
    else:
        keys = [data_key_select_1.value, data_key_select_2.value]

    all_columns = df.columns.to_list()
    columns_for_training = [all_columns[i] for i in unsupervised_cols_checkboxgroup.active]
    data_to_fit = create_analytics_dataframe(df, keys=keys, keep_columns=columns_for_training, remove_columns=['unnamed:0'])
    model.fit(data_to_fit)
    # This one needs to be in such a way that we have one analytical frame, and one cluster frame, OR, one cluster serie
    # which we can append to the scatter... maybe the second one is better!
    global clusters
    clusters = model.labels_
    print("fitted the model")


# Scatter Controls
x_select_options = unsup_cols.copy()
x_select_options.append('ALL')
x_select = Select(title='X-axis feature', options=x_select_options, value='beatsperminute')

y_select_options = unsup_cols.copy()
y_select_options.append('ALL')
y_select = Select(title='Y-axis feature', options=x_select_options, value='energy')

controls = [x_select, y_select]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

# Cluster Buttons
cluster_button = Button(label='Plot inertia', button_type='success')
cluster_button.on_event(ButtonClick, calculate_inertia)

# KMeans controls
data_key_options = df.columns.tolist()
data_key_options.append('None')
data_key_select_1 = Select(title='Data Key #1', options=data_key_options, value='None')
data_key_select_2 = Select(title='Data Key #2', options=data_key_options, value='None')
n_cluster_slider = Slider(title='n_clusters=', start=1, end=10, value=3, step=1)
unsupervised_cols_checkboxgroup = CheckboxGroup(labels=df.columns.to_list(), active=[0, 1])
train_button = Button(label='Train the model', button_type='success')
train_button.on_event(ButtonClick, fit_model)

# Create the layout
inputs = row(*controls, width=column_width, height=100, sizing_mode='scale_width')
buttons = row(cluster_button, width=column_width, height=100, sizing_mode='scale_width')
scatter_layout = column(scatter_div, inputs, scat_plot)
inertia_layout = column(evaluation_div, buttons, inertia_plot)
kmeans_layout = column(training_div, data_key_select_1, data_key_select_2, n_cluster_slider, unsupervised_cols_checkboxgroup, train_button)

l = layout([
    [scatter_layout, inertia_layout, kmeans_layout],
],
    sizing_mode='fixed')

update()

curdoc().add_root(l)
curdoc().title = 'Interactive Unsupervised Learning'


