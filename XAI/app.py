import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import dash
from dash import Dash, dcc, html, Input, Output, State
from import_data import tsne_data
from lime_explainer import lime_explain_img
from explanation_v2 import generate_heatmap_and_plot

# Import data and create Pandas dataframe
(t_sne_softmax_x, t_sne_softmax_y,
 t_sne_first_block_x, t_sne_first_block_y,
 t_sne_second_block_x, t_sne_second_block_y,
 t_sne_third_block_x, t_sne_third_block_y,
 t_sne_fourth_block_x, t_sne_fourth_block_y,
 predicted_fine_categories,
 coarse_labels, coarse_categories, fine_categories, images) = tsne_data(N_IMAGES_PER_CLASS=25)

data = {
    't_sne_softmax_x': t_sne_softmax_x,
    't_sne_softmax_y': t_sne_softmax_y,
    't_sne_first_block_x': t_sne_first_block_x,
    't_sne_first_block_y': t_sne_first_block_y,
    't_sne_second_block_x': t_sne_second_block_x,
    't_sne_second_block_y': t_sne_second_block_y,
    't_sne_third_block_x': t_sne_third_block_x,
    't_sne_third_block_y': t_sne_third_block_y,
    't_sne_fourth_block_x': t_sne_fourth_block_x,
    't_sne_fourth_block_y': t_sne_fourth_block_y,
    'predicted_fine_category': predicted_fine_categories.flatten(),
    'coarse_label': coarse_labels,          # Numbers from 0 to 19
    'coarse_category': coarse_categories,   # Strings
    'fine_category': fine_categories,       # Strings
    'image': images.tolist(),                # Raw pixels values

}

tsne_options = [
    {'label': 'First Convolutional Block Output', 'value': 'first_block'},
    {'label': 'Second Convolutional Block Output', 'value': 'second_block'},
    {'label': 'Third Convolutional Block Output', 'value': 'third_block'},
    {'label': 'Fourth Convolutional Block Output', 'value': 'fourth_block'},
    {'label': 'Softmax', 'value': 'softmax'},
]

df = pd.DataFrame(data)

# Add a column to indicate whether the prediction was correct
df['correct_prediction'] = df['predicted_fine_category'] == df['fine_category']

# Create app
app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

app.layout = dbc.Container([
    html.H3('Interactive t-SNE plot of CIFAR-100 dataset'),
    html.H6('Click on single points to display their descriptions', style={'opacity': '0.80'}),
    dbc.Row([
        html.Div([
            dcc.Dropdown(
                id='tsne-dropdown',
                options=[{'label': option['label'], 'value': option['value']} for option in tsne_options],
                value='softmax',
                clearable=False
            ),
        ], style={'width': '100%', 'margin': '0 auto'}),
    ]),
    dbc.Row([
        html.Div([
            dcc.Dropdown(
                id='pred-dropdown',
                options=[{'label': 'All predictions', 'value': 'all',
                          },
                          {'label': 'Correct predictions', 'value': 'correct',
                          },
                          {'label': 'Wrong predictions', 'value': 'wrong',
                          },],
                value='all', 
                clearable=False
            ),
        ], style={'width': '100%', 'margin': '0 auto'}),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="scatter-plot"), width=8),
        dbc.Col([
            html.H6("Image Description", style={'text-align': 'center'}),
            html.Div(html.Img(id='clicked-image', style={'height': '40%', 'width': '40%', 'display': 'block', 'margin': '0 auto'})),
            html.Div(id='image-text-description', style={'text-align': 'center'}),
            html.Div(id='predicted-class-description', style={'text-align': 'center'}),
            html.Div(dbc.Button("Get explanation", id="explanation-button", style={'width': '40%'}), style={'text-align': 'center'}),
            dbc.Row([
                dbc.Col(html.Div(html.Img(id='explanation-clicked-image', style={'height': '90%', 'width': '90%', 'display': 'block', 'margin': '0 auto'}))),
                dbc.Col(html.Div(html.Img(id='lime-explanation-clicked-image', style={'height': '90%', 'width': '90%', 'display': 'block', 'margin': '0 auto'})))
            ]),

        ], width=4),
    ]),
    dbc.Row([
        html.H5("Number of points to show:", style={'text-align': 'center'}),
        dcc.Slider(
            id='max-slider',
            min=0, max=len(df.index), step=1,
            marks={0: '0', len(df.index): f'{len(df.index)}'},
            value=len(df.index)/2,  # Initial value for the number of points to show
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        dbc.Button("Update Plot", id="update-plot-button", style={'width': '10%', 'margin': '0 auto'}),
        html.H6("Created by Stijn Oosterlinck, Justine Rayp and Francesco Bizzarri",
                style={'margin-top': '2.5em', 'font-size': '0.8em', 'font-weight': 'lighter'})
    ]),
])


# Callback for showing image of clicked point
@app.callback(
    Output("clicked-image", 'src'),
    Input("scatter-plot", "clickData"))
def show_clicked_image(clickData):
    if clickData:
        image_data = np.array(clickData['points'][0]['customdata'][0], dtype='uint8')
        image = image_data.reshape(32, 32, 3)
        plt.imsave("clicked-image.png", image)
        encoded_image = base64.b64encode(open("clicked-image.png", 'rb').read()).decode('ascii')
        return 'data:image/png;base64,{}'.format(encoded_image)
    

# Callback for showing explanation of clicked point
@app.callback(
    Output("explanation-clicked-image", 'src'),
    Input("explanation-button", "n_clicks"),
    State("scatter-plot", "clickData"))
def show_explanation_image(n_clicks, clickData):
    if clickData:
        image_data = np.array(clickData['points'][0]['customdata'][0], dtype='uint8')
        image = image_data.reshape(32, 32, 3)
        generate_heatmap_and_plot(image)
        encoded_image = base64.b64encode(open("cam.jpg", 'rb').read()).decode('ascii')
        return 'data:image/png;base64,{}'.format(encoded_image)

# Callback for showing lime explanation of clicked point
@app.callback(
    Output("lime-explanation-clicked-image", 'src'),
    Input("explanation-button", "n_clicks"),
    State("scatter-plot", "clickData"))
def show_explanation_image(n_clicks, clickData):
    if clickData:
        image_data = np.array(clickData['points'][0]['customdata'][0], dtype='uint8')
        image = image_data.reshape(32, 32, 3)
        image = lime_explain_img(image)
        plt.imsave("lime-explanation-image.png", image)
        encoded_image = base64.b64encode(open("lime-explanation-image.png", 'rb').read()).decode('ascii')
        return 'data:image/png;base64,{}'.format(encoded_image)

# Callback for updating image ground truth description
@app.callback(
    Output("image-text-description", 'children'),
    Input("scatter-plot", "clickData"))
def update_img_description(clickData):
    if clickData:
        fine_category = clickData['points'][0]['customdata'][1]
        return f'This picture shows a {fine_category}'

# Callback for updating image predicted class description
@app.callback(
    Output("predicted-class-description", 'children'),
    Input("scatter-plot", "clickData"))
def update_predicted_class_description(clickData):
    if clickData:
        predicted_category = clickData['points'][0]['customdata'][2]
        return f'The model predicted a {predicted_category}'

# Callback for updating scatter plot based on slider number selection and button click
@app.callback(
    Output("scatter-plot", "figure"),
    Input("update-plot-button", "n_clicks"),
    Input("pred-dropdown", "value"),
    Input("tsne-dropdown", "value"),  # Change from tsne-slider to tsne-dropdown
    State("max-slider", "value")
)
def update_scatter_plot_on_button_click(n_clicks, preds,selected_tsne, value):
    num_points_to_show = value  # Value from slider

    selected_points = df.head(n=num_points_to_show)

    if preds == 'correct':
        selected_points = selected_points[selected_points['correct_prediction'] == True]
    elif preds == 'wrong':
        selected_points = selected_points[selected_points['correct_prediction'] == False]
    # Get unique values in the 'coarse_category' column
    unique_categories = df['coarse_category'].unique()

    # Create a color mapping dictionary using a color sequence
    color_mapping = dict(zip(unique_categories, px.colors.qualitative.Light24[:len(unique_categories)]))

    fig = px.scatter(selected_points, x=f't_sne_{selected_tsne}_x', y=f't_sne_{selected_tsne}_y',
                     color='coarse_category',
                     color_discrete_map=color_mapping,
                     hover_data=['coarse_category'],
                     custom_data=['image', 'fine_category', 'predicted_fine_category'],
                     labels={
                         'coarse_category': 'Coarse Category',
                         f't_sne_{selected_tsne}_x': 't-SNE first dimension',
                         f't_sne_{selected_tsne}_y': 't-SNE second dimension'},)
                    
                     #color_discrete_sequence=px.colors.qualitative.Light24)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig





app.run_server(debug=True)


