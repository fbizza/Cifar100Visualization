import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import dash
from dash import Dash, dcc, html, Input, Output, State
from import_data import tsne_data

# Import data and create Pandas dataframe
x, y, coarse_labels, coarse_categories, fine_categories, images = tsne_data(N_IMAGES_PER_CLASS=20)

data = {
    't_sne_x': x,
    't_sne_y': y,
    'coarse_label': coarse_labels,
    'coarse_category': coarse_categories,
    'fine_category': fine_categories,
    'image': images.tolist()
}

df = pd.DataFrame(data)

# Create app
app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

app.layout = dbc.Container([
    html.H3('Interactive t-SNE plot of CIFAR-100 dataset'),
    html.H6('Click on single points to display their descriptions', style={'opacity': '0.80'}),
    dbc.Row([
        dbc.Col(dcc.Graph(id="scatter-plot"), width=8),
        dbc.Col([
            html.H6("Image Description", style={'text-align': 'center'}),
            html.Div(html.Img(id='clicked-image', style={'height': '40%', 'width': '40%', 'display': 'block', 'margin': '0 auto'})),
            html.Div(id='image-text-description', style={'text-align': 'center'}),
            # TODO: add here the explanation image from LIME framework (implement relative callback)
            # html.H6("Image Explanation", style={'text-align': 'center'}),
            # html.Div(html.Img(id='explanation-clicked-image', style={'height': '40%', 'width': '40%', 'display': 'block', 'margin': '0 auto'})),
        ], width=4),
    ]),
    dbc.Row([
        html.H5("Number of points to show:", style={'text-align': 'center'}),
        dcc.Slider(
            id='max-slider',
            min=0, max=len(df.index), step=1,
            marks={0: '0', len(df.index): f'{len(df.index)}'},
            value=100,  # Initial value for the number of points to show
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

# Callback for updating image description
@app.callback(
    Output("image-text-description", 'children'),
    Input("scatter-plot", "clickData"))
def show_clicked_image(clickData):
    if clickData:
        fine_category = clickData['points'][0]['customdata'][1]
        return f'This picture shows a {fine_category}'

# Callback for updating scatter plot based on slider number selection and button click
@app.callback(
    Output("scatter-plot", "figure"),
    Input("update-plot-button", "n_clicks"),
    State("max-slider", "value")
)
def update_scatter_plot_on_button_click(n_clicks, value):

    num_points_to_show = value  # Value from slider

    selected_points = df.sample(n=num_points_to_show)

    fig = px.scatter(selected_points, x='t_sne_x', y='t_sne_y',
                     color='coarse_category',
                     hover_data=['coarse_category'],
                     custom_data=['image', 'fine_category'],
                     labels={
                         'coarse_category': 'Coarse Category',
                         't_sne_x': 't-SNE first dimension',
                         't_sne_y': 't-SNE second dimension'},
                     color_discrete_sequence=px.colors.qualitative.Light24)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


app.run_server(debug=True)


