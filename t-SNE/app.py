import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from dash import Dash, dcc, html, Input, Output
from import_data import tsne_data

# Import data and create Pandas dataframe
x, y, coarse_labels, coarse_categories, images = tsne_data(N_IMAGES_PER_CLASS=15)

data = {
    't_sne_x': x,
    't_sne_y': y,
    'coarse_label': coarse_labels,
    'coarse_category': coarse_categories,
    'image': images.tolist()
}

df = pd.DataFrame(data)

# Create app
app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])

app.layout = dbc.Container([
    html.H3('Interactive t-SNE plot of CIFAR-10 dataset'),
    dbc.Row([
        dbc.Col(dcc.Graph(id="scatter-plot"), width=8),
        dbc.Col([
            # TODO:USE FINE LABELS TO CREATE AN IMAGE DESCRIPTION
            html.H6("Image Description", style={'text-align': 'center'}),
            html.Div(html.Img(id='clicked-image', style={'height': '40%', 'width': '40%', 'display': 'block', 'margin': '0 auto'}))
        ], width=4),
    ]),
    dbc.Row([
        html.H5("Number of points to show:", style={'text-align': 'center'}),
        dcc.Slider(
            id='max-slider',
            min=0, max=len(df.index), step=1,
            marks={0: '0', len(df.index): f'{len(df.index)}'},
            value=10,  # Initial value for the number of points to show
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ]),
])

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


@app.callback(
    Output("scatter-plot", "figure"),
    Input("max-slider", "value"))
def update_scatter_plot(num_points_high):
    num_points_low = 0  # Fixed low point
    num_points_to_show = num_points_high - num_points_low

    selected_points = df.sample(n=num_points_to_show)

    fig = px.scatter(selected_points, x='t_sne_x', y='t_sne_y',
                     color='coarse_category',
                     hover_data=['coarse_category'],
                     custom_data=['image'],
                     labels={
                         'coarse_category': 'Coarse Category',
                         't_sne_x': 't-SNE first dimension',
                         't_sne_y': 't-SNE second dimension'},
                     color_discrete_sequence=px.colors.qualitative.Light24)

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


app.run_server(debug=True)


