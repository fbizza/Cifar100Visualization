from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from import_data import tsne_data
import pandas as pd

# Import data and create Pandas dataframe
x, y, coarse_labels, coarse_categories, images = tsne_data(N_IMAGES_PER_CLASS=5)

data = {
    't_sne_x': x,
    't_sne_y': y,
    'coarse_label': coarse_labels,
    'coarse_category': coarse_categories,
    'image': images.tolist()
}
df = pd.DataFrame(data)

# Create app
app = Dash(__name__)

app.layout = html.Div([
    html.H4('Interactive t-SNE plot of CIFAR-10 dataset'),
    dcc.Graph(id="scatter-plot"),
    html.P("Number of points to show:"),
    dcc.Slider(
        id='max-slider',
        min=0, max=100, step=1,
        marks={0: '0', 100: '100'},
        value=10  # Initial value for the number of points to show
    ),
])


# Callbacks
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
                     labels={
                         'coarse_category': 'Coarse Category',
                         't_sne_x': 't-SNE first dimension',
                         't_sne_y': 't-SNE second dimension'})

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


app.run_server(debug=True)
