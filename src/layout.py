from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import charts


def get_app_description():
    description_text = '''
        TODO: Write a small paragraph about the dataset and what you are going to show.
        Dash uses the [CommonMark](http://commonmark.org/)
        specification of Markdown.
        Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
        if this is your first introduction to Markdown!
        '''
    return dcc.Markdown(children=description_text)


def get_data_insights():
    insights = '''
        TODO: Write a small paragraph on the general insights based on your charts.
    '''
    return dcc.Markdown(children=insights)


def get_source_text():
    source_text = '''
    Data from [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html).
    '''
    return dcc.Markdown(children=source_text)


def get_exercise1_charts():
    row = html.Div(
        [
            dbc.Row(
                dbc.Col(html.H2("Data Exploration", style={"margin-top": "1em"})),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            #TODO chart 1
                            # get_basic_chart_A() from charts.py
                        ],
                    ),
                    dbc.Col(
                        [
                            #TODO chart 2
                        ],
                    ),
                ],
            ),
        ]
    )

    return row


def get_exercise3_map():
    return dbc.Row(
        dbc.Col(
            [
                html.H2("Choropleth Map", style={"margin-top": "1em"}),
                #TODO
            ],
        )
    )


def get_app_layout():
    return dbc.Container(
        [
            html.H1(children='Data Visualisation',
                    style={"margin-top": "1rem"}),
            get_app_description(),
            get_exercise1_charts(),
            #get_exercise3_map(),
            html.H2(children='Conclusion',
                    style={"margin-top": "1rem"}),
            get_data_insights(),
            dbc.Row(
                [
                    dbc.Col(html.P("Created by Francesco Bizzarri, Stijn Oosterlinck and Justine Rayp")),
                    dbc.Col(get_source_text(), width="auto")
                ],
                justify="between",
                style={"margin-top": "3rem"}),
        ],
        fluid=True
    )
