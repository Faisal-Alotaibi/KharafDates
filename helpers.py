from dash import Input, Output, dcc, html


# ---------- Helper Functions -----------------
def parse_content(contents, filename, date):
    return html.Div([
        html.H5(filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Div(
            [html.Img(src=contents, className="image_preview")],
            
        ),
    ], className="image-preview-div")