import dash
import webbrowser
import threading
import time
from src.layout import create_layout
from src.callbacks import register_callbacks

app = dash.Dash(__name__)
app.title = 'CardioGuard AI'

# Set Layout
app.layout = create_layout()

# Register Interaction Logic
register_callbacks(app)

if __name__ == '__main__':
    threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open("http://127.0.0.1:8050"))).start()
    
    app.run(debug = True, host='127.0.0.1', port = 8050, use_reloader = False)